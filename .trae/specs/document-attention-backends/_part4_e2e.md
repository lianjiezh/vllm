## 端到端调用链

attention backend 的生命周期分为三个阶段：模型构造时选定 backend 并实例化 `AttentionImpl`、runner 初始化时为每个 attention 分组实例化 `AttentionMetadataBuilder`、每一步执行时在 CPU 侧构造元数据并在 GPU 侧由 impl 消费。本节追踪这条链路上各组件的衔接点。

### 阶段一：模型构造

模型代码按层声明 attention：

```python
self.attn = Attention(num_heads, head_size, scale, ...)
```

进入 [attention.py](../../vllm/model_executor/layers/attention/attention.py) 中 `Attention.__init__` 后，backend 的接入顺序如下：

1. 调用 [selector.py](../../vllm/v1/attention/selector.py) 的 `get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla=False, has_sink=..., use_mm_prefix=..., use_per_head_quant_scales=..., attn_type=...)`，依据当前 `vllm_config.attention_config.backend` 与平台 `get_attn_backend_cls` 选出一个 `AttentionBackend` 子类（选择机制详见 backend 选择机制一章）。
2. 通过 `self.attn_backend.get_impl_cls()` 拿到 impl 类，并以层参数实例化：
   ```python
   self.impl = impl_cls(
       num_heads, head_size, scale, num_kv_heads, alibi_slopes,
       sliding_window, kv_cache_dtype, logits_soft_cap, attn_type,
       kv_sharing_target_layer_name, **extra_impl_args,
   )
   ```
3. 记录枚举别名 `self.backend = AttentionBackendEnum[self.attn_backend.get_name()]`（[registry.py](../../vllm/v1/attention/backends/registry.py)），供 runner 等处用统一枚举引用。
4. 将自身注册到编译期的 `static_forward_context`：`compilation_config.static_forward_context[prefix] = self`，索引键即该层的 `prefix`（层名）。运行时 custom op 通过 `_encode_layer_name(self.layer_name)` 把层名包装为 dispatch key，`get_attention_context` 再解析回层名，从 `ForwardContext.no_compile_layers`（即 `static_forward_context` 的快照）取出对应的 `Attention` 实例。

需要强调的是：同一模型的不同层可能选用不同 backend。例如混合架构中 MLA 层走 MLA backend、普通 full attention 层走 FlashAttention。同一 `kv_cache_group` 内的层共享同一份 KV cache 内存池与 block table；它们通常使用同一 backend，但框架也允许同组内出现多个 backend——每种 backend 各自派生一个独立的 `AttentionGroup` 与 `AttentionMetadataBuilder`（见阶段二）。`kv_cache_group` 划分的是"KV cache 存储与回收的粒度"，而非"backend 的粒度"。

### 阶段二：runner 初始化

[gpu_model_runner.py](../../vllm/v1/worker/gpu_model_runner.py) 的 `initialize_attention_backends` 负责把模型里散落的 attention 层归并为可执行的分组：

1. 对每个 `kv_cache_group`，`get_attn_backends_for_group` 遍历组内所有层，调用 `layer.get_attn_backend()` 取回该层选定的 backend 类，并按 `(full_cls_name, kv_cache_spec, num_heads_q)` 去重，得到该组内的 backend 集合。返回结构中包含 `attention_backends: list[set[type[AttentionBackend]]]`——列表每个元素对应一个 `kv_cache_group`，元素是一个 backend 类的集合，体现"同组内多层可能用不同 backend"。
2. 在真正实例化 builder 之前，先调用 `_check_and_update_cudagraph_mode(attention_backends, kv_cache_groups)`：对每个组的每个 backend，取 `attn_backend.get_builder_cls().get_cudagraph_support(vllm_config, kv_cache_group.kv_cache_spec)`，在所有分组之间取 `min(cg_support)` 作为整体 cudagraph 支持上界，交给 `compilation_config.resolve_cudagraph_mode_and_sizes` 决定最终的 cudagraph mode 与 capture sizes。
3. `create_attn_groups` 为组内每个 backend 创建一个 `AttentionGroup`（绑定 backend 类、层名列表、kv_cache_spec、group id）；随后 `initialize_metadata_builders` 对每个 `AttentionGroup` 调用 `backend.get_builder_cls()(kv_cache_spec, layer_names, vllm_config, device)` 实例化其 `AttentionMetadataBuilder`。

至此，runner 持有 `self.attn_groups: list[list[AttentionGroup]]`（外层按 `kv_cache_group` 索引，内层按 backend 分组），每个 `AttentionGroup` 持有自己的 builder。

### 阶段三：每 step 执行

每一步调度完成后：

1. scheduler 产出请求，runner 构造一个跨层共享的 `CommonAttentionMetadata`（[backend.py](../../vllm/v1/attention/backend.py)），包含 `query_start_loc` / `seq_lens` / `block_table_tensor` / `slot_mapping` / `num_reqs` / `max_query_len` 等字段；不同 `kv_cache_group` 间仅 `block_table_tensor`、`slot_mapping`、`encoder_seq_lens` 等少数字段不同，其余浅拷贝复用。
2. 对每个 `kv_cache_group` 的每个 `AttentionGroup`，调用 `builder.build(common_prefix_len, common_attn_metadata)` 得到该组 per-layer 的 `AttentionMetadata`，并写入 `attn_metadata_dict[layer_name] = attn_metadata_i`——同组内多层共享同一份 metadata。capture 路径则改走 `builder.build_for_cudagraph_capture(common_attn_metadata)`。
3. 该 dict 作为 `attn_metadata` 传入 `set_forward_context`，成为当前 forward 的 `ForwardContext.attn_metadata`（[forward_context.py](../../vllm/forward_context.py)）；`ForwardContext.no_compile_layers` 即 `static_forward_context` 的快照，保存了层名→`Attention` 实例的映射。
4. 模型 `forward` 中每层调用 `self.attn(query, key, value, kv_cache)`，进入 `Attention.forward`。这里 reshape Q/K/V 后，先（视情况）调用 custom op `unified_kv_cache_update`，再调用 `unified_attention_with_output`。后者通过 `get_attention_context(layer_name)` 从 `ForwardContext` 取出该层的 `attn_metadata` 与 `Attention` 实例，dispatch 到 `self.impl.forward(self, query, key, value, kv_cache, attn_metadata, output=...)`。

`forward_includes_kv_cache_update` 字段控制第 4 步中"谁来写 KV cache"：

- 字段语义是"impl 的 `forward` 是否**包含** KV cache 更新"。`AttentionBackend` 基类默认为 `True`（[backend.py](../../vllm/v1/attention/backend.py)）。
- 当为 `False` 时（如 [flash_attn.py](../../vllm/v1/attention/backends/flash_attn.py) 的 `FlashAttentionBackend`、[triton_attn.py](../../vllm/v1/attention/backends/triton_attn.py) 的 `TritonAttentionBackend`），`Attention.forward` 会在调用 impl 之前先触发 `unified_kv_cache_update`，由框架经 `impl.do_kv_cache_update(...)` 执行 `reshape_and_cache_flash` / `concat_and_cache_mla` 把新 K/V 写入 KV cache；随后 `unified_attention_with_output` 只做注意力计算。
- 当为 `True` 时，跳过外部 `unified_kv_cache_update`，由 impl 在 `forward` 内部自行完成 KV cache 更新。

换言之，`False` 表示"需要外部更新"，`True` 表示"impl 自包更新"；两种语义下 `do_kv_cache_update` 都是把新 K/V 散写到 paged KV cache 的统一入口。

## 横切关注点

### KV cache 布局

vLLM 支持两种 KV cache 物理布局，由 [utils.py](../../vllm/v1/attention/backends/utils.py) 的 `KVCacheLayoutType = Literal["NHD", "HND"]` 描述：

- `NHD`：逻辑形状 `(num_blocks, 2, block_size, num_kv_heads, head_size)`，`num_blocks` 位于物理最外层。
- `HND`：逻辑维度相同，但 stride 把 heads 维提到 block 维之前，适合某些 SM100 系 kernel。

相关接口：

- `AttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)` 返回**逻辑形状**。
- `get_kv_cache_stride_order(include_num_layers_dimension=False)` 返回物理排列的置换元组；未实现时物理布局等同于逻辑形状。
- `get_required_kv_cache_layout()` 返回 backend 强制的布局（默认 `None`，即不强制）。例如 [flashinfer.py](../../vllm/v1/attention/backends/flashinfer.py) 在 SM100 上返回 `"HND"`。选中 backend 后，[selector.py](../../vllm/v1/attention/selector.py) 会调用 `set_kv_cache_layout(required_layout)` 写入全局 override。XPU 平台则在 `get_attn_backend_cls` 入口直接 `set_kv_cache_layout("NHD")`（[xpu.py](../../vllm/platforms/xpu.py)），因为 XPU kernel 只支持 NHD。
- `indexes_kv_by_block_stride()` 判断 `num_blocks` 是否为物理最外层（通过比对带/不带 `num_layers` 维的 stride order），返回 `True` 时 backend 容忍非连续 block 维，从而支持 page size padding 与跨层统一 KV 布局。

用户可通过 `VLLM_KV_CACHE_LAYOUT` 环境变量预先指定布局；`get_kv_cache_layout()` 依次读取代码 override → `VLLM_KV_CACHE_LAYOUT` → KV connector 默认布局。

### CUDA Graph 支持

[backend.py](../../vllm/v1/attention/backend.py) 定义了 `AttentionCGSupport` 四级枚举：

| 取值 | 数值 | 语义 |
| --- | --- | --- |
| `ALWAYS` | 3 | mixed-prefill-decode 均支持（如 FlashAttention） |
| `UNIFORM_BATCH` | 2 | 仅支持 query_len 全相同的批次（spec-decode 可用，如 "decode = 1 + num_speculative_tokens"） |
| `UNIFORM_SINGLE_TOKEN_DECODE` | 1 | 仅支持 `query_len == 1` 的纯 decode |
| `NEVER` | 0 | 不支持 cudagraph |

查询入口是 `AttentionMetadataBuilder.get_cudagraph_support(vllm_config, kv_cache_spec)`；其默认实现返回 ClassVar `_cudagraph_support`（默认 `NEVER`），子类可覆盖 ClassVar 或覆盖该方法做更细粒度判断。[gpu_model_runner.py](../../vllm/v1/worker/gpu_model_runner.py) 的 `_check_and_update_cudagraph_mode` 在所有 `kv_cache_group` 的所有 backend 之间取 `min(cg_support)` 作为整体 cudagraph mode 上界，再由 `resolve_cudagraph_mode_and_sizes` 落地为 FULL / PIECEWISE / NONE。

capture 时的钩子是 `build_for_cudagraph_capture(common_attn_metadata)`，默认实现委托 `build(common_prefix_len=0, common_attn_metadata=...)`；需要为 capture 准备特殊元数据的 builder 可覆盖它，但应调用 `self.build` 或 `super().build_for_cudagraph_capture` 以保持一致。

### 特性能力矩阵

`AttentionBackend` 暴露一组 `supports_*` / `is_*` 类方法，集中由 `validate_configuration`（[backend.py](../../vllm/v1/attention/backend.py)）在 backend 选择时统一读取，少数字段由模型层在 `Attention.__init__` 直接读取。

| 字段 | 语义 | 典型消费者 |
| --- | --- | --- |
| `is_mla()` | 是否 MLA backend（DeepSeek 风格） | `validate_configuration` 校验 `use_mla == is_mla` |
| `is_sparse()` | 是否 sparse MLA | 模型层触发 `use_sparse=True`；`validate_configuration` 校验 |
| `is_ssm()` | 是否 SSM/Mamba 家族 | runner 据此区分 mamba/attention 分组与 cache 类型 |
| `supports_sink()` | 是否支持 attention sink（如 Gemma） | `validate_configuration` 在 `has_sink` 时校验 |
| `supports_alibi_sqrt()` | 是否支持 ALiBi √d 缩放 | `Attention.__init__` 直接读取，`use_alibi_sqrt=True` 但不支持时 raise `ValueError` |
| `supports_mm_prefix()` | 是否支持 PrefixLM 多模态双向注意力 | `validate_configuration` 在 `use_mm_prefix` 时校验 |
| `supports_non_causal()` | 是否支持 decoder 路径的非因果双向 | `validate_configuration` 在 `use_non_causal` 时校验 |
| `supports_batch_invariance()` | 是否支持 `VLLM_BATCH_INVARIANT` 模式 | `validate_configuration` 在 `use_batch_invariant` 时校验 |
| `supports_kv_connector()` | 是否兼容 KV connector（默认 `True`） | `validate_configuration` 在 `use_kv_connector` 时校验 |
| `supports_per_head_quant_scales()` | 是否支持 per-head KV 量化 scale | `validate_configuration` 在 `use_per_head_quant_scales` 时校验 |
| `supports_attn_type(attn_type)` | 是否支持该 attention 类型（`DECODER` / `ENCODER` / `ENCODER_ONLY` / `ENCODER_DECODER`） | `validate_configuration`；默认仅支持 `DECODER` |
| `supports_compute_capability(cc)` | 是否支持该 compute capability | `validate_configuration`；ViT 路径也读取 |
| `supports_combination(...)` | 多维度组合校验，返回拒绝原因字符串 | `validate_configuration` 汇总，用于诸如 "sink + CC<9.0 不支持" 这类跨字段约束 |

### 用户可调旋钮

影响 backend 选择或运行行为的六种方式：

- `--attention-backend <NAME>` CLI 参数：解析后写入 `vllm_config.attention_config.backend`（`AttentionBackendEnum`），走平台 `get_attn_backend_cls` 的"显式指定"分支，`validate_configuration` 校验失败时直接 `raise ValueError`。
- `VLLM_ATTENTION_BACKEND` 环境变量：与 `--attention-backend` 等价，同样落到 `attention_config.backend` 的显式分支，校验失败即报错。适合在不修改启动脚本的情况下临时强制 backend（如 `VLLM_ATTENTION_BACKEND=TRITON_ATTN` 排查问题）。
- `VLLM_KV_CACHE_LAYOUT` 环境变量：预设 KV cache 布局（`NHD` / `HND`），由 `get_kv_cache_layout()` 读取。XPU 平台会强制覆盖为 `NHD`，FlashInfer 在 SM100 上会通过 `get_required_kv_cache_layout()` 强制 `HND`。
- `VLLM_BATCH_INVARIANT` 环境变量：启用 batch invariance 模式，进入 selector 后触发 `supports_batch_invariance` 校验；此外 `Attention.__init__` 会对 `FLASHINFER` / `TRITON_MLA` 自动关闭 prefix caching（该组合当前不支持）。
- `register_backend(Backend.X, "module.Cls")`：[registry.py](../../vllm/v1/attention/backends/registry.py) 提供的运行时覆盖，把枚举成员重定向到第三方实现路径，供扩展注册自定义 backend（或替换默认实现）。
- `Attention(attn_backend=MyBackend)`：模型代码直接传入 backend 类，**绕过** selector 与 `AttentionSelectorConfig`，由模型作者完全自负其责。

前两者走显式分支（校验失败抛错），中间两者影响布局与批不变性等横切行为，后两者分别在运行时注册与模型构造层旁路选择器。

### ViT 独立路径

ViT（encoder-only vision tower）的 backend 选择走独立路径，不经过 `get_attn_backend` 与 `AttentionSelectorConfig`，入口是 `current_platform.get_vit_attn_backend(head_size, dtype, backend=None)`（[cuda.py](../../vllm/platforms/cuda.py)）。

CUDA 平台的 `get_supported_vit_attn_backends` 在 SM80+ 返回 `[FLASH_ATTN, TRITON_ATTN, TORCH_SDPA, FLASHINFER]`，低于 SM80 时返回相同成员但顺序不同（`[FLASH_ATTN, TORCH_SDPA, TRITON_ATTN, FLASHINFER]`）。`get_vit_attn_backend` 按列表顺序逐个尝试 `supports_head_size` + `supports_dtype` + `supports_compute_capability`，第一个全部通过即采用；遇 `TORCH_SDPA` 直接返回（视为兜底）；全部失败时回退 `TORCH_SDPA`。ROCm 平台（[rocm.py](../../vllm/platforms/rocm.py)）的 `get_supported_vit_attn_backends` 返回 `[FLASH_ATTN, ROCM_AITER_FA, TRITON_ATTN, TORCH_SDPA]`，并在 `get_vit_attn_backend` 内按 gfx 架构（gfx9 / RDNA3+）做分支选择。

ViT 走独立路径的根本原因：vision tower 不使用 KV cache、不需要 paged attention，调度与内存模型与 decoder attention 完全不同，因此用独立的 backend 选择与独立的 attention 层类型（`ENCODER_ONLY`）实现。
