## 核心抽象层

vLLM v1 的 attention 子系统围绕几个边界清晰的对象组织：声明能力的 `AttentionBackend`、执行计算的 `AttentionImpl`、构造 per-step 元数据的 `AttentionMetadataBuilder`、模型代码直接使用的 `Attention` nn.Module，以及跨层共享的 `CommonAttentionMetadata`。它们共同把"如何选中并驱动一个 attention kernel"这一问题分层解耦。本节聚焦对象职责边界与协作关系，不涉及 kernel 内部实现。

### AttentionBackend 抽象基类

`AttentionBackend`（见 [backend.py](../../vllm/v1/attention/backend.py)）是所有 backend 的抽象基类。**它本身不持有任何运行时状态，是纯声明性的能力元数据**：所有方法都是 `@staticmethod` 或 `@classmethod`，实例化时不接收任何配置。它的职责是回答"我是什么、我能做什么、我的 KV cache 长什么样"。

**身份与工厂方法（staticmethod）**：

- `get_name()`：返回 backend 的字符串标识，用于日志、`AttentionBackendEnum` 索引与配置选择。
- `get_impl_cls()`：返回对应的 `AttentionImplBase` 子类，由 `Attention` 层在 `__init__` 时实例化为 `self.impl`。
- `get_builder_cls()`：返回对应的 `AttentionMetadataBuilder` 子类，由 model_runner 在初始化时实例化。
- `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)`：声明该 backend 期望的 KV cache tensor 形状。
- `get_kv_cache_stride_order(include_num_layers_dimension)`：声明 KV cache 维度的物理内存排列顺序；未实现时抛 `NotImplementedError`，框架按逻辑形状处理。
- `get_supported_kernel_block_sizes()`：返回 kernel 支持的 block size 列表，元素可为 `int`（固定值）或 `MultipleOf(base)`（约束为某整数倍）。
- `get_kv_cache_block_dim(...)`：通过注入哨兵值 `_S = 1234567` 调用 `get_kv_cache_shape`，再 `shape.index(_S)` 反查 num_blocks 所在维度，便于框架跨 backend 统一定位 block 索引维。
- `get_preferred_block_size(default_block_size)`：在默认值不满足 `supports_block_size` 时，回退到 kernel 支持集合中的最小值。

**能力查询（classmethod）**：backend 通过一组 `supports_*` / `is_*` 钩子声明自身能力，框架在选择阶段逐一查询：

- `supports_head_size` / `supports_dtype` / `supports_kv_cache_dtype` / `supports_block_size` / `supports_compute_capability`：基础硬件与张量属性匹配。
- `supports_attn_type`：默认只支持 `AttentionType.DECODER`，encoder / encoder-decoder 类型需 backend 显式覆盖。
- `supports_sink` / `supports_alibi_sqrt` / `supports_mm_prefix` / `supports_non_causal` / `supports_batch_invariance` / `supports_kv_connector` / `supports_per_head_quant_scales`：模型特性开关。
- `is_mla` / `is_sparse` / `is_ssm`：标记 backend 所属家族（MLA / Sparse MLA / 状态空间模型），默认均为 `False`。
- `supports_combination(...)`：当单项能力都满足但**组合**不兼容时，返回描述原因的字符串；返回 `None` 表示组合可用。这是 backend 表达"head_size + dtype + MLA + sink 同时出现才不工作"这类交叉约束的通道。
- `get_required_kv_cache_layout()`：声明 backend 要求的 KV cache 布局类型，返回 `None` 表示无特殊要求。
- `indexes_kv_by_block_stride()`：通过对比 `get_kv_cache_stride_order` 在含/不含 layers 维度下的返回，判断 num_blocks 是否为物理最外维，从而决定是否容忍非连续 block 维、是否支持跨层统一布局与 page size padding。
- `forward_includes_kv_cache_update`：实例属性（默认 `True`），声明 `forward` 是否同时完成 KV cache 写入；为 `False` 时 `Attention.forward` 会显式调用 `unified_kv_cache_update` 建立数据依赖。

**`validate_configuration` 契约**：这是 backend 选择的核心入口。它接收一组配置参数（`head_size` / `dtype` / `kv_cache_dtype` / `block_size` / `use_mla` / `has_sink` / `use_sparse` / `use_mm_prefix` / `use_per_head_quant_scales` / `device_capability` / `attn_type`，以及可选的 `use_non_causal` / `use_batch_invariant` / `use_kv_connector`），逐项调用上述 `supports_*` 钩子，并最后调用 `supports_combination`。**返回值是 `list[str]`**：空列表表示配置完全有效；每个元素是一条人类可读的"无效原因"（如 `"head_size not supported"`、`"MLA not supported"`、组合原因字符串等）。框架据此挑选出 invalid_reasons 为空的 backend，若有多个再用优先级排序。这一契约让 backend 选择逻辑集中、可测试，并能在选择失败时给出可读的诊断信息——显式指定场景下若返回非空列表会直接 `raise ValueError`，不会静默回退。

### AttentionImpl 家族

`AttentionImplBase`（见 [backend.py](../../vllm/v1/attention/backend.py)）是所有 impl 的基类，泛型参数 `T` 绑定到该 backend 的 `AttentionMetadata` 子类型。它在 `__new__` 中通过 `get_dcp_group` / `get_pcp_group` 探测 Decode Context Parallelism 与 Prefill Context Parallelism 的 world size / rank，初始化 `dcp_world_size` / `dcp_rank` / `pcp_world_size` / `pcp_rank` / `total_cp_world_size` / `total_cp_rank`，并在 DCP world size > 1 且 `can_return_lse_for_decode` 为真时自动置 `need_to_return_lse_for_decode = True`。**impl 实例持有真正的运行时状态**（`num_heads` / `head_size` / `scale` 等），是 forward 计算的实际执行者，与纯声明性的 backend 类形成对照。

**共享字段（定义在 `AttentionImplBase`）**：

- `can_return_lse_for_decode`：impl 是否能在 decode 时返回 softmax LSE，DCP 等特性依赖此能力。
- `lse_base_on_e`：返回的 LSE 使用自然对数（`True`，多数 backend）还是 log2（`False`，如 FlashInfer trtllm-gen MLA）。DCP 跨 shard 合并 kernel 会据此分支，配错会静默损坏 softmax 分母。
- `supports_pcp`：是否支持 Prefill Context Parallelism。
- `supports_mtp_with_cp_non_trivial_interleave_size`：在 `cp_kv_cache_interleave_size > 1` 时是否支持 MTP。
- `supports_quant_query_input`：是否接受预量化的 query 输入，使 `torch.compile` 能把量化融合进上游算子。
- `dcp_world_size` / `pcp_world_size`：并行组规模，由 `__new__` 探测；`total_cp_world_size = pcp_world_size * dcp_world_size`。

**三个 impl 基类与 forward 接口差异**：

1. `AttentionImpl`（标准 attention）：定义 `forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None, output_block_scale=None)`。这是最常见的接口：接收 `AttentionLayer` 协议对象（提供 q/k/v scale）、Q/K/V 张量、KV cache、per-step metadata 与输出缓冲。它额外提供 fused 路径钩子：`fused_output_quant_supported(quant_key)` 声明是否支持输出量化融合（用于 AttnFusionPass）；`fused_rope_kvcache_supported()` 声明是否支持 RoPE 与 KV cache 更新融合；若后者为 `True`，`do_rope_and_kv_cache_update(...)` 会被 `torch.ops.vllm.fused_rope_and_unified_kv_cache_update` 调用以原地完成 RoPE 与 KV cache 写入。

2. `MLAAttentionImpl`（DeepSeek 风格 MLA）：不提供 `forward`，而是暴露两个分支：`forward_mha(q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata, k_scale, output, output_scale=None)` 用于 prefill（MHA 风格）；`forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer) -> (output, lse)` 用于 decode（MQA 风格）。由模型层根据 phase 选择调用哪个。`do_kv_cache_update(...)` 提供默认实现，通过 `ops.concat_and_cache_mla` 写入 MLA 格式的 KV cache。`fused_output_quant_supported` 默认对一组 FP8/NVFP4 quant key 返回 `True`。

3. `SparseMLAAttentionImpl`（Sparse MLA，仅 decode）：只暴露 `forward_mqa`，不支持 `forward_mha`，因此只能用于 decode 路径，prefill 须由另一个 backend 承担。`do_kv_cache_update` 与 `fused_output_quant_supported` 行为同 `MLAAttentionImpl`。

这一家族划分让模型代码可以针对不同 MLA 子模式（full MLA / sparse-only）选择不同 impl，同时共享 `AttentionImplBase` 的 DCP/PCP 探测与 LSE 处理逻辑。

### AttentionMetadataBuilder

`AttentionMetadataBuilder`（见 [backend.py](../../vllm/v1/attention/backend.py)）是 per-step 在 CPU 侧构造 `AttentionMetadata` 的对象，由 model_runner 持有（每个 kv_cache group 一个）。它的输入是跨层共享的 `CommonAttentionMetadata`，输出是 backend 专属的 `AttentionMetadata`，供该 step 内所有层的 `impl.forward` 使用。

**关键方法**：

- `build(common_prefix_len, common_attn_metadata, fast_build=False)`：核心构造方法。`common_prefix_len` 是 batch 公共前缀长度（cascade attention 用）；`fast_build` 为 `True` 时优先构造速度而非执行速度，适用于 spec-decode 中只用到少数层的场景。部分 backend（如 MLA）要求在 `build` 前先调用 `reorder_batch`。
- `build_for_cudagraph_capture(common_attn_metadata)`：为 CUDA graph 捕获构造 metadata，默认调用 `build(common_prefix_len=0, ...)`。
- `build_for_drafting(common_attn_metadata, draft_index)`：为 draft model 构造 metadata，`draft_index` 标识当前 draft 步（链式 spec 对应第 i 个 token，树状 spec 对应第 i 层）；默认调用 `build` 并置 `fast_build=True`。
- `use_cascade_attention(...)`：根据公共前缀长度、query lens、head 数、ALiBi / sliding window / local attention 开关、SM 数与 DCP world size，判断本 step 是否值得启用 cascade attention。默认返回 `False`。
- `update_block_table(metadata, blk_table, slot_mapping)`：当存在多个 kv_cache group 共享几乎相同 metadata、仅 block table 不同时，可复用已构造的 metadata 仅替换 block table，避免重复构造。仅当 `supports_update_block_table = True` 时需实现。
- `get_cudagraph_support(vllm_config, kv_cache_spec)`（classmethod）：返回本 builder 的 CUDA graph 支持等级。

**CUDA graph 支持等级**：`_cudagraph_support` 是 `ClassVar[AttentionCGSupport]`，默认 `NEVER`。`AttentionCGSupport` 四级枚举（见 [backend.py](../../vllm/v1/attention/backend.py)）：

- `ALWAYS = 3`：始终支持 CUDA graph，包括混合 prefill-decode。
- `UNIFORM_BATCH = 2`：当 batch 内所有 query 长度相同时支持，可用于 spec-decode（decode 即 `1 + num_speculative_tokens`）。
- `UNIFORM_SINGLE_TOKEN_DECODE = 1`：仅当 batch 全为 `query_len == 1` 的 decode 时支持。
- `NEVER = 0`：不支持 CUDA graph。

注意 cascade attention 当前一律不支持 CUDA graph，该枚举仅描述非 cascade 路径。

**batch 重排与 block table 更新**：

- `reorder_batch_threshold`：`None` 表示不重排；否则表示会把 query 长度 ≤ 该阈值的请求拉到 batch 前部。`_init_reorder_batch_threshold(...)` 会根据 `speculative_config.num_speculative_tokens` 与 `parallel_drafting` 自动放大阈值，并在 DCP > 1 且 backend 不支持 varlen 时强制置 1。
- `supports_update_block_table`：声明 builder 是否实现 `update_block_table`，用于多 kv_cache group 场景的 metadata 复用。

### Attention 层（nn.Module）

`Attention`（见 [attention.py](../../vllm/model_executor/layers/attention/attention.py)）是模型代码直接使用的 `nn.Module`，继承自 `AttentionLayerBase`。它把"选择 backend、实例化 impl、注册到 forward context、dispatch 到 impl.forward"串成一条链。

**`__init__` 阶段**：

1. 若调用方未传入 `attn_backend=`，则调用 `get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla=False, has_sink=..., use_mm_prefix=..., use_per_head_quant_scales=..., attn_type=...)` 选中一个 `AttentionBackend` 子类；否则直接使用传入的 backend 类。`get_attn_backend` 内部即依据上一节描述的 `validate_configuration` 契约筛选。
2. 调用 `backend.get_impl_cls()` 得到 impl 类，用 `num_heads / head_size / scale / num_kv_heads / alibi_slopes / sliding_window / kv_cache_dtype / logits_soft_cap / attn_type / kv_sharing_target_layer_name` 及 `**extra_impl_args` 实例化 `self.impl`。
3. 通过 `compilation_config.static_forward_context[prefix] = self` 把自己注册到静态 forward context，键为 `layer_name`（即 `prefix`）。重复 layer 名会抛 `ValueError`。`kv_sharing_target_layer_name` 也会在此校验其指向的层已注册。
4. `self.backend = AttentionBackendEnum[self.attn_backend.get_name()]` 记录枚举值便于追溯。
5. 根据 `current_platform.opaque_attention_op()` 决定 `self.use_direct_call`：cuda-alike（CUDA/ROCm）与 CPU 平台为 `False`（走 opaque custom op 路径），其他平台为 `True`。

**`forward` 阶段**：

1. 若 `calculate_kv_scales` 为真，调用 `torch.ops.vllm.maybe_calc_kv_scales(query, key, value, _encode_layer_name(self.layer_name))`。
2. 若 `self.query_quant` 存在且 impl 支持 `supports_quant_query_input`，先把 query 量化（使 `torch.compile` 能融合）。
3. reshape Q/K/V 到 `[num_tokens, heads, head_dim]`，分配 `output` 缓冲。
4. 取出 attn_metadata：它并非作为参数传入，而是由 model_runner 的 `execute_model` 通过 context manager 设置，forward 内通过 `vllm.forward_context.get_forward_context().attn_metadata` 取回（见方法 docstring）。
5. 当 `forward_includes_kv_cache_update` 为 `False` 且本层不共享 KV cache 时，先调用 `unified_kv_cache_update(key, value, layer_name)` 显式写入 KV cache，得到 `kv_cache_dummy_dep` 以建立数据依赖。
6. dispatch：
   - `use_direct_call=True`：直接调用 `unified_attention_with_output(...)`（Python 函数），由其内部取出 `self.impl` 并调用 `impl.forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)`。
   - `use_direct_call=False`：调用 `torch.ops.vllm.unified_attention_with_output(query, key, value, output, encoded_layer_name, kv_cache_dummy_dep=...)`，把 attention 注册为一个 opaque custom op，便于 `torch.compile` 把整层 attention 当作不可穿透的算子处理。

`use_direct_call` 的核心作用是：在 cuda-alike 平台把 attention 包成 opaque op 以控制 `torch.compile` 行为；在非 cuda-alike 平台则直接调用 impl，让 `torch.compile` 自行处理。

### CommonAttentionMetadata

`CommonAttentionMetadata`（见 [backend.py](../../vllm/v1/attention/backend.py)）是 per-batch、跨层跨 backend 共享的 attention 元数据 dataclass，由 scheduler/model_runner 侧构造，喂给各 `AttentionMetadataBuilder.build`。许多张量同时保留 GPU 与 CPU 版本以避免隐式 H↔D 同步。

**关键字段**：

- `query_start_loc`：`(batch_size + 1,)`，每个 request 在 query 张量中的起始偏移。
- `query_start_loc_cpu`：同上的 CPU 副本。
- `seq_lens`：`(batch_size,)`，每个 request 已计算的 token 数（上下文长度）。
- `num_reqs`：batch 中 request 数。
- `num_actual_tokens`：batch 内总 token 数（可能含 padding，命名待重构）。
- `max_query_len`：batch 内最长 query。
- `max_seq_len`：最长上下文长度（可能是上界）。
- `block_table_tensor`：分页 KV cache 的 block 映射表。
- `slot_mapping`：token 到 KV cache slot 的映射。
- `causal`：`bool` 或 `(batch_size,)` 张量，标记是否因果注意力。
- `is_prefilling`：`(batch_size,)` bool 张量，True 表示该 request 仍在 prefill 阶段。
- `positions`：`(num_actual_tokens,)` token 位置，供 builder 预计算 position-dependent 的 sparse metadata。
- `mm_req_doc_ranges`：`dict[int, list[tuple[int, int]]]`，PrefixLM 多模态 token 的双向注意力区间，按 request index 映射；纯文本或非 PrefixLM 模型为 `None`。
- `rswa_prefix_lens`：`(batch_size,)`，Reference Sliding Window Attention 的每 request 前缀长度；低于该值的 token 全局可见，之后的额外走滑动窗。
- `dcp_local_seq_lens` / `dcp_local_seq_lens_cpu`：DCP 本地 rank 的序列长度。
- `encoder_seq_lens` / `encoder_seq_lens_cpu`：encoder-decoder 场景的 encoder 序列长度。
- `logits_indices_padded` / `num_logits_indices`：FastPrefillAttentionBuilder 使用。

**`unpadded(num_actual_tokens, num_actual_reqs)`**：返回一个裁剪后的 `CommonAttentionMetadata`，把各张量切片到实际 token / request 数（如 `query_start_loc[:num_actual_reqs+1]`、`seq_lens[:num_actual_reqs]`、`slot_mapping[:num_actual_tokens]` 等）。主要用于 spec-decode 场景：当 batch 被 padding 到 CUDA graph 静态形状时，实际有效的 token/request 数小于 padding 后的尺寸，`unpadded` 让 builder 看到去除 padding 后的真实规模。

## backend 注册与发现

vLLM 通过两个枚举（`AttentionBackendEnum` / `MambaAttentionBackendEnum`）登记内置 backend，通过 `register_backend` 支持运行时覆盖与第三方扩展，通过 `subclass_attention_backend` 工具函数在不修改原 backend 类的前提下派生子类。三者协同覆盖了"内置声明 → 运行时覆盖 → 轻量派生"三种接入方式。

### AttentionBackendEnum 与 MambaAttentionBackendEnum

`AttentionBackendEnum`（见 [registry.py](../../vllm/v1/attention/backends/registry.py)）枚举所有内置 attention backend。**枚举值即默认 fully-qualified 类路径**，例如：

```python
FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
FLASHINFER_MLA = "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
```

`MambaAttentionBackendEnum` 对 SSM 类 backend（`MAMBA1` / `MAMBA2` / `SHORT_CONV` / `LINEAR` / `GDN_ATTN`）做同样的事，结构完全对称。

**`get_path()` / `get_class()`**：这两个方法在返回类路径前会查询全局覆盖表 `_ATTN_OVERRIDES` / `_MAMBA_ATTN_OVERRIDES`：若该枚举成员被注册过，返回覆盖后的路径；否则返回枚举默认值。`get_class()` 进一步通过 `resolve_obj_by_qualname` 把字符串解析为真实的 `AttentionBackend` 子类。`is_overridden()` / `clear_override()` 用于查询与清除覆盖。

**`CUSTOM = None` 占位符**：两个枚举都有一个值为 `None` 的 `CUSTOM` 成员，专为第三方 backend 预留。由于值为 `None`，`get_path()` 在未注册时会抛 `ValueError`，提示必须先调用 `register_backend(Backend.CUSTOM, 'your.module.YourClass')`。把 `CUSTOM` 设为 `None`（而非空字符串）是为了避免与 `TORCH_SDPA = ""` 这类空字符串值的 backend 产生别名冲突。

**`_AttentionBackendEnumMeta`**：自定义元类，重写 `__getitem__`，在 `AttentionBackendEnum["UNKNOWN"]` 时把所有合法成员名拼进错误信息，给出比原生 `KeyError` 更友好的提示。

### register_backend 覆盖机制

`register_backend(backend, class_path=None, is_mamba=False)`（见 [registry.py](../../vllm/v1/attention/backends/registry.py)）既是装饰器也可直接调用，是第三方扩展接入 vLLM attention 子系统的标准入口。它把 `f"{cls.__module__}.{cls.__qualname__}"`（或显式传入的 `class_path`）写入 `_ATTN_OVERRIDES` / `_MAMBA_ATTN_OVERRIDES`。

**装饰器用法（覆盖已有枚举）**：

```python
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum, register_backend)

@register_backend(AttentionBackendEnum.FLASH_ATTN)
class MyCustomFlashAttn(AttentionBackend):
    ...
```

此后 `AttentionBackendEnum.FLASH_ATTN.get_class()` 返回 `MyCustomFlashAttn`，所有未显式指定 backend 的 `Attention` 层都会使用它。

**直接调用（注册 CUSTOM）**：

```python
register_backend(
    AttentionBackendEnum.CUSTOM,
    "third_party.pkg.MyBackend",
)
# 之后即可通过 AttentionBackendEnum.CUSTOM.get_class() 取回
```

`is_mamba=True` 时写入 `_MAMBA_ATTN_OVERRIDES`，用于覆盖 `MambaAttentionBackendEnum` 成员。装饰器形式下若同时传 `class_path`，则 `class_path` 优先并返回 no-op decorator。注意覆盖是进程级的全局状态，`clear_override()` 可恢复默认。

### subclass_attention_backend 工具函数

有时需要在不修改原 backend 类、也不全局覆盖枚举的前提下派生一个子类——例如只为某个 kv_cache group 替换 builder，或覆盖一两个能力查询方法。`subclass_attention_backend` 与 `subclass_attention_backend_with_overrides`（见 [backend.py](../../vllm/v1/attention/backend.py)）提供这种"轻量派生"能力。

- `subclass_attention_backend(name_prefix, attention_backend_cls, builder_cls)`：用 `type(name, (attention_backend_cls,), {"get_builder_cls": lambda: builder_cls})` 动态生成子类，使其 `get_builder_cls` 返回指定的 `builder_cls`，其余方法继承自原 backend。`name` 为 `name_prefix + 原类名`。适用于"backend kernel 没问题、只想换 metadata builder"的场景。

- `subclass_attention_backend_with_overrides(name_prefix, attention_backend_cls, overrides)`：把 `overrides` 字典作为类属性注入到新子类，可覆盖任意方法/属性（如某个 `supports_*` 钩子或 `forward_includes_kv_cache_update`），适用于只需改一两处行为的场景。

两者都返回新生成的子类（不修改原类、不写注册表），调用方可以将其作为 `attn_backend=` 显式传给 `Attention` 层，实现 per-layer 的 backend 定制，而不影响其他层或全局选择结果。
