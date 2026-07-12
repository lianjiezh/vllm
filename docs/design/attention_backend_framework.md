<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM project -->

# vLLM Attention Backend 设计框架

本文档讲解 vLLM v1 attention 子系统的整体设计框架：支持哪些 attention backend、在什么场景下选择哪种 backend，以及框架层抽象出的 `Attention` / `AttentionBackend` / `AttentionImpl` / `AttentionMetadataBuilder` 对象如何协作，使同一份模型代码可以无缝接入不同 backend。文档聚焦"框架如何接入"，不涉及具体 kernel 的数值算法实现。

> 如果你想查看各 backend 的特性支持矩阵（auto-generated），请参阅 [attention_backends.md](attention_backends.md)。

## 目录

- [概述与设计目标](#概述与设计目标)
- [核心抽象层](#核心抽象层)
  - [AttentionBackend 抽象基类](#attentionbackend-抽象基类)
  - [AttentionImpl 家族](#attentionimpl-家族)
  - [AttentionMetadataBuilder](#attentionmetadatabuilder)
  - [Attention 层（nn.Module）](#attention-层nnmodule)
  - [CommonAttentionMetadata](#commonattentionmetadata)
- [backend 注册与发现](#backend-注册与发现)
  - [AttentionBackendEnum 与 MambaAttentionBackendEnum](#attentionbackendenum-与-mambaattentionbackendenum)
  - [register_backend 覆盖机制](#register_backend-覆盖机制)
  - [subclass_attention_backend 工具函数](#subclass_attention_backend-工具函数)
- [backend 选择机制](#backend-选择机制)
  - [入口：get_attn_backend](#入口get_attn_backend)
  - [显式指定分支](#显式指定分支)
  - [自动选择分支](#自动选择分支)
  - [validate_configuration 契约](#validate_configuration-契约)
  - [KV cache layout 副作用](#kv-cache-layout-副作用)
- [各平台优先级表](#各平台优先级表)
  - [CUDA 平台](#cuda-平台)
  - [ROCm 平台](#rocm-平台)
  - [XPU 平台](#xpu-平台)
  - [CPU 平台](#cpu-平台)
  - [ViT 独立路径](#vit-独立路径)
- [完整 backend 清单](#完整-backend-清单)
  - [标准注意力 backend](#标准注意力-backend)
  - [MLA backend](#mla-backenddeepseek-风格多头潜在注意力)
  - [Sparse MLA backend](#sparse-mla-backend)
  - [SSM / Mamba / Linear backend](#ssm--mamba--linear-backend来自-mambaattentionbackendenum)
  - [量化/ROCm 专用 backend](#量化rocm-专用-backend)
- [端到端调用链](#端到端调用链)
  - [阶段一：模型构造](#阶段一模型构造)
  - [阶段二：runner 初始化](#阶段二runner-初始化)
  - [阶段三：每 step 执行](#阶段三每-step-执行)
- [横切关注点](#横切关注点)
  - [KV cache 布局](#kv-cache-布局)
  - [CUDA Graph 支持](#cuda-graph-支持)
  - [特性能力矩阵](#特性能力矩阵)
  - [用户可调旋钮](#用户可调旋钮)
  - [ViT 独立路径](#vit-独立路径-1)

## 概述与设计目标

vLLM v1 的 attention 子系统是该框架最复杂、最关键的部分之一。它需要在同一份模型代码下支持多种硬件平台（CUDA / ROCm / XPU / CPU）和数十种 attention 实现（FlashAttention、FlashInfer、Triton、MLA 系列、Mamba/SSM 系列等），并按"配置 + 优先级 + 能力校验"的方式自动选择最佳 backend。

围绕这一目标，attention 子系统遵循四条设计原则：

1. **平台无关性**：模型代码只调用 `Attention(num_heads, head_size, ...)`，不感知底层是 FlashAttention 还是 Triton；平台差异由 selector 与 platform 层吸收。
2. **可插拔 backend**：每个 backend 是一组 `(AttentionBackend, AttentionImpl, AttentionMetadataBuilder)` 三元组，新 backend 只需实现这三个抽象并注册到枚举即可接入。
3. **能力驱动选择**：backend 通过 `supports_*` / `is_*` / `validate_configuration` 声明自己的能力边界，selector 按优先级表逐一校验，选第一个合法者。这把"该 backend 能否在此配置下运行"的判断权完全交给 backend 自己。
4. **运行时覆盖**：用户既可通过 `--attention-backend` CLI 显式指定，也可通过 `register_backend` 在运行时把某个枚举成员重定向到第三方实现，还可在模型代码中直接传入 `attn_backend=` 绕过 selector。

这些原则分别落地在 [backend.py](../../vllm/v1/attention/backend.py)（抽象层与能力契约）、[selector.py](../../vllm/v1/attention/selector.py)（选择入口）、[registry.py](../../vllm/v1/attention/backends/registry.py)（注册表）、以及各 [platform](../../vllm/platforms/) 模块的 `get_attn_backend_cls`（平台优先级）中。后续章节按这一脉络展开。

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

## backend 选择机制

vLLM 在运行时根据模型形状、dtype、设备能力等条件，从注册表中选择一个 attention backend。整个选择过程集中在 [selector.py](../../vllm/v1/attention/selector.py) 与各 platform 的 `get_attn_backend_cls` 中，分为"显式指定"与"自动选择"两条分支，统一通过 `validate_configuration` 做能力校验。

### 入口：get_attn_backend

`Attention.__init__` 在构造每一层 attention 时调用 `get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla, has_sink, use_sparse, use_mm_prefix, use_per_head_quant_scales, attn_type, num_heads)`。其内部流程如下：

1. 校验 `kv_cache_dtype`（若非 None）属于 `CacheDType` 的合法取值。
2. 通过 `get_current_vllm_config()` 取出当前 `vllm_config`：
   - 从 `cache_config` 取 `block_size`——仅当 `cache_config.user_specified_block_size` 为真（即用户显式传了 `--block-size`）时才填入，否则置为 `None`，表示"block_size 不参与过滤"。
   - 从 `kv_transfer_config` 判断 `use_kv_connector`（`kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance`）。
3. 组装一个 `AttentionSelectorConfig` NamedTuple（见 [selector.py](../../vllm/v1/attention/selector.py)），包含 `head_size / dtype / kv_cache_dtype / block_size / use_mla / has_sink / use_sparse / use_mm_prefix / use_per_head_quant_scales / attn_type / use_non_causal / use_batch_invariant / use_kv_connector` 共 13 个字段。其中 `use_non_causal` 取自 `attention_config.use_non_causal`，`use_batch_invariant` 取自 `envs.VLLM_BATCH_INVARIANT`。
4. 调用被 `@cache` 装饰的 `_cached_get_attn_backend(backend, attn_selector_config, num_heads)`，传入 `vllm_config.attention_config.backend`（用户是否显式指定）。`@cache` 使得**相同 `(backend, attn_selector_config, num_heads)` 三元组只解析一次**，后续同配置层直接复用结果。
5. `_cached_get_attn_backend` 委托 `current_platform.get_attn_backend_cls(backend, attn_selector_config, num_heads)` 得到一个类的全限定名字符串。
6. 若返回值为空则 `raise ValueError("Invalid attention backend for ...")`；否则用 `resolve_obj_by_qualname` 懒加载真正的类对象。
7. 加载完成后，调用 `backend.get_required_kv_cache_layout()`：若返回非 None（如 `"NHD"` / `"HND"`），则调用 `set_kv_cache_layout(required_layout)` 调整全局 KV 布局并打印日志（详见 [KV cache layout 副作用](#kv-cache-layout-副作用)）。

简化的文字流程图：

```
Attention.__init__
  └─ get_attn_backend(head_size, dtype, kv_cache_dtype, use_mla, ...)
       ├─ get_current_vllm_config() ──► block_size(仅 user_specified) / use_kv_connector
       ├─ 组装 AttentionSelectorConfig NamedTuple
       └─ _cached_get_attn_backend(backend, config, num_heads)   # @cache
            ├─ current_platform.get_attn_backend_cls(backend, config, num_heads)
            │     ├─ backend is not None ──► 显式指定分支（fail-fast）
            │     └─ backend is None     ──► 自动选择分支（get_valid_backends + min(priority)）
            ├─ resolve_obj_by_qualname(cls_path) ──► AttentionBackend 子类
            └─ backend.get_required_kv_cache_layout() != None
                  └─ set_kv_cache_layout(required_layout)   # 全局副作用
```

### 显式指定分支

当 `attention_config.backend` 不为 None（用户通过 `--attention-backend` CLI 或 `LLM(attention_backend=...)` Python API 指定，最终写入 `vllm_config.attention_config.backend`）时，平台 `get_attn_backend_cls` 先单独校验该 backend：

1. 取出该 backend 的类，调用 `validate_configuration(device_capability, **attn_selector_config._asdict())`；若加载抛 `ImportError`，则视作 `invalid_reasons = ["ImportError"]`。
2. 若 `validate_configuration` 返回**非空** invalid_reasons 列表，则**直接 `raise ValueError`**，错误信息形如 `"Selected backend X is not valid for this configuration. Reason: [...]"`。
3. 若返回空列表，则日志记录 `"Using X backend."`（CUDA）或 `"Using X backend (selected via --attention-backend)."`（ROCm），并返回该 backend 的类路径。

注意这里**不会静默回退**到其他 backend：用户指定的 backend 一旦不满足能力校验，就立即失败。这是"fail-fast"语义——避免用户以为自己在用某个 backend，实际却被悄悄换成另一个。

### 自动选择分支

当未显式指定 backend 时，调用 `get_valid_backends(device_capability, attn_selector_config, num_heads)`：

1. 调用平台 `_get_backend_priorities(...)`，得到一个**已按优先级排序**的 `AttentionBackendEnum` 列表（priority 索引越小越优，即列表越靠前优先级越高）。
2. 对列表中每个 backend，`enumerate` 得到其 priority 索引，取出类并调用 `validate_configuration(device_capability, **attn_selector_config._asdict())` 过滤：
   - 抛 `ImportError` 记为 `["ImportError"]`。
   - 返回非空 invalid_reasons 列表则记入 `invalid_reasons` 字典（CUDA 同时存 `(priority, reasons)` 元组）。
   - 返回空列表则加入合法候选集 `valid_backends_priorities`。
3. 若合法候选集为空，`raise ValueError("No valid attention backend found for ...")`。
4. 在合法候选中用 `min(..., key=lambda c: c.priority)` 选 priority 最小（最优）者作为最终 backend。

`--block-size` 排除更高优先级 backend 的 warning 逻辑（CUDA 实现）：若 `attn_selector_config.block_size is not None`（即用户指定了 `--block-size` 但没指定 `--attention-backend`），则扫描 `all_invalid_reasons`，找出所有 `priority < selected_priority` 且 `reasons == ["block_size not supported"]` 的 backend 组成 `excluded` 列表；若 `excluded` 非空，打印 warning 提示用户"`--block-size N` 排除了更高优先级的 backend X, Y，当前改用 Z，性能可能下降，建议移除 `--block-size` 以自动选择最优 block size"。ROCm 的自动选择分支结构类似，但通过 `sorted` 选最小 priority，且不包含 block_size warning 逻辑（ROCm 的 `_get_backend_priorities` 不依赖 block_size）。

### validate_configuration 契约

`AttentionBackend.validate_configuration`（见 [backend.py](../../vllm/v1/attention/backend.py)）是能力校验的总入口，依次执行下列检查，每项失败向 `invalid_reasons` 列表追加一条字符串：

| 检查项 | 失败时追加的 reason |
|---|---|
| `supports_head_size(head_size)` | `head_size not supported` |
| `supports_dtype(dtype)` | `dtype not supported` |
| `supports_kv_cache_dtype(kv_cache_dtype)` | `kv_cache_dtype not supported` |
| `supports_block_size(block_size)` | `block_size not supported` |
| `use_mm_prefix and not supports_mm_prefix()` | `partial multimodal token full attention not supported` |
| `use_mla != is_mla()` | `MLA not supported` 或 `non-MLA not supported` |
| `has_sink and not supports_sink()` | `attention sinks not supported` |
| `use_sparse != is_sparse()` | `sparse not supported` 或 `non-sparse not supported` |
| `use_per_head_quant_scales and not supports_per_head_quant_scales()` | `per-head quant scales not supported` |
| `supports_compute_capability(device_capability)` | `compute capability not supported` |
| `supports_attn_type(attn_type)` | `attention type {attn_type} not supported` |
| `use_non_causal and not supports_non_causal()` | `non-causal attention not supported` |
| `use_batch_invariant and not supports_batch_invariance()` | `batch invariance not supported` |
| `use_kv_connector and not supports_kv_connector()` | `KV connector not supported` |
| `supports_combination(...)` 返回非 None | 该返回字符串（组合约束） |

返回**空列表**表示该 backend 对当前配置完全合法；返回非空列表则每条 reason 描述一项不满足的能力。注意最后一项 `supports_combination` 是"组合约束"检查：即使单项都通过，某些 backend 也可能在特定参数组合下不可用（如 `TOKENSPEED_MLA` 仅在 R1 dims + FP8 KV 时合法，否则被 `supports_combination` 拒绝）。

### KV cache layout 副作用

选中 backend 并 `resolve_obj_by_qualname` 加载类之后，`_cached_get_attn_backend` 调用 `backend.get_required_kv_cache_layout()`：

- 若返回 `None`（默认实现），不做任何事。
- 若返回 `"NHD"` 或 `"HND"`，则调用 `set_kv_cache_layout(required_layout)`，该函数将全局变量 `_KV_CACHE_LAYOUT_OVERRIDE` 设为该布局并清空 `get_kv_cache_layout` 的缓存，从而**全局生效**为后续 KV cache 分配使用的布局；同时打印日志 `"Using %s KV cache layout for %s backend."`。

这是 backend 选择对全局状态的**唯一副作用**：除此之外，选择过程本身是纯函数式的（输入 config，输出类路径）。正因如此，`@cache` 才能安全地缓存结果——但前提是首次选择决定了全局 KV 布局，后续同配置的层会复用同一 backend 与同一布局。

## 各平台优先级表

各平台的 `_get_backend_priorities` 返回已排序的 backend 列表，priority 即列表索引（0 最优）。以下表格严格按源码顺序列出。

### CUDA 平台

CUDA 的优先级由 `use_mla` 与 `device_capability.major` 共同决定（见 [cuda.py](../../vllm/platforms/cuda.py) 的 `_get_backend_priorities`）。

**子表 1：MLA + `device_capability.major == 10`（Blackwell）**

基础列表后接 `*sparse_backends`：

| priority | backend |
|---|---|
| 0 | `FLASHINFER_MLA` |
| 1 | `TOKENSPEED_MLA` |
| 2 | `CUTLASS_MLA` |
| 3 | `FLASH_ATTN_MLA` |
| 4 | `FLASHMLA` |
| 5 | `TRITON_MLA` |
| 6+ | `*sparse_backends`（见下） |

`sparse_backends` 的两种排序：

- 当 `is_quantized_kv_cache(kv_cache_dtype)` 为真（FP8 KV cache）时：`[FLASHINFER_MLA_SPARSE, FLASHMLA_SPARSE]`（优先 FlashInfer，因其 FP8 KV 表现更好）。
- BF16 KV cache 且 `num_heads <= 16` 时：`[FLASHINFER_MLA_SPARSE, FLASHMLA_SPARSE]`（低 head 数时 FlashMLA 会 padding，故优先 FlashInfer）。
- BF16 KV cache 且 `num_heads > 16` 时：反序为 `[FLASHMLA_SPARSE, FLASHINFER_MLA_SPARSE]`。

排序理由：Blackwell MLA 优先 `FLASHINFER_MLA` 是因 benchmark 显示其在该架构上表现最佳（详见 issue #35807）；`TOKENSPEED_MLA` 仅在 R1 dims + FP8 KV 时被 `supports_combination` 放行，否则会被过滤。

**子表 2：MLA + `device_capability.major == 12`（SM120）**

| priority | backend |
|---|---|
| 0 | `TRITON_MLA` |
| 1 | `FLASHINFER_MLA_SPARSE_SM120` |

排序理由：SM120 上 FlashInfer/FlashMLA 等 MLA kernel 不可用，仅 Triton MLA 与专为 SM120 适配的 FlashInfer sparse 变体可用，Triton 优先。

**子表 3：MLA + 其他（含 SM90 H100/H200）**

| priority | backend |
|---|---|
| 0 | `FLASH_ATTN_MLA` |
| 1 | `FLASHMLA` |
| 2 | `FLASHINFER_MLA` |
| 3 | `TRITON_MLA` |
| 4 | `FLASH_ATTN_MLA_SPARSE` |
| 5 | `FLASHMLA_SPARSE` |

排序理由：Hopper 等非 Blackwell 架构上 `FLASH_ATTN_MLA` 综合表现最优，故置首。

**子表 4：非 MLA + `major == 10`（Blackwell）**

| priority | backend |
|---|---|
| 0 | `FLASHINFER` |
| 1 | `FLASH_ATTN` |
| 2 | `TRITON_ATTN` |
| 3 | `FLEX_ATTENTION` |
| 4 | `TURBOQUANT` |

排序理由：Blackwell 上 FlashInfer 非 MLA 路径表现最优。

**子表 5：非 MLA + 其他**

| priority | backend |
|---|---|
| 0 | `FLASH_ATTN` |
| 1 | `FLASHINFER` |
| 2 | `TRITON_ATTN` |
| 3 | `FLEX_ATTENTION` |
| 4 | `TURBOQUANT` |

排序理由：非 Blackwell 架构上 `FLASH_ATTN` 综合最优，置首；FlashInfer 次之。

### ROCm 平台

ROCm 的优先级由 `use_sparse / use_mla / use_kv_connector` 与 aiter 可用性共同决定（见 [rocm.py](../../vllm/platforms/rocm.py) 的 `_get_backend_priorities`）。

**子表 1：`use_sparse=True`**

| priority | backend |
|---|---|
| 0 | `ROCM_AITER_MLA_SPARSE` |

排序理由：sparse MLA 在 ROCm 上仅有 aiter 的 sparse 实现可用。

**子表 2：`use_mla=True`（非 sparse）**

若 `rocm_aiter_ops.is_mla_enabled()` 为真：

| priority | backend |
|---|---|
| 0 | `ROCM_AITER_MLA` |
| 1 | `TRITON_MLA` |
| 2 | `ROCM_AITER_TRITON_MLA` |

否则（aiter MLA 未启用）：

| priority | backend |
|---|---|
| 0 | `TRITON_MLA` |

排序理由：aiter MLA 启用时优先其专用 kernel，Triton MLA 作为可移植兜底，`ROCM_AITER_TRITON_MLA` 作为第三选项。

**子表 3：非 MLA（非 sparse）**

`backends = []`，按条件依次追加：

| 顺序 | backend | 追加条件 |
|---|---|---|
| 0 | `ROCM_ATTN` | `not use_kv_connector` |
| 1 | `ROCM_AITER_FA` | `rocm_aiter_ops.is_mha_enabled()` |
| 2 | `ROCM_AITER_UNIFIED_ATTN` | `is_aiter_found_and_supported()` |
| 3 | `TRITON_ATTN` | 总是追加 |
| 4 | `TURBOQUANT` | 总是追加 |

注意 `ROCM_ATTN` 仅在 `not use_kv_connector` 时追加：因其使用 `(2, num_blocks, ...)` 的 KV cache 布局，与 KV connector 要求的 blocks-first 布局不兼容。`TRITON_ATTN` 与 `TURBOQUANT` 始终追加，保证至少有可移植兜底。

### XPU 平台

XPU 的 `get_attn_backend_cls`（见 [xpu.py](../../vllm/platforms/xpu.py)）不走 `get_valid_backends`，而是用 if/elif 链硬编码选择，按源码顺序如下：

1. 先调用 `set_kv_cache_layout("NHD")` **强制**全局 KV 布局为 NHD（XPU kernel 仅支持 NHD），并打印日志。
2. 若 `kv_cache_dtype` 以 `turboquant_` 开头 → 返回 `TURBOQUANT`。
3. 若 `use_sparse` → 返回 `XPU_MLA_SPARSE`。
4. 若 `use_mla` → 返回 `TRITON_MLA`。
5. 若用户指定 `selected_backend == TRITON_ATTN` → 返回 `TRITON_ATTN`。
6. 若 `use_mm_prefix` → 回退 `TRITON_ATTN`（XPU 上 Flash Attention 没有 FA4 kernel，无法应用多模态 prefix-LM 双向 mask，故回退到支持 mm_prefix 的 Triton）。
7. 若 `dtype == torch.float32` → 回退 `TRITON_ATTN`（XPU 上 Flash Attention 不支持 float32）。
8. 若用户指定 `selected_backend == FLASH_ATTN` → 返回 `FLASH_ATTN`。
9. 若用户指定了其他 backend → `raise ValueError("Invalid attention backend for xpu, ...")`。
10. 默认（未指定）→ 返回 `FLASH_ATTN`。

### CPU 平台

CPU 的 `get_attn_backend_cls`（见 [cpu.py](../../vllm/platforms/cpu.py)）是单一选择：

- 若用户指定了非 `CPU_ATTN` 的 backend，打印 `"Cannot use X backend on CPU."`，但仍返回 `CPU_ATTN`（不 raise，静默降级）。
- 若 `use_mla` → `raise NotImplementedError("MLA is not supported on CPU.")`。
- 若 `use_sparse` → `raise NotImplementedError("Sparse Attention is not supported on CPU.")`。
- 否则返回 `CPU_ATTN`。

即 CPU 上 MLA/sparse 直接报错不可用，其他情况一律用 `CPU_ATTN`，用户指定的非 CPU backend 会被忽略并告警。

### ViT 独立路径

ViT（视觉 Transformer）模型走独立的 backend 选择路径 `get_vit_attn_backend(head_size, dtype, backend=None)`（见 [cuda.py](../../vllm/platforms/cuda.py)），**不经过** `get_attn_backend` / `AttentionSelectorConfig` / `validate_configuration` 这套流程。

CUDA 上 `get_supported_vit_attn_backends()` 返回候选列表：

- SM80+：`[FLASH_ATTN, TRITON_ATTN, TORCH_SDPA, FLASHINFER]`。
- 低于 SM80：`[FLASH_ATTN, TORCH_SDPA, TRITON_ATTN, FLASHINFER]`（`TORCH_SDPA` 与 `TRITON_ATTN` 顺序不同，提前 `TORCH_SDPA` 作为更稳的兜底）。

选择逻辑：

1. 若调用方显式传入 `backend`，断言其在候选列表中，直接返回。
2. 否则按列表顺序逐个尝试：遇到 `TORCH_SDPA` 直接返回（纯 PyTorch 实现，始终可用）；对其他 backend，取其类并检查 `supports_head_size(head_size)` + `supports_dtype(dtype)` + `supports_compute_capability(cc)` 三项**同时**通过（`ImportError` 视为不通过），第一个全通过的即用。
3. 全部失败时兜底返回 `TORCH_SDPA`。

注意 ViT 路径不做 MLA/sparse/kv_cache_dtype 等检查，因为 ViT 模型不涉及这些特性；其校验维度远少于 decoder attention。

## 完整 backend 清单

vLLM 的 attention backend 在 [registry.py](../../vllm/v1/attention/backends/registry.py) 中以两个枚举登记：`AttentionBackendEnum` 覆盖标准注意力、MLA、Sparse MLA 与量化/ROCm 专用 backend；`MambaAttentionBackendEnum` 则覆盖 SSM/Mamba/Linear/GDN 这一支状态空间家族。每个枚举成员的值是默认的类路径字符串，可在运行时被 `register_backend` 覆盖（`CUSTOM` 成员值设为 `None`，必须先注册才能使用）。下面按功能类别分 5 组列出所有 backend 及其平台、约束与典型用途，便于在排障或定制选择时快速对照。

### 标准注意力 backend

| 枚举名 | 类路径 | 平台/CC | 关键约束 | 典型场景 |
|---|---|---|---|---|
| `FLASH_ATTN` | `vllm.v1.attention.backends.flash_attn.FlashAttentionBackend` | CUDA/XPU，SM80+ | dtype fp16/bf16；head_size 为 8 的倍数且 ≤256（FA4 支持到 512）；block_size 为 16 的倍数；fp8 KV cache 仅 FA3+SM90 或 XPU | CUDA 通用首选 |
| `FLASH_ATTN_DIFFKV` | `vllm.v1.attention.backends.flash_attn_diff_kv.FlashAttentionDiffKVBackend` | CUDA，SM80+ | Q/K/V head_size 不一致 | Q/K/V 维度不等的模型 |
| `TRITON_ATTN` | `vllm.v1.attention.backends.triton_attn.TritonAttentionBackend` | 跨平台（CUDA/XPU/CPU 等） | dtype fp16/bf16/fp32；head_size ≥32；block_size 为 16 的倍数；支持 sink/mm_prefix；KV cache 可为 fp8/int4/int8 per-token-head | 兜底/调试/特殊特性 |
| `TRITON_ATTN_DIFFKV` | `vllm.v1.attention.backends.triton_attn_diff_kv.TritonAttentionDiffKVBackend` | 跨平台 | 同 `TRITON_ATTN`，Q/K/V head_size 不等 | diff-kv 兜底 |
| `FLASHINFER` | `vllm.v1.attention.backends.flashinfer.FlashInferBackend` | CUDA，SM80–SM121 | dtype fp16/bf16；head_size ∈ {64,128,256,512}；支持 fp8/fp8_e4m3/fp8_e5m2/nvfp4 KV cache；SM10 强制 HND 布局 | 高性能/量化 KV cache |
| `FLEX_ATTENTION` | `vllm.v1.attention.backends.flex_attention.FlexAttentionBackend` | 跨平台（torch.compile） | dtype fp16/bf16/fp32；仅 DECODER/ENCODER_ONLY；支持 mm_prefix；不使用 cascade | torch.compile flex attention 路径 |
| `HPC_ATTN` | `vllm.v1.attention.backends.hpc_attn.HpcAttentionBackend` | CUDA，SM90+（Hopper） | dtype fp16/bf16；head_size 固定 128；block_size 固定 64；KV cache auto/bf16/fp8_e4m3；NHD 布局；需安装 `hpc` 模块 | Tencent Hy3 模型专用 |
| `TORCH_SDPA` | _（空字符串占位）_ | — | 仅作为 ViT 路径的 tag | ViT 内部使用 |
| `NO_ATTENTION` | `vllm.v1.attention.backends.no_attention.NoAttentionBackend` | — | attention-free 模型占位 | 无注意力层模型 |
| `CPU_ATTN` | `vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend` | CPU | dtype fp16/bf16/fp32；head_size ∈ {32,64,80,96,112,128,160,192,224,256,512}；block_size 为 16 的倍数；HND 布局 | CPU 平台唯一选择 |

`FLASH_ATTN` 在 SM80+ 的 CUDA 上是默认选择，并通过 `supports_combination` 在 sink（需 SM90+）与 mm_prefix（需 FA4）等组合上做二次校验。`TRITON_ATTN` 是真正的跨平台兜底：它对所有 compute capability 返回支持，并能处理 fp8/int4/int8 per-token-head 等较冷门的量化 KV cache，常在 FA/FlashInfer 因 head_size、sink 或 diff-kv 不可用时被选中。`FLASHINFER` 在 SM100（Blackwell）上会切到 TRTLLM-gen 内核并强制 HND 布局，是 nvfp4 KV cache 的唯一去处；`HPC_ATTN` 则是面向腾讯 Hy3 的窄口径实现，由 [hpc-ops](https://github.com/Tencent/hpc-ops) 提供算子。

### MLA backend（DeepSeek 风格多头潜在注意力）

| 枚举名 | 类路径 | 平台/CC | 关键约束 | 典型场景 |
|---|---|---|---|---|
| `FLASH_ATTN_MLA` | `vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend` | CUDA，SM9（Hopper） | dtype fp16/bf16；KV cache auto/fp16/bf16；block_size 为 16 的倍数；支持 batch invariance | H100 上的默认 MLA |
| `FLASHMLA` | `vllm.v1.attention.backends.mla.flashmla.FlashMLABackend` | CUDA，SM9/SM10 | dtype fp16/bf16；KV cache +fp8/fp8_e4m3；block_size 固定 64 | Hopper/Blackwell MLA |
| `FLASHINFER_MLA` | `vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend` | CUDA，SM10（Blackwell） | dtype fp16/bf16；KV cache +fp8/fp8_e4m3；block_size ∈ {32,64}；qk_nope_head_dim ∈ {64,128,192} | Blackwell 优先选择 |
| `TOKENSPEED_MLA` | `vllm.v1.attention.backends.mla.tokenspeed_mla.TokenspeedMLABackend` | CUDA，SM10 | dtype fp16/bf16；KV cache 仅 fp8/fp8_e4m3；block_size ∈ {32,64}；需安装 `tokenspeed_mla` | R1 dims + FP8 KV，大 batch（≈8+）占优 |
| `TRITON_MLA` | `vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend` | 任意（含 XPU MLA、SM120 MLA） | dtype fp16/bf16；KV cache +fp8/fp8_e4m3；block_size 为 16 的倍数；head_size 不限 | 纯 Triton 跨平台兜底 |
| `CUTLASS_MLA` | `vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend` | CUDA，SM10 | dtype fp16/bf16；KV cache +fp8/fp8_e4m3；block_size 固定 128 | Blackwell CUTLASS MLA |
| `ROCM_AITER_MLA` | `vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend` | ROCm（aiter 启用） | dtype fp16/bf16；KV cache +fp8/fp8_e4m3/fp8_e5m2；block_size 为 1 的倍数（内部 page_size=1） | ROCm aiter MLA |
| `ROCM_AITER_TRITON_MLA` | `vllm.v1.attention.backends.mla.aiter_triton_mla.AiterTritonMLABackend` | ROCm（aiter 启用） | 继承 `AiterMLABackend`，impl 走 Triton 路径 | ROCm aiter Triton MLA 变体 |

MLA backend 的选择高度依赖显卡代际：在 H100（SM90）上默认落到 `FLASH_ATTN_MLA`，在 Blackwell（SM100）上则优先 `FLASHINFER_MLA`；`TRITON_MLA` 因 `supports_compute_capability` 恒为真而成为唯一能覆盖 XPU MLA 与 SM120 MLA 的兜底实现。`TOKENSPEED_MLA` 仅接受 fp8 类 KV cache 且要求 R1 风格 dims，在 batch 较大（≈8+）时相对占优；`CUTLASS_MLA` 与 `FLASHMLA` 则把 block_size 钉死在 128/64，选型时需注意 `cache_config.block_size` 的匹配。

### Sparse MLA backend

| 枚举名 | 类路径 | 平台/CC | 关键约束 | 典型场景 |
|---|---|---|---|---|
| `FLASHMLA_SPARSE` | `vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend` | CUDA，SM9/SM10 | 仅 bf16；head_size 固定 576（512 NoPE + 64 RoPE）；block_size 固定 64；KV cache auto/bf16/fp8_ds_mla | DeepSeek V3.2 风格 sparse MLA |
| `FLASHINFER_MLA_SPARSE` | `vllm.v1.attention.backends.mla.flashinfer_mla_sparse.FlashInferMLASparseTRTLLMBackend` | CUDA，SM10（Blackwell） | dtype fp16/bf16；head_size 固定 576；block_size ∈ {32,64}；qk_nope_head_dim ∈ {128,192}；需模型带 `index_topk`；KV cache auto/fp16/bf16/fp8/fp8_e4m3 | SM10 + 量化 KV cache 的 sparse MLA |
| `FLASHINFER_MLA_SPARSE_SM120` | `vllm.v1.attention.backends.mla.flashinfer_mla_sparse.FlashInferMLASparseSM120Backend` | CUDA，SM12 | 仅 bf16；head_size 固定 576；block_size ∈ {64,256}；KV cache auto/fp8/fp8_e4m3/fp8_ds_mla；需 FlashInfer sparse MLA SM120 API | SM120 专用 sparse MLA |
| `FLASH_ATTN_MLA_SPARSE` | `vllm.v1.attention.backends.mla.flashattn_mla_sparse.FlashAttnMLASparseBackend` | CUDA，SM9 | dtype fp16/bf16；KV cache 仅 auto/fp16/bf16；block_size 固定 64；需模型带 `index_topk`；不支持 DCP | Hopper 上 FA 路径的 sparse MLA |
| `ROCM_AITER_MLA_SPARSE` | `vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseBackend` | ROCm | dtype fp16/bf16；KV cache auto/fp16/bf16/fp8/fp8_e4m3；block_size ∈ {1,64} | ROCm sparse MLA 唯一选择 |
| `XPU_MLA_SPARSE` | `vllm.v1.attention.backends.mla.xpu_mla_sparse.XPUMLASparseBackend` | XPU | dtype fp16/bf16；head_size 固定 576；KV cache auto/fp16/bf16 | XPU sparse MLA |
| `FLASHMLA_SPARSE_DSV4` | `vllm.models.deepseek_v4.sparse_mla.DeepseekV4FlashMLABackend` | CUDA，SM9/SM10 | 仅 bf16；head_size 固定 512（448 NoPE + 64 RoPE）；block_size 固定 256；KV cache auto/fp8_ds_mla；支持 sink；无独立 impl（走 `DeepseekV4Attention`） | DeepSeek V4 模型驱动 |
| `FLASHINFER_MLA_SPARSE_DSV4` | `vllm.models.deepseek_v4.nvidia.flashinfer_sparse.DeepseekV4FlashInferMLASparseBackend` | CUDA，SM10/SM12 | 仅 bf16；head_size 固定 512；block_size 固定 256；KV cache auto/bf16/fp8/fp8_e4m3/fp8_ds_mla；SM10 不用 fp8_ds_mla | DeepSeek V4 NVIDIA 路径 |
| `ROCM_FLASHMLA_SPARSE_DSV4` | `vllm.models.deepseek_v4.amd.rocm.DeepseekV4ROCMAiterMLASparseBackend` | ROCm | 继承 `DeepseekV4FlashMLABackend`；impl 走 ROCm aiter | DeepSeek V4 AMD 路径 |
| `MINIMAX_M3_SPARSE` | `vllm.models.minimax_m3.common.sparse_attention.MiniMaxM3SparseBackend` | 跨平台 | dtype bf16/fp16；head_size 固定 128；block_size 固定 128（page==sparse block）；KV cache bf16/fp8/fp8_e4m3/fp8_e5m2；`is_sparse=True`（非 MLA） | MiniMax M3 模型驱动 |

Sparse MLA 的入口与普通 MLA 不同：它由模型层显式触发 `use_sparse=True`，并在 `supports_combination` 中要求模型配置带 `index_topk` 等稀疏索引字段，因此普通标准 MLA 的自动选择流程不会误选这些 backend。其中 `FLASHMLA_SPARSE_DSV4` / `FLASHINFER_MLA_SPARSE_DSV4` / `ROCM_FLASHMLA_SPARSE_DSV4` 三者是 DeepSeek V4 模型驱动的，类路径位于 `vllm/models/deepseek_v4/` 下，且没有独立 impl 类——注意力直接在 `DeepseekV4Attention.forward` 中执行；`MINIMAX_M3_SPARSE` 同样是模型驱动，但 `is_mla` 为假、仅 `is_sparse` 为真，本质是 block-sparse GQA 而非潜在注意力。

### SSM / Mamba / Linear backend（来自 MambaAttentionBackendEnum）

| 枚举名 | 类路径 | 平台/CC | 关键约束 | 典型场景 |
|---|---|---|---|---|
| `MAMBA1` | `vllm.v1.attention.backends.mamba1_attn.Mamba1AttentionBackend` | 跨平台 | `is_ssm=True`；走 `MambaSpec` 状态缓存 | Mamba1 SSM |
| `MAMBA2` | `vllm.v1.attention.backends.mamba2_attn.Mamba2AttentionBackend` | 跨平台 | `is_ssm=True`；支持 chunked prefill 与 initial_states | Mamba2 SSM |
| `SHORT_CONV` | `vllm.v1.attention.backends.short_conv_attn.ShortConvAttentionBackend` | 跨平台 | `is_ssm=True` | 短卷积（short conv）层 |
| `LINEAR` | `vllm.v1.attention.backends.linear_attn.LinearAttentionBackend` | 跨平台 | `is_ssm=True`；`MambaSpec` 状态缓存；CUDA graph 仅支持 uniform single-token decode | 线性注意力（如 Bailing） |
| `GDN_ATTN` | `vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend` | 跨平台 | `is_ssm=True`；支持 spec decode 元数据 | 扩散模型 GDN |
| `CUSTOM` | _（`None`，需注册）_ | — | 占位符，需 `register_backend(..., is_mamba=True)` 注册 | 第三方 SSM backend |

这一组 backend 不经过 `get_attn_backend`，而是通过 [selector.py](../../vllm/v1/attention/selector.py) 中的 `get_mamba_attn_backend(mamba_type)` 独立入口按需懒加载，并在 `VLLM_BATCH_INVARIANT` 开启时校验 `supports_batch_invariance`。它们共享 `BaseMambaAttentionMetadata` 体系与 `MambaSpec` 状态缓存（`LinearAttentionBackend` 与 `GDNAttentionBackend` 因不复用 base metadata 而自行定义 dataclass），`is_ssm()` 一律返回 `True`，这是区分标准注意力与状态空间家族的关键标志。`CUSTOM` 成员值同样为 `None`，必须先以 `is_mamba=True` 调用 `register_backend` 才能解析出类路径。

### 量化/ROCm 专用 backend

| 枚举名 | 类路径 | 平台/CC | 关键约束 | 典型场景 |
|---|---|---|---|---|
| `TURBOQUANT` | `vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend` | XPU 与 CUDA 均可触发 | dtype fp16/bf16；KV cache 仅 `turboquant_*`（k8v4/4bit_nc/k3v4_nc/3bit_nc）；block_size ∈ {16,32,64,128}；仅 DECODER；K+V 打包成单 slot，不复用标准 cache 形状 | TurboQuant KV cache 压缩 |
| `ROCM_ATTN` | `vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend` | ROCm | dtype fp16/bf16/fp32；head_size ∈ {32,64,80,96,128,160,192,224,256}；block_size 为 16 的倍数；KV cache auto/fp16/bf16/fp8/fp8_e4m3/fp8_e5m2；`(2, num_blocks, ...)` 布局；不支持 sink 与 KV connector | ROCm 通用 MHA |
| `ROCM_AITER_FA` | `vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend` | ROCm（mi3xx） | dtype fp16/bf16；head_size ∈ {64,128,256}；block_size ∈ {16,32}；KV cache auto/fp16/bf16/fp8/fp8_e4m3/fp8_e5m2；仅 DECODER；`get_name` 返回 `FLASH_ATTN` | ROCm aiter MHA |
| `ROCM_AITER_UNIFIED_ATTN` | `vllm.v1.attention.backends.rocm_aiter_unified_attn.RocmAiterUnifiedAttentionBackend` | ROCm（继承 `RocmAttentionBackend`） | 仅 bf16；head_size ≥32；block_size 为 16 的倍数（preferred 64）；KV cache auto/bf16/fp8/fp8_e4m3；不支持 non-causal | ROCm aiter 统一 attention |
| `CUSTOM` | _（`None`，需注册）_ | — | 占位符，需 `register_backend` 注册 | 第三方量化/ROCm backend |

`TURBOQUANT` 是为 TurboQuant KV cache 压缩定制的：它把 K、V 打包进单个 interleaved slot（`get_kv_cache_shape` 不带前导 2），因此不能与其它 backend 共享 cache 张量，且 `supports_kv_cache_dtype` 仅在 dtype 以 `turboquant_` 开头时返回真。ROCm 系列三者中，`ROCM_ATTN` 是通用兜底，但因其 `(2, num_blocks, ...)` 的布局与 KV connector 要求的 blocks-first 不兼容（`supports_kv_connector` 返回假），且原生 C++ kernel 不支持 sink；`ROCM_AITER_FA` 仅在 mi3xx 上启用，注意它的 `get_name` 实际返回 `FLASH_ATTN`；`ROCM_AITER_UNIFIED_ATTN` 继承自 `RocmAttentionBackend` 但改用 aiter 统一算子并默认 block_size 64，`CUSTOM` 则是给第三方实现预留的注册口。

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

影响 backend 选择或运行行为的五种方式：

- `--attention-backend <NAME>` CLI 参数：解析后写入 `vllm_config.attention_config.backend`（`AttentionBackendEnum`），走平台 `get_attn_backend_cls` 的"显式指定"分支，`validate_configuration` 校验失败时直接 `raise ValueError`。也可通过 `--attention-config.backend` / `-ac.backend` 以结构化配置传入，或通过 Python API `LLM(attention_backend="FLASH_ATTN")` 指定。`--attention-backend` 与 `--attention-config.backend` 互斥。
- `VLLM_KV_CACHE_LAYOUT` 环境变量：预设 KV cache 布局（`NHD` / `HND`），由 `get_kv_cache_layout()` 读取。XPU 平台会强制覆盖为 `NHD`，FlashInfer 在 SM100 上会通过 `get_required_kv_cache_layout()` 强制 `HND`。
- `VLLM_BATCH_INVARIANT` 环境变量：启用 batch invariance 模式，进入 selector 后触发 `supports_batch_invariance` 校验；此外 `Attention.__init__` 会对 `FLASHINFER` / `TRITON_MLA` 自动关闭 prefix caching（该组合当前不支持）。
- `register_backend(Backend.X, "module.Cls")`：[registry.py](../../vllm/v1/attention/backends/registry.py) 提供的运行时覆盖，把枚举成员重定向到第三方实现路径，供扩展注册自定义 backend（或替换默认实现）。
- `Attention(attn_backend=MyBackend)`：模型代码直接传入 backend 类，**绕过** selector 与 `AttentionSelectorConfig`，由模型作者完全自负其责。

第一个走显式分支（校验失败抛错），中间两个影响布局与批不变性等横切行为，后两个分别在运行时注册与模型构造层旁路选择器。

### ViT 独立路径

ViT（encoder-only vision tower）的 backend 选择走独立路径，不经过 `get_attn_backend` 与 `AttentionSelectorConfig`，入口是 `current_platform.get_vit_attn_backend(head_size, dtype, backend=None)`（[cuda.py](../../vllm/platforms/cuda.py)）。

CUDA 平台的 `get_supported_vit_attn_backends` 在 SM80+ 返回 `[FLASH_ATTN, TRITON_ATTN, TORCH_SDPA, FLASHINFER]`，低于 SM80 时返回相同成员但顺序不同（`[FLASH_ATTN, TORCH_SDPA, TRITON_ATTN, FLASHINFER]`）。`get_vit_attn_backend` 按列表顺序逐个尝试 `supports_head_size` + `supports_dtype` + `supports_compute_capability`，第一个全部通过即采用；遇 `TORCH_SDPA` 直接返回（视为兜底）；全部失败时回退 `TORCH_SDPA`。ROCm 平台（[rocm.py](../../vllm/platforms/rocm.py)）的 `get_supported_vit_attn_backends` 返回 `[FLASH_ATTN, ROCM_AITER_FA, TRITON_ATTN, TORCH_SDPA]`，并在 `get_vit_attn_backend` 内按 gfx 架构（gfx9 / RDNA3+）做分支选择。

ViT 走独立路径的根本原因：vision tower 不使用 KV cache、不需要 paged attention，调度与内存模型与 decoder attention 完全不同，因此用独立的 backend 选择与独立的 attention 层类型（`ENCODER_ONLY`）实现。
