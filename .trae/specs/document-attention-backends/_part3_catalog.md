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
