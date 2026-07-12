# Tasks

本任务清单描述如何按 `spec.md` 产出 `docs/design/attention_backend_framework.md` 文档
（原计划名 `attention_backends.md`，因该名已被 `tools/pre_commit/generate_attention_backend_docs.py`
自动生成占用，故改用 `attention_backend_framework.md`）。
文档撰写以"先骨架后填充"的方式进行，每一节都可独立验证。

- [x] Task 1: 创建文档骨架与"概述与设计目标"章节
  - [x] SubTask 1.1: 新建 `docs/design/attention_backend_framework.md`，写入标题、
    SPDX 头部、目录占位与"概述"段落（说明 attention 子系统在 vLLM 中的位置、
    抽象目标：平台无关 / 多 backend / 自动选择 / 可强制覆盖）。
  - [x] SubTask 1.2: 写"设计目标"小节，列出平台无关性、可插拔 backend、
    能力驱动选择、运行时覆盖四条原则，并对应到
    `vllm/v1/attention/backend.py` 与 `vllm/v1/attention/selector.py`。

- [x] Task 2: 撰写"核心抽象层"章节
  - [x] SubTask 2.1: 写 `AttentionBackend` 抽象基类，列出关键 staticmethod
    (`get_name` / `get_impl_cls` / `get_builder_cls` / `get_kv_cache_shape` /
    `get_kv_cache_stride_order` / `get_supported_kernel_block_sizes`) 与
    classmethod 能力查询 (`supports_*` / `validate_configuration` /
    `get_required_kv_cache_layout`)，引用
    [vllm/v1/attention/backend.py](file:///workspace/vllm/v1/attention/backend.py)。
  - [x] SubTask 2.2: 写三种 `AttentionImpl` 变体
    (`AttentionImpl` / `MLAAttentionImpl` / `SparseMLAAttentionImpl`)，
    说明 `forward` / `forward_mha` / `forward_mqa` 的接口差异，
    以及 `AttentionImplBase` 中 DCP/PCP/lse 相关字段含义。
  - [x] SubTask 2.3: 写 `AttentionMetadataBuilder`，列出 `build` /
    `build_for_cudagraph_capture` / `build_for_drafting` /
    `use_cascade_attention` / `update_block_table` 与
    `AttentionCGSupport` 四级 cudagraph 支持枚举。
  - [x] SubTask 2.4: 写 `Attention` (`nn.Module`) 层，说明它在 `__init__`
    中调用 `get_attn_backend`、`backend.get_impl_cls()`、注册到
    `static_forward_context` 的流程；引用
    [vllm/model_executor/layers/attention/attention.py](file:///workspace/vllm/model_executor/layers/attention/attention.py)。
  - [x] SubTask 2.5: 写 `CommonAttentionMetadata` 字段表
    (`query_start_loc` / `seq_lens` / `block_table_tensor` / `slot_mapping` /
    `is_prefilling` / `rswa_prefix_lens` 等) 与 `unpadded()` 的用途。

- [x] Task 3: 撰写"backend 注册与发现"章节
  - [x] SubTask 3.1: 说明 `AttentionBackendEnum` / `MambaAttentionBackendEnum`
    的设计：枚举值即默认类路径，`get_class()` / `get_path()` 尊重 override；
    引用 [vllm/v1/attention/backends/registry.py](file:///workspace/vllm/v1/attention/backends/registry.py)。
  - [x] SubTask 3.2: 说明 `register_backend()` 装饰器/直调两种用法与
    `_ATTN_OVERRIDES` / `_MAMBA_ATTN_OVERRIDES` 全局表，给出第三方扩展示例。
  - [x] SubTask 3.3: 说明 `subclass_attention_backend` /
    `subclass_attention_backend_with_overrides` 工具函数的用途。

- [x] Task 4: 撰写"backend 选择机制"章节
  - [x] SubTask 4.1: 画文字版流程图，从 `Attention.__init__` →
    `selector.get_attn_backend` → `AttentionSelectorConfig` →
    `@cache _cached_get_attn_backend` →
    `current_platform.get_attn_backend_cls`；引用
    [vllm/v1/attention/selector.py](file:///workspace/vllm/v1/attention/selector.py)。
  - [x] SubTask 4.2: 说明显式 backend 分支（`selected_backend is not None`）：
    先 `validate_configuration`，失败直接 `ValueError`，不回退。
  - [x] SubTask 4.3: 说明自动选择分支：`get_valid_backends` →
    `validate_configuration` 过滤 → `min(priority)` 选择；以及
    `--block-size` 排除更高优先级 backend 时的告警逻辑。
  - [x] SubTask 4.4: 说明 `get_required_kv_cache_layout` 触发
    `set_kv_cache_layout` 的副作用与日志。

- [x] Task 5: 撰写"各平台优先级表"章节
  - [x] SubTask 5.1: CUDA 优先级表（`vllm/platforms/cuda.py::_get_backend_priorities`）：
    分 MLA(SM100/SM120/其他) 与非 MLA(SM100/其他) 共 5 张子表，
    标注 `num_heads` / `kv_cache_dtype` 量化 / sparse_backends 排序条件。
  - [x] SubTask 5.2: ROCm 优先级表（`vllm/platforms/rocm.py::_get_backend_priorities`）：
    分 sparse / MLA(`is_mla_enabled`) / 非 MLA(kv_connector 影响) 三张子表。
  - [x] SubTask 5.3: XPU 选择链（`vllm/platforms/xpu.py::get_attn_backend_cls`）：
    按turboquant / sparse / mla / triton / mm_prefix / fp32 / flash 顺序列出。
  - [x] SubTask 5.4: CPU 选择（强制 `CPU_ATTN`，MLA/Sparse 直接报错）；
    ViT 独立路径 `get_vit_attn_backend` 单列一节。

- [x] Task 6: 撰写"完整 backend 清单"章节
  - [x] SubTask 6.1: 标准注意力 backend 表（`FLASH_ATTN` / `FLASH_ATTN_DIFFKV` /
    `TRITON_ATTN` / `TRITON_ATTN_DIFFKV` / `FLASHINFER` / `FLEX_ATTENTION` /
    `HPC_ATTN` / `TORCH_SDPA` / `NO_ATTENTION` / `CPU_ATTN`）。
  - [x] SubTask 6.2: MLA backend 表（`FLASH_ATTN_MLA` / `FLASHMLA` /
    `FLASHINFER_MLA` / `TOKENSPEED_MLA` / `TRITON_MLA` / `CUTLASS_MLA` /
    `ROCM_AITER_MLA` / `ROCM_AITER_TRITON_MLA`）。
  - [x] SubTask 6.3: Sparse MLA backend 表（`FLASHMLA_SPARSE` /
    `FLASHINFER_MLA_SPARSE` / `FLASHINFER_MLA_SPARSE_SM120` /
    `FLASH_ATTN_MLA_SPARSE` / `ROCM_AITER_MLA_SPARSE` / `XPU_MLA_SPARSE` /
    DSV4 系列 / `MINIMAX_M3_SPARSE`）。
  - [x] SubTask 6.4: SSM/Mamba/Linear backend 表（`MAMBA1` / `MAMBA2` /
    `SHORT_CONV` / `LINEAR` / `GDN_ATTN`）。
  - [x] SubTask 6.5: 量化/专用 backend 表（`TURBOQUANT` / `ROCM_ATTN` /
    `ROCM_AITER_FA` / `ROCM_AITER_UNIFIED_ATTN` / `CUSTOM`）。
  - [x] SubTask 6.6: 每张表统一列：枚举名 / 类路径 / 平台与 CC 限制 /
    MLA/Sparse/SSM 标记 / 关键 dtype·head_size·block_size 约束 / 一句话场景。

- [x] Task 7: 撰写"端到端调用链"章节
  - [x] SubTask 7.1: 模型构造阶段：`Attention(...)` → backend 选择 →
    `impl = backend.get_impl_cls()(...)`。
  - [x] SubTask 7.2: runner 初始化阶段：`gpu_model_runner` 为每个
    kv_cache_group 调用 `backend.get_builder_cls()` 实例化 builder；
    说明 `attention_backends: list[set[type[AttentionBackend]]]` 的结构。
  - [x] SubTask 7.3: 每 step 执行阶段：scheduler → `CommonAttentionMetadata` →
    `builder.build(common_prefix_len, common_attn_metadata)` →
    `static_forward_context[layer_name].attn_metadata = meta` →
    模型 forward → `Attention.forward` → `impl.forward`。
  - [x] SubTask 7.4: KV cache 更新语义：说明
    `forward_includes_kv_cache_update` True/False 两种 backend 的差异
    （实测 FlashAttn 与 Triton 均为 False，需外部 `do_kv_cache_update`，
    基类默认 True 时由 impl 内部完成）。

- [x] Task 8: 撰写"横切关注点"章节
  - [x] SubTask 8.1: KV cache 布局：`NHD` vs `HND`、
    `get_kv_cache_shape` 与 `get_kv_cache_stride_order` 的关系、
    `indexes_kv_by_block_stride` 的含义、`VLLM_KV_CACHE_LAYOUT` 环境变量。
  - [x] SubTask 8.2: CUDA Graph 支持：`AttentionCGSupport`
    (ALWAYS/UNIFORM_BATCH/UNIFORM_SINGLE_TOKEN_DECODE/NEVER) 四级、
    `get_cudagraph_support` 如何在 `gpu_model_runner` 中聚合多 backend 取最小值、
    `build_for_cudagraph_capture` 钩子。
  - [x] SubTask 8.3: 特性能力矩阵表：逐项解释
    `supports_sink` / `supports_alibi_sqrt` / `supports_mm_prefix` /
    `is_sparse` / `supports_per_head_quant_scales` / `supports_non_causal` /
    `supports_batch_invariance` / `supports_kv_connector` /
    `supports_attn_type` / `supports_compute_capability` /
    `supports_combination` 的语义与典型消费者。
  - [x] SubTask 8.4: 用户旋钮（实际五种，原文档误写六种）：
    `--attention-backend` / `--attention-config.backend` / `LLM(attention_backend=...)` /
    `VLLM_KV_CACHE_LAYOUT` / `VLLM_BATCH_INVARIANT` /
    `register_backend` 运行时覆盖 / 自定义 `Attention` 子类传入 `attn_backend=` 参数。
    **修正**：源码中并不存在 `VLLM_ATTENTION_BACKEND` 环境变量
    （`grep VLLM_ATTENTION_BACKEND vllm/` 返回 no matches），
    此前 sub-agent A/B 写入该字段属误报，最终文档已删除并改为五种。
  - [x] SubTask 8.5: ViT attention 独立路径：`get_vit_attn_backend`
    按 head_size/dtype/CC 选择，`TORCH_SDPA` 作为兜底。

- [x] Task 9: 通读校对与目录补全
  - [x] SubTask 9.1: 在文档开头补全完整目录（TOC），与各章节标题对齐。
  - [x] SubTask 9.2: 检查所有代码引用都使用相对路径 markdown 链接
    （如 `[backend.py](../../vllm/v1/attention/backend.py)`），
    不使用裸 `line N`。
  - [x] SubTask 9.3: 通读全篇，确认"框架如何接入不同 backend"这一主线在每节
    都有体现，删除与 kernel 内部实现相关的细节。

# Task Dependencies

- Task 2 依赖 Task 1（需先有骨架文件）。
- Task 3–8 可在 Task 1 完成后并行展开（各章节相对独立）。
- Task 9 依赖 Task 2–8 全部完成。
