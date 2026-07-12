# vLLM Attention Backend 设计文档规范

## Why

vLLM 的 attention 子系统是该框架最复杂、最关键的部分之一。它通过统一的抽象层接入
多种硬件平台（CUDA / ROCm / XPU / CPU）和数十种 attention 实现（FlashAttention、
FlashInfer、Triton、MLA 系列、Mamba/SSM 系列等），并按"配置 + 优先级 + 能力校验"
的方式自动选择最佳 backend。当前缺少一份系统讲解整体框架与"抽象 attention 对象如何
接入不同 backend"的中文文档，新贡献者和使用者难以快速建立心智模型。

本规范定义一份面向开发者/高级使用者的设计文档，讲清楚三件事：

1. vLLM 支持哪些 attention backend；
2. 在什么场景/配置下会选择哪种 backend；
3. 框架层抽象出的 `Attention`/`AttentionBackend`/`AttentionImpl`/`AttentionMetadataBuilder`
   对象如何协作，让同一份模型代码可以无缝接入不同 backend。

## What Changes

- 新增一份 Markdown 设计文档，输出到 `docs/design/attention_backends.md`，
  覆盖以下章节：
  - 概述与设计目标
  - 核心抽象（`AttentionBackend` / `AttentionImpl` / `AttentionMetadataBuilder`
    / `AttentionLayer` / `CommonAttentionMetadata` 五大对象及其职责边界）
  - backend 注册与发现（`AttentionBackendEnum` / `MambaAttentionBackendEnum` /
    `register_backend` 覆盖机制）
  - backend 选择机制（`get_attn_backend` → 平台 `get_attn_backend_cls` →
    优先级表 → `validate_configuration`）
  - 各硬件平台的优先级表与触发条件（CUDA / ROCm / XPU / CPU）
  - 完整 backend 清单与适用场景表格（标准 attention、MLA、Sparse MLA、
    SSM/Mamba/Linear、ViT、Quant 专用等）
  - 端到端调用链：从模型 `Attention(...)` 构造 → `model_runner` metadata 构建 →
    forward 执行
  - KV cache 布局、CUDA Graph 支持、特性矩阵等横切关注点
  - 用户如何通过 `--attention-backend` / 环境变量影响选择
- 文档不涉及具体 kernel 实现细节（如 softmax 数值算法），聚焦"框架如何接入"。
- 文档使用中文撰写，符合现有 `docs/design/` 目录风格。
- **BREAKING**：无。仅新增文档。

## Impact

- Affected specs: 无既有 spec 受影响（本仓库未使用 spec-driven 流程，本次为独立文档）。
- Affected code:
  - `vllm/v1/attention/backend.py` — 抽象基类定义
  - `vllm/v1/attention/selector.py` — `get_attn_backend` 入口
  - `vllm/v1/attention/backends/registry.py` — backend 枚举与注册
  - `vllm/platforms/{cuda,rocm,xpu,cpu}.py` — 平台级优先级与选择
  - `vllm/v1/attention/backends/*.py` — 各具体 backend
  - `vllm/v1/attention/backends/mla/*.py` — MLA 子家族
  - `vllm/model_executor/layers/attention/attention.py` — `Attention` 层接入点
  - `vllm/v1/worker/gpu_model_runner.py` — metadata builder 编排
- Affected docs: 新增 `docs/design/attention_backends.md`。

## ADDED Requirements

### Requirement: 设计文档须覆盖核心抽象层

文档 SHALL 包含一节，用类图/列表讲清楚以下对象的职责与边界：

- `AttentionBackend`（抽象基类）：声明 `get_name` / `get_impl_cls` /
  `get_builder_cls` / `get_kv_cache_shape` 等静态能力，以及
  `supports_*` / `validate_configuration` 等能力查询方法。
- `AttentionImpl` / `MLAAttentionImpl` / `SparseMLAAttentionImpl`：
  per-layer 的实际计算实现，由 `Attention` 层在 `__init__` 中通过
  `backend.get_impl_cls()` 实例化。
- `AttentionMetadataBuilder`：per-step 构造 `AttentionMetadata`，由
  `model_runner` 持有并调用 `build()`。
- `Attention`（`nn.Module`）：模型代码直接使用的层，封装 backend 选择、
  impl 实例化、forward dispatch。
- `CommonAttentionMetadata`：跨 backend 共享的 per-batch 元数据。

#### Scenario: 读者理解对象关系

- **WHEN** 读者阅读"核心抽象"章节
- **THEN** 能回答"`Attention` 层如何拿到 impl 和 builder"以及
  "backend 类本身是否持有运行时状态"这两个问题

### Requirement: 设计文档须覆盖 backend 选择机制

文档 SHALL 包含一节，逐步说明从模型配置到最终 backend 类的完整路径：

1. `Attention.__init__` 调用 `get_attn_backend(...)`；
2. `selector.get_attn_backend` 组装 `AttentionSelectorConfig`，缓存调用
   `_cached_get_attn_backend`；
3. 委托给 `current_platform.get_attn_backend_cls(...)`；
4. 若用户显式指定 backend（`--attention-backend` / `attention_config.backend`），
   优先校验该 backend，失败则报错；
5. 否则取平台优先级表 `_get_backend_priorities(...)`，依次用
   `validate_configuration` 过滤，选 priority 最小的合法 backend；
6. `resolve_obj_by_qualname` 加载类，必要时调整 KV cache layout。

#### Scenario: 用户指定了不兼容的 backend

- **WHEN** 用户通过 `--attention-backend` 指定了一个对当前 dtype/head_size
  不支持的 backend
- **THEN** 文档应说明系统会抛出 `ValueError` 并列出 invalid reasons，而非静默回退

### Requirement: 设计文档须包含各平台优先级表

文档 SHALL 以表格/列表形式给出以下四个平台在"MLA / 非 MLA / Sparse MLA"
分支下的 backend 优先级顺序，并标注每条优先级表的触发条件（如 compute capability
major、`num_heads`、`kv_cache_dtype` 是否量化、`rocm_aiter_ops.is_mla_enabled()` 等）：

- CUDA（`vllm/platforms/cuda.py::_get_backend_priorities`）
- ROCm（`vllm/platforms/rocm.py::_get_backend_priorities`）
- XPU（`vllm/platforms/xpu.py::get_attn_backend_cls` 中的 if/elif 链）
- CPU（`vllm/platforms/cpu.py::get_attn_backend_cls`，仅 `CPU_ATTN`）

#### Scenario: 读者想预判默认 backend

- **WHEN** 读者在 H100（SM90）上跑 DeepSeek-V2（MLA）且未指定 backend
- **THEN** 能从文档表格中查到优先级顺序为
  `FLASH_ATTN_MLA > FLASHMLA > FLASHINFER_MLA > TRITON_MLA > ...`，
  并理解为何最终落到 `FLASH_ATTN_MLA`

### Requirement: 设计文档须包含完整 backend 清单

文档 SHALL 列出 `AttentionBackendEnum` 与 `MambaAttentionBackendEnum` 的所有成员，
对每个 backend 给出：

- 枚举名与类路径
- 适用平台 / compute capability 限制
- 是否 MLA / Sparse / SSM
- 支持的 dtype / kv_cache_dtype / head_size / block_size 关键约束
- 典型适用场景（一句话）

#### Scenario: 读者查找特定 backend

- **WHEN** 读者想知道 `FLASHINFER_MLA_SPARSE` 何时被选中
- **THEN** 能在清单中找到该 backend 并看到它仅用于 SM90 + 量化 KV cache 的
  sparse MLA 场景

### Requirement: 设计文档须包含端到端调用链

文档 SHALL 用一节（含序列化文字描述）讲清楚一次 forward 的完整路径：

1. 模型代码：`self.attn = Attention(num_heads, head_size, ...)`；
2. `Attention.__init__` → `get_attn_backend` → 选中 backend 类；
3. `backend.get_impl_cls()` 实例化 `impl`；
4. `model_runner` 在初始化时为每个 kv_cache_group 调用
   `backend.get_builder_cls()` 实例化 `AttentionMetadataBuilder`；
5. 每个 step：scheduler 产出请求 → `model_runner` 构造
   `CommonAttentionMetadata` → `builder.build(...)` 得到 per-layer
   `AttentionMetadata` → 模型 `forward` 调用 `self.attn(q,k,v, kv_cache, attn_meta)`；
6. `Attention.forward` → `impl.forward(layer, q, k, v, kv_cache, meta, ...)`；
7. KV cache 更新与 attention 计算的关系（`forward_includes_kv_cache_update`）。

#### Scenario: 读者理解 builder 与 impl 的分工

- **WHEN** 读者阅读"端到端调用链"章节
- **THEN** 能说明 `AttentionMetadataBuilder` 在 CPU 侧每步构造元数据、
  `AttentionImpl` 在 GPU 侧消费元数据执行 kernel 的分工关系

### Requirement: 设计文档须覆盖横切关注点

文档 SHALL 包含一节，简要讲清楚：

- KV cache 布局（`NHD` vs `HND`、`get_kv_cache_shape` /
  `get_kv_cache_stride_order`、`get_required_kv_cache_layout`）；
- CUDA Graph 支持（`AttentionCGSupport` 四级、`get_cudagraph_support`、
  builder 的 `build_for_cudagraph_capture`）；
- 特性能力矩阵（`supports_sink` / `supports_mm_prefix` / `supports_non_causal` /
  `supports_batch_invariance` / `supports_kv_connector` / `supports_per_head_quant_scales`
  等的字段含义）；
- 用户可调旋钮（`--attention-backend`、`VLLM_ATTENTION_BACKEND`、
  `VLLM_KV_CACHE_LAYOUT`、`VLLM_BATCH_INVARIANT`、
  `register_backend` 运行时覆盖）；
- ViT attention 的独立选择路径（`get_vit_attn_backend`）。

#### Scenario: 读者想强制使用某 backend

- **WHEN** 读者想强制使用 `TRITON_ATTN` 排查问题
- **THEN** 能从文档找到 `--attention-backend TRITON_ATTN` 或
  `VLLM_ATTENTION_BACKEND=TRITON_ATTN` 的用法，并理解校验失败会直接报错

## MODIFIED Requirements

无（本规范为新增文档，不修改既有 requirement）。

## REMOVED Requirements

无。
