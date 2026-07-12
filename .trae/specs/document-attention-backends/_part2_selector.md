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

当 `attention_config.backend` 不为 None（用户通过 `--attention-backend` 或环境变量 `VLLM_ATTENTION_BACKEND` 指定）时，平台 `get_attn_backend_cls` 先单独校验该 backend：

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
