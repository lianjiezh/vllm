# vLLM 数据并行(DP)切分实现深度解析

本文档深入解析 vLLM 中数据并行(Data Parallelism)的完整实现机制，从请求分配、批处理切分、通信协调到最终输出的全流程。

---

## 一、系统架构概览

### 1.1 核心组件

```
┌──────────────────────────────────────────────────────────────────────┐
│                         前端 API 服务器集群                           │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ API Server 0 │  │ API Server 1 │  │ API Server N │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼─────────────────┼─────────────────┼───────────────────────┘
          │                 │                 │
          └─────────────────┴─────────────────┘
                            │
                    ┌───────▼────────┐
                    │  DPCoordinator  │  ← 负责请求波次协调与负载均衡
                    └───────┬────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌───────▼────────┐  ┌───────▼────────┐
│   Engine 0    │  │   Engine 1    │  │   Engine K    │  ← DP 引擎
│  (DP Rank 0)  │  │  (DP Rank 1)  │  │  (DP Rank K)  │
│               │  │               │  │               │
│ ┌───────────┐ │  │ ┌───────────┐ │  │ ┌───────────┐ │
│ │ Scheduler │ │  │ │ Scheduler │ │  │ │ Scheduler │ │
│ └─────┬─────┘ │  │ └─────┬─────┘ │  │ └─────┬─────┘ │
│       │       │  │       │       │  │       │       │
│ ┌─────▼─────┐ │  │ ┌─────▼─────┐ │  │ ┌─────▼─────┐ │
│ │ Model     │ │  │ │ Model     │ │  │ │ Model     │ │
│ │ Runner    │ │  │ │ Runner    │ │  │ │ Runner    │ │
│ └───────────┘ │  │ └───────────┘ │  │ └───────────┘ │
└───────────────┘  └───────────────┘  └───────────────┘
```

### 1.2 DP 模式类型

vLLM 支持三种 DP 部署模式：

| 模式 | 说明 | 负载均衡 | 通信方式 |
|------|------|----------|----------|
| **内部 LB 模式** | 一个客户端管理所有 DP 引擎 | 内置轮询/权重 | ZMQ + AllReduce |
| **混合 LB 模式** | 客户端管理部分引擎，部分外部管理 | 部分内置 | ZMQ + AllReduce |
| **外部 LB 模式** | 每个引擎有独立客户端 | 外部系统 | ZMQ |

---

## 二、请求分配流程

### 2.1 数据并行协调器 (DPCoordinator)

**文件位置**: [`vllm/v1/engine/coordinator.py`](file:///workspace/vllm/v1/engine/coordinator.py#L23-L527)

DPCoordinator 是数据并行的核心协调组件，主要功能：

1. **负载均衡统计**：收集各 DP 引擎的 [waiting, running] 请求队列长度
2. **请求波次管理**：同步 DP 引擎的运行/暂停状态
3. **波次唤醒**：当有新请求到达时，广播 START_DP_WAVE 信号

#### 核心数据结构

```python
class EngineState:
    def __init__(self):
        self.request_counts = [0, 0]  # [num_waiting_reqs, num_running_reqs]
```

#### 协调流程

```
前端 API Server (发送请求)
    │
    ▼
DPCoordinator (发布统计信息)
    │
    ▼
所有前端订阅最新 [waiting, running] 计数
    │
    ▼
前端选择负载最低的 DP 引擎发送请求
```

### 2.2 请求分配与波次机制

**关键概念**：Request Wave（请求波次）

1. **波次编号**：全局递增计数器 `current_wave`，每轮从运行到暂停过渡后 +1
2. **全局状态同步**：通过 AllReduce 检查是否所有 DP 引擎都无待处理请求
3. **唤醒信号**：START_DP_WAVE 广播信号，通知所有引擎进入下一轮

**代码位置**: [`coordinator.py:204-L285`](file:///workspace/vllm/v1/engine/coordinator.py#L204-L285)

```python
def process_input_socket(...):
    current_wave = 0
    engines_running = False
    
    while True:
        # 1. 收集各引擎的队列统计信息
        if output_back in events:
            eng_index = outputs.engine_index
            scheduler_stats = outputs.scheduler_stats
            stats[0] = scheduler_stats.num_waiting_reqs
            stats[1] = scheduler_stats.num_running_reqs
        
        # 2. 检查波次完成（仅 DP Rank 0 报告）
        if (wave := outputs.wave_complete) is not None:
            if current_wave <= wave:
                new_wave = wave + 1
                current_wave = new_wave
                engines_running = False
        
        # 3. 新请求到达 → 唤醒所有引擎
        if publish_front in events:
            decoded = msgspec.msgpack.decode(buffer)
            engine_to_exclude, wave = decoded
            if not engines_running:
                engines_running = True
                self._send_start_wave(publish_back, current_wave, 
                                      engine_to_exclude)
```

---

## 三、DP 通信与同步机制

### 3.1 通信基础设施

**文件位置**: [`vllm/v1/worker/dp_utils.py`](file:///workspace/vllm/v1/worker/dp_utils.py#L1-L225)

#### 通信模式选择

```python
def _get_device_and_group(parallel_config: ParallelConfig):
    """
    返回用于 DP 通信的设备和通信组
    - 默认使用 GPU (NCCL)
    - 通过环境变量可以切换到 CPU 同步（避免 GPU 同步点开销）
    """
    device = get_dp_group().device
    group = get_dp_group().device_group
    
    if parallel_config.disable_nccl_for_dp_synchronization:
        # 使用 CPU AllReduce，减少 GPU 同步
        device = "cpu"
        group = get_dp_group().cpu_group
    return device, group
```

#### 核心同步操作: `coordinate_batch_across_dp`

这是 DP 批处理的协调入口函数：

```python
def coordinate_batch_across_dp(
    num_tokens_unpadded: int,
    allow_microbatching: bool,
    parallel_config: ParallelConfig,
    num_tokens_padded: int | None = None,
    uniform_decode: bool | None = None,
    cudagraph_mode: int = 0,
) -> tuple[bool, torch.Tensor | None, int]:
    """
    所有 DP 引擎协同确定：
    1. 是否进行微批处理 (Micro-batching, DBO)
    2. 每个 DP 引擎的 token 数量（是否需要 DP padding）
    3. 同步的 CUDA Graph 模式
    
    返回:
    - should_ubatch: 是否使用微批处理
    - num_tokens_after_padding: 各 DP rank 填充后的 token 数
    - synced_cudagraph_mode: 同步后的 CUDA Graph 模式
    """
    
    # 单个 DP 秩 → 直接返回
    if parallel_config.data_parallel_size == 1:
        return False, None, cudagraph_mode
    
    # 通过 AllReduce 同步各 DP 秩的批信息
    tensor = _run_ar(
        should_ubatch=should_attempt_ubatching,
        orig_num_tokens_per_ubatch=num_tokens_unpadded,
        padded_num_tokens_per_ubatch=num_tokens_padded,
        cudagraph_mode=cudagraph_mode,
        parallel_config=parallel_config,
    )
    
    # 处理同步结果
    synced_cudagraph_mode = _post_process_cudagraph_mode(tensor)
    should_ubatch = _post_process_ubatch(tensor, parallel_config.num_ubatches)
    num_tokens_after_padding = _post_process_dp_padding(tensor, should_dp_pad)
    
    return should_ubatch, num_tokens_after_padding, synced_cudagraph_mode
```

### 3.2 全量同步操作: `_run_ar`

```python
def _run_ar(
    should_ubatch: bool,
    orig_num_tokens_per_ubatch: int,
    padded_num_tokens_per_ubatch: int,
    cudagraph_mode: int,
    parallel_config: ParallelConfig,
) -> torch.Tensor:
    """
    执行 4-element AllReduce:
      [0]: 各 DP 秩的原始 token 数量
      [1]: 各 DP 秩的填充后 token 数量  
      [2]: 是否进行微批处理 (1/0)
      [3]: 各 DP 秩的 CUDA Graph 模式
    """
    
    dp_size = parallel_config.data_parallel_size
    dp_rank = parallel_config.data_parallel_rank
    device, group = _get_device_and_group(parallel_config)
    
    # 构造本地贡献值张量
    tensor_cpu = torch.zeros(4, dp_size, dtype=torch.int32)
    tensor_cpu[0][dp_rank] = orig_num_tokens_per_ubatch
    tensor_cpu[1][dp_rank] = padded_num_tokens_per_ubatch
    tensor_cpu[2][dp_rank] = 1 if should_ubatch else 0
    tensor_cpu[3][dp_rank] = cudagraph_mode
    
    # AllReduce 聚合（得到各 DP 秩的完整信息）
    tensor = tensor_cpu.to(device, non_blocking=True)
    dist.all_reduce(tensor, group=group)
    
    return tensor
```

**通信图示**:

```
DP Rank 0         DP Rank 1         DP Rank 2
    │                 │                 │
[128, 0, 0, 1]  [64, 0, 0, 1]  [192, 0, 0, 2]
    │                 │                 │
    └─────────────────┴─────────────────┘
                      │
                 AllReduce (sum)
                      │
    ┌─────────────────┴─────────────────┐
    │                 │                 │
[128,64,192]  [128,64,192]  [128,64,192]  ← 每个 rank 都能看到所有 rank 的值
[1,1,2]       [1,1,2]       [1,1,2]
```

### 3.3 同步后处理

#### 3.3.1 CUDA Graph 模式同步 (`_post_process_cudagraph_mode`)

```python
def _post_process_cudagraph_mode(tensor: torch.Tensor) -> int:
    """
    同步所有 DP 秩的 CUDA Graph 模式
    规则：取最小值 → 如果有任何一个 rank 不能用 CUDA Graph，所有 rank 都不用
    
    模式枚举:
      0 = NONE (eager)
      1 = PIECEWISE
      2 = FULL
    """
    return int(tensor[3, :].min().item())
```

#### 3.3.2 微批处理共识 (`_post_process_ubatch`)

```python
def _post_process_ubatch(tensor: torch.Tensor, num_ubatches: int) -> bool:
    """
    确定是否进行微批处理
    规则1：所有 DP 秩都必须同意 (tensor[2,:] 全为 1)
    规则2：没有 "空" 的第二个微批 (避免浪费)
    """
    
    # 检查是否都同意微批
    should_ubatch: bool = bool(torch.all(tensor[2] == 1).item())
    if not should_ubatch:
        return False
    
    # 检查第二个微批是否为空
    orig_min_num_tokens = int(tensor[0, :].min().item())
    padded_max_num_tokens = int(tensor[1, :].max().item())
    if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens, num_ubatches):
        should_ubatch = False
    
    return should_ubatch
```

#### 3.3.3 DP 填充策略 (`_post_process_dp_padding`)

```python
def _post_process_dp_padding(tensor: torch.Tensor, should_dp_pad: bool) -> torch.Tensor:
    """
    确定各 DP 秩的最终 token 数量
    
    如果 should_dp_pad = True:
      → 所有 DP 秩都填充到最大 token 数
        (使 CUDA Graph 可以捕获固定形状)
    否则:
      → 保持各 DP 秩的原始 token 数
    """
    
    num_tokens_across_dp = tensor[1, :]
    if should_dp_pad:
        max_num_tokens = int(num_tokens_across_dp.max().item())
        return torch.tensor([max_num_tokens] * len(num_tokens_across_dp), 
                           device="cpu", dtype=torch.int32)
    else:
        return num_tokens_across_dp.cpu()
```

---

## 四、DP 元数据: DPMetadata

**文件位置**: [`vllm/forward_context.py`](file:///workspace/vllm/forward_context.py#L72-L126)

### 4.1 数据结构

```python
@dataclass
class DPMetadata:
    num_tokens_across_dp_cpu: torch.Tensor
    # 形状: [dp_size]
    # 存储每个 DP 秩的 token 数量（可能经过填充）
    
    local_sizes: list[int] | None = None
    # 仅在序列并行(SP)或微批处理(DBO)时使用
```

### 4.2 创建过程

```python
@staticmethod
def make(
    parallel_config: ParallelConfig,
    num_tokens: int,
    num_tokens_across_dp_cpu: torch.Tensor,
) -> "DPMetadata":
    """创建 DPMetadata"""
    dp_rank = parallel_config.data_parallel_rank
    
    # 验证当前 rank 的 token 数一致
    assert num_tokens_across_dp_cpu[dp_rank] == num_tokens
    
    return DPMetadata(num_tokens_across_dp_cpu)
```

### 4.3 在 Forward Context 中的集成

**文件位置**: [`forward_context.py:270-L291`](file:///workspace/vllm/forward_context.py#L270-L291)

```python
@contextmanager
def set_forward_context(...):
    # ...
    dp_metadata: DPMetadata | None = None
    if (
        vllm_config.parallel_config.data_parallel_size > 1
        and vllm_config.parallel_config.is_moe_model is not False
        and (attn_metadata is not None or num_tokens is not None)
    ):
        # 如果未初始化 num_tokens_across_dp，先同步
        if num_tokens_across_dp is None:
            _, num_tokens_across_dp, _ = coordinate_batch_across_dp(
                num_tokens_unpadded=num_tokens,
                parallel_config=vllm_config.parallel_config,
                allow_microbatching=False,
            )
        
        # 创建 DPMetadata
        dp_metadata = DPMetadata.make(
            vllm_config.parallel_config, num_tokens or 0, num_tokens_across_dp
        )
    # ...
```

---

## 五、DP 批处理与微批处理 (DBO)

### 5.1 微批处理包装器: UBatchWrapper

**文件位置**: [`vllm/v1/worker/gpu_ubatch_wrapper.py`](file:///workspace/vllm/v1/worker/gpu_ubatch_wrapper.py#L113-L527)

#### 架构

```
完整 Batch
    │
    ▼
┌───────────────────────────────────────────┐
│         UBatchWrapper                     │
│                                           │
│  ┌──────────────┐      ┌──────────────┐ │
│  │  Microbatch  │      │  Microbatch  │ │
│  │     0        │      │     1        │ │
│  └──────┬───────┘      └──────┬───────┘ │
│         │                      │         │
│         └──────────┬───────────┘         │
│                    ▼                     │
│         ┌───────────────────┐            │
│         │   Concatenate     │            │
│         │   Outputs         │            │
│         └───────────┬───────┘            │
└─────────────────────┼────────────────────┘
                      ▼
               最终输出
```

#### 微批元数据创建

```python
def _make_ubatch_metadata(...):
    # 为每个微批创建独立的 forward_context
    forward_contexts = []
    for i, ubatch_slice in enumerate(ubatch_slices):
        # 创建该微批的 dp_metadata
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        ubatch_num_tokens_across_dp = torch.tensor(
            [ubatch_slice.num_tokens] * dp_size, 
            device="cpu", dtype=torch.int32
        )
        ubatch_dp_metadata.append(
            DPMetadata.make(
                self.vllm_config.parallel_config,
                ubatch_slice.num_tokens,
                ubatch_num_tokens_across_dp,
            )
        )
        
        forward_contexts.append(
            create_forward_context(
                attn_metadata[i] if attn_metadata is not None else None,
                self.vllm_config,
                dp_metadata=ubatch_dp_metadata[i],
                # ...
            )
        )
    
    # ...
```

#### 微批输入切分

```python
def _slice_model_inputs(...):
    """
    从完整输入中切分出微批的部分
    """
    sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
    
    # 支持 mrope 位置编码（二维）
    if positions.ndim == 2:
        sliced_positions = positions[:, tokens_slice]
    else:
        sliced_positions = positions[tokens_slice]
    
    sliced_inputs_embeds = (
        inputs_embeds[tokens_slice] if inputs_embeds is not None else None
    )
    
    return sliced_input_ids, sliced_positions, sliced_inputs_embeds, ...
```

### 5.2 微批执行流程

#### 5.2.1 CUDA Graph 捕获模式

```python
def _capture_ubatches(...):
    """
    捕获微批处理的 CUDA Graph
    
    多线程架构：
      - 主线程：捕获 CUDA Graph
      - 微批线程：实际执行
    """
    
    results: list[tuple[int, torch.Tensor]] = []
    
    # 启动多个微批线程
    ubatch_threads = []
    for metadata in ubatch_metadata:
        thread = threading.Thread(
            target=_capture_ubatch_thread,
            args=(results, ubatch_metadata),
        )
        ubatch_threads.append(thread)
        thread.start()
    
    # 同步 barrier: 等待所有线程就绪
    self.ready_barrier.wait()
    
    # 主线程捕获 Graph
    with torch.cuda.graph(...):
        # 唤醒第一个微批线程
        ubatch_metadata[0].context.cpu_wait_event.set()
        
        # 等待所有线程完成
        for thread in ubatch_threads:
            thread.join()
        
        # 拼接输出
        sorted_results = [value for position, value in sorted(results)]
        result = _cat_ubatch_outputs(sorted_results)
        cudagraph_metadata.outputs = result
```

#### 5.2.2 Eager 模式执行

```python
def _run_ubatches(...):
    """
    在 eager 模式下执行微批处理
    
    和 capture 模式类似的多线程架构，但不捕获 Graph
    """
    
    results: list[tuple[int, torch.Tensor]] = []
    
    ubatch_threads = []
    for metadata in ubatch_metadata:
        thread = threading.Thread(
            target=_ubatch_thread,
            args=(results, model, metadata),
        )
        ubatch_threads.append(thread)
        thread.start()
    
    self.ready_barrier.wait()
    ubatch_metadata[0].context.cpu_wait_event.set()
    
    for thread in ubatch_threads:
        thread.join()
    
    sorted_results = [value for position, value in sorted(results)]
    return _cat_ubatch_outputs(sorted_results)
```

---

## 六、MoE 中的 DP 集成

### 6.1 DP + SP 组合使用

对于 MoE 模型，DP 经常和序列并行(SP)组合使用：

```python
@contextmanager
def sp_local_sizes(self, sequence_parallel_size: int):
    """
    上下文管理器，设置 DP+SP 的局部 size 信息
    """
    self.local_sizes = _compute_sp_num_tokens(
        self.num_tokens_across_dp_cpu, sequence_parallel_size
    )
    try:
        yield self.local_sizes
    finally:
        self.local_sizes = None

def _compute_sp_num_tokens(
    num_tokens_across_dp_cpu: torch.Tensor, sequence_parallel_size: int
) -> list[int]:
    """
    计算每个 SP rank 的 token 数量
    
    公式: sp_tokens = ceil(num_tokens / sp_size)
    然后重复 sp_size 次
    
    例子:
      num_tokens_across_dp = [128, 64, 192]
      sequence_parallel_size = 2
      
      sp_tokens = [64, 64, 32, 32, 96, 96]
    """
    sp_tokens = (num_tokens_across_dp_cpu + sequence_parallel_size - 1) // sequence_parallel_size
    sp_tokens = sp_tokens.repeat_interleave(sequence_parallel_size)
    return sp_tokens.tolist()
```

### 6.2 跨 DP+SP 的累积 Token 数

```python
def cu_tokens_across_sp(self, sp_size: int) -> torch.Tensor:
    """
    计算跨所有 DP 和 SP rank 的累积 token 数
    
    用于 MoE 专家并行中的索引计算
    """
    num_tokens_across_sp_cpu = (self.num_tokens_across_dp_cpu - 1 + sp_size) // sp_size
    num_tokens_across_sp_cpu = num_tokens_across_sp_cpu.repeat_interleave(sp_size)
    return torch.cumsum(num_tokens_across_sp_cpu, dim=0)
```

---

## 七、完整工作流示例

### 7.1 示例场景

- 配置: `dp_size = 3`, `num_ubatches = 2`
- 请求分布:
  - DP Rank 0: 128 tokens
  - DP Rank 1: 64 tokens
  - DP Rank 2: 192 tokens

### 7.2 完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│  步骤 1: 各 DP 引擎独立调度，得到本地批大小                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ DP0: num_tokens = 128
    ├─ DP1: num_tokens = 64
    └─ DP2: num_tokens = 192

┌─────────────────────────────────────────────────────────────────┐
│  步骤 2: coordinate_batch_across_dp() 同步                         │
└─────────────────────────────────────────────────────────────────┘
    │
    └─ AllReduce:
         tensor = [
           [128, 64, 192],  # orig_num_tokens
           [128, 64, 192],  # padded_num_tokens
           [  1,  1,   1],  # should_ubatch
           [  2,  2,   2]   # cudagraph_mode
         ]

    │
    ├─ synced_cudagraph_mode = min([2, 2, 2]) = 2 (FULL)
    ├─ should_ubatch = all([1, 1, 1]) = True
    └─ num_tokens_after_padding = [192, 192, 192] (因为 should_dp_pad = True)

┌─────────────────────────────────────────────────────────────────┐
│  步骤 3: 构造 DPMetadata                                          │
└─────────────────────────────────────────────────────────────────┘
    │
    └─ dp_metadata = DPMetadata(
         num_tokens_across_dp_cpu = [192, 192, 192]
       )

┌─────────────────────────────────────────────────────────────────┐
│  步骤 4: 划分微批 (如果 should_ubatch = True)                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ UBatch 0: [0:96] tokens
    │   ├─ DP0: input_ids[0:96] (pad to 96 if needed)
    │   ├─ DP1: input_ids[0:64] + pad 32
    │   └─ DP2: input_ids[0:96]
    │
    └─ UBatch 1: [96:192] tokens
        ├─ DP0: input_ids[96:128] + pad 64
        ├─ DP1: all padding (original 0-64 is done)
        └─ DP2: input_ids[96:192]

┌─────────────────────────────────────────────────────────────────┐
│  步骤 5: 执行微批处理 (CUDA Graph 或 Eager)                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ├─ Thread 0: compute UBatch 0
    ├─ Thread 1: compute UBatch 1
    └─ Concatenate outputs

┌─────────────────────────────────────────────────────────────────┐
│  步骤 6: 返回结果                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 八、关键技术决策

### 8.1 DP Padding 的设计原因

| 场景 | 是否需要 DP Padding | 原因 |
|------|-------------------|------|
| CUDA Graph = FULL | ✓ | 需要固定形状 |
| CUDA Graph = PIECEWISE | ✓ | 同上 |
| 微批处理 (DBO) | ✓ | 简化跨 rank 同步 |
| Eager 模式 | ✗ | 灵活度优先 |

### 8.2 CPU vs GPU AllReduce

**默认**: GPU (NCCL) AllReduce
**可选**: CPU AllReduce (`disable_nccl_for_dp_synchronization=True`)

**权衡**:
- GPU: 更快，但会引入同步点
- CPU: 避免 GPU 同步，但需要额外的 H2D/D2H 传输

### 8.3 微批处理取消条件

```
if is_last_ubatch_empty(orig_min_num_tokens, padded_max_num_tokens):
    should_ubatch = False
```

**设计原因**：避免空的微批导致的 GPU 利用率下降。

---

## 九、DP 与其他并行策略的交互

### 9.1 与 Tensor Parallelism (TP)

- **关系**: 正交，可组合使用
- **通信**: TP 和 DP 有独立的通信组
- **实现**: `ParallelConfig` 中分别配置

### 9.2 与 Expert Parallelism (EP)

- **关系**: DP 可与 DeepEP 组合
- **SM 控制**: `SMControlContextManager` 在微批处理时预留部分 SMs 给通信

```python
# gpu_ubatch_wrapper.py
class SMControlContextManager:
    def __enter__(self):
        self.set_comm_sms(self.comm_sms)  # 预留 SM 给 EP 通信
        self.set_compute_sms(self.compute_sms)
```

### 9.3 与 Pipeline Parallelism (PP)

- **关系**: vLLM v1 暂不支持 PP + DP 组合

---

## 十、调试与性能分析

### 10.1 关键环境变量

| 变量 | 说明 |
|------|------|
| `VLLM_LOG_BATCHSIZE_INTERVAL` | 记录 batch size 转发时间 |
| `VLLM_DISABLE_NCCL_FOR_DP_SYNCHRONIZATION` | 使用 CPU AllReduce |
| `VLLM_DBO_COMM_SMS` | 为 DP/EP 通信预留的 SM 数 |

### 10.2 监控指标

从 DPCoordinator 发布的统计信息：
- `num_waiting_reqs`: 各引擎的等待队列长度
- `num_running_reqs`: 各引擎的运行队列长度
- `current_wave`: 当前波次编号
- `engines_running`: 全局运行状态

---

## 总结

vLLM 的 DP 实现是一个完整的协同系统，包含：

1. **请求分配**: DPCoordinator 进行负载均衡与波次协调
2. **批处理协同**: `coordinate_batch_across_dp` 通过 AllReduce 同步批信息
3. **微批处理**: UBatchWrapper 支持在 DP 上进一步微批化
4. **MoE 集成**: DPMetadata 支持与 SP 的组合使用
5. **CUDA Graph**: DP 填充确保所有 DP rank 形状一致，使 Graph 可以捕获

这种设计使得 vLLM 可以在保持低延迟的同时，实现高吞吐量的多卡部署。
