# vLLM 调度器深度解析

本文档深入解析 vLLM v1 调度器的工作原理、核心设计理念和关键优化技术，从代码层面彻底理解为什么这样设计。

---

## 一、vLLM 调度系统整体架构

### 1.1 核心组件层次

vLLM v1 调度系统由以下几个关键组件组成：

```
┌─────────────────────────────────────────────────────────────────┐
│                         Scheduler Interface                     │
├─────────────────────────────────────────────────────────────────┤
│                      ┌───────────────────┐                       │
│                      │    Scheduler      │                       │
│                      │  (FIFO/Priority)  │                       │
│                      └─────────┬─────────┘                       │
├────────────────────────────────┼────────────────────────────────┤
│     ┌───────────────────┐      │      ┌───────────────────┐    │
│     │   Request Queue   │      │      │  KV Cache Manager │    │
│     │  (Waiting/Running)│      │      │  (Block Allocator)│    │
│     └───────────────────┘      │      └───────────┬───────┘    │
├────────────────────────────────┼──────────────────┼────────────┤
│                      ┌─────────▼─────────┐                    │
│                      │    Block Pool      │                    │
│                      │  (Block Hash Map) │                    │
│                      └─────────┬─────────┘                    │
├────────────────────────────────┼────────────────────────────────┤
│                      ┌─────────▼─────────┐                    │
│                      │   KV Cache Config  │                    │
│                      │  (Multiple Groups) │                    │
│                      └───────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

**核心代码位置**:
- 主调度器: [`/workspace/vllm/v1/core/sched/scheduler.py`](file:///workspace/vllm/v1/core/sched/scheduler.py#L64)
- 调度器配置: [`/workspace/vllm/config/scheduler.py`](file:///workspace/vllm/config/scheduler.py#L26)
- 请求定义: [`/workspace/vllm/v1/request.py`](file:///workspace/vllm/v1/request.py#L59)
- KV缓存管理: [`/workspace/vllm/v1/core/kv_cache_manager.py`](file:///workspace/vllm/v1/core/kv_cache_manager.py#L110)
- 块池管理: [`/workspace/vllm/v1/core/block_pool.py`](file:///workspace/vllm/v1/core/block_pool.py#L130)

---

## 二、关键创新：统一的调度范式

### 2.1 革命性的设计理念

**最核心的创新**在于消除了 Prefill 阶段和 Decode 阶段的二分法！

#### 传统 LLM 推理系统的问题

在传统系统中：
```python
# 传统范式的问题：两个独立且互斥的阶段
def traditional_schedule(requests):
    prefill_requests = [r for r in requests if r.status == "prefill"]
    decode_requests  = [r for r in requests if r.status == "decode"]
    
    # 必须做出艰难选择：是优先 prefill 还是 decode？
    # 要么造成首字延迟高，要么造成吞吐量低
    if len(prefill_requests) > 0:
        batch = prefill_requests
    else:
        batch = decode_requests
```

#### vLLM v1 的统一范式

**核心洞察**：每个请求只需要跟踪两个关键值：
- `num_computed_tokens`: 已经计算过的 token 数量
- `num_tokens_with_spec`: 需要计算的 token 总数（包括草稿）

在 [`scheduler.py:329`](file:///workspace/vllm/v1/core/sched/scheduler.py#L329) 的 `schedule()` 方法开头的注释中：

```python
# NOTE(woosuk) on the scheduling algorithm:
# There's no "decoding phase" nor "prefill phase" in the scheduler.
# Each request just has the num_computed_tokens and
# num_tokens_with_spec. num_tokens_with_spec =
# len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
# At each step, the scheduler tries to assign tokens to the requests
# so that each request's num_computed_tokens can catch up its
# num_tokens_with_spec. This is general enough to cover
# chunked prefills, prefix caching, speculative decoding,
# and the "jump decoding" optimization in the future.
```

这是一个**统一的、优雅的抽象**！不论 Prefill 还是 Decode，本质上都是让 `num_computed_tokens` 追赶 `num_tokens_with_spec`。

### 2.2 Request 数据结构详解

让我们看 [`Request`](file:///workspace/vllm/v1/request.py#L59) 类的核心状态字段：

```python
class Request:
    def __init__(self, ...):
        # ─────────────────────────────────────────────────────────
        # 核心状态：定义了请求的"计算进度"
        # ─────────────────────────────────────────────────────────
        self.num_computed_tokens = 0  # ↑↑ 已算过多少 token
        self.num_tokens = ...         # ← prompt + output (已生成)
        self.num_tokens_with_spec = ... # ← prompt + output + spec 草稿
        
        # ─────────────────────────────────────────────────────────
        # 扩展功能
        # ─────────────────────────────────────────────────────────
        self.spec_token_ids = []      # 推测解码的草稿 token
        self.num_output_placeholders = 0
        self.status = RequestStatus.WAITING
        
        # ─────────────────────────────────────────────────────────
        # Prefix Caching 支持
        # ─────────────────────────────────────────────────────────
        self.block_hashes: list[BlockHash] = []  # 每块的哈希值
```

---

## 三、调度流程的核心：schedule() 方法完整解析

[`schedule()`](file:///workspace/vllm/v1/core/sched/scheduler.py#L329) 方法是调度器的核心入口。让我们一步步解析：

### 3.1 调度约束初始化

```python
def schedule(self) -> SchedulerOutput:
    # 初始化调度结果收集
    scheduled_new_reqs: list[Request] = []
    scheduled_resumed_reqs: list[Request] = []
    scheduled_running_reqs: list[Request] = []
    preempted_reqs: list[Request] = []
    
    req_to_new_blocks: dict[str, KVCacheBlocks] = {}
    num_scheduled_tokens: dict[str, int] = {}
    
    # Token 预算 (每次 step 最多算这么多 token)
    token_budget = self.max_num_scheduled_tokens
    if self._pause_state == PauseState.PAUSED_ALL:
        token_budget = 0
    
    # KV Cache Manager 开始新的一步
    self.kv_cache_manager.new_step_starts()
```

### 3.2 第一优先级：先调度已在运行的请求

调度器首先处理 [`self.running`](file:///workspace/vllm/v1/core/sched/scheduler.py#L366) 队列中的请求：

```python
# 第一阶段：调度已在运行的请求
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    
    # 检查是否已经完成 (避免冗余调度)
    if (
        request.num_output_placeholders > 0 and
        request.num_computed_tokens + 2 - request.num_output_placeholders
        >= request.num_prompt_tokens + request.max_tokens
    ):
        req_index += 1
        continue
    
    # ─────────────────────────────────────────────────────────────
    # 计算还需要补多少 token
    # ─────────────────────────────────────────────────────────────
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )
    
    # 如果是长 Prefill，可能被截断
    if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = self.scheduler_config.long_prefill_token_threshold
    
    # 受限于剩余 Token 预算
    num_new_tokens = min(num_new_tokens, token_budget)
    
    # 同时也不能超出 max_model_len
    num_new_tokens = min(
        num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
    )
```

**设计决策思考**：为什么优先运行中的请求？
- **吞吐量优先**：正在 decode 的请求可以批处理，KV Cache 已就绪
- **保持连续性**：避免切换导致的上下文切换开销
- **延迟可预期**：正在生成的请求能尽快完成

### 3.3 KV Cache 分配与抢占逻辑

这是最复杂也最关键的部分：

```python
    # 尝试分配 KV 块
    while True:
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_lookahead_tokens=self.num_lookahead_tokens,
        )
        
        if new_blocks is not None:
            # 成功分配，可以调度
            break
        
        # ─────────────────────────────────────────────────────────────
        # 分配失败：需要抢占
        # ─────────────────────────────────────────────────────────────
        if self.policy == SchedulingPolicy.PRIORITY:
            # 优先抢占：优先级低的 + 来得晚的
            preempted_req = max(
                self.running,
                key=lambda r: (r.priority, r.arrival_time),
            )
            self.running.remove(preempted_req)
            # ... 撤销已调度的 ...
        else:
            # FIFO 策略：抢占队列尾的（后进先出）
            preempted_req = self.running.pop()
        
        # 执行抢占
        self._preempt_request(preempted_req, scheduled_timestamp)
        preempted_reqs.append(preempted_req)
        
        # 如果抢占的就是当前请求，说明没有足够空间了
        if preempted_req == request:
            break
```

**设计决策思考**：为什么这样设计抢占？
- **FIFO 策略的 LIFO 抢占**：后来的请求被抢占，先来的保留，这是公平且简单的
- **Priority 策略**：按优先级+到达时间，灵活可配置
- **关键优化**：如果发现需要抢占当前请求，直接放弃，不做死循环

### 3.4 第二优先级：调度新的等待请求

只有在没有被抢占且 token 预算还剩时，才会处理 [`waiting`](file:///workspace/vllm/v1/core/sched/scheduler.py#L548) 队列：

```python
if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
    step_skipped_waiting = create_request_queue(self.policy)
    
    while (self.waiting or self.skipped_waiting) and token_budget > 0:
        if len(self.running) == self.max_num_running_reqs:
            break  # 达到了并发数上限
        
        # 获取下一个请求
        request_queue = self._select_waiting_queue_for_scheduling()
        request = request_queue.peek_request()
        
        # ─────────────────────────────────────────────────────────────
        # Step 1: 尝试查找 Prefix Cache 命中
        # ─────────────────────────────────────────────────────────────
        if request.num_computed_tokens == 0:
            new_computed_blocks, num_new_local_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(request)
            )
            
            # 外部 KV Cache 命中 (例如远程 KV)
            if self.connector is not None:
                ext_tokens, load_kv_async = (
                    self.connector.get_num_new_matched_tokens(
                        request, num_new_local_computed_tokens
                    )
                )
                num_external_computed_tokens = ext_tokens
        
        # ─────────────────────────────────────────────────────────────
        # Step 2: 分配 KV 块
        # ─────────────────────────────────────────────────────────────
        new_blocks = self.kv_cache_manager.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_local_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            ...
        )
        
        if new_blocks is None:
            break  # 没有足够空间了
        
        # ─────────────────────────────────────────────────────────────
        # Step 3: 加入 running 队列
        # ─────────────────────────────────────────────────────────────
        self.running.append(request)
        if request.status == RequestStatus.WAITING:
            scheduled_new_reqs.append(request)
        elif request.status == RequestStatus.PREEMPTED:
            scheduled_resumed_reqs.append(request)
        
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens = num_computed_tokens
```

---

## 四、KVCacheManager 深度解析

[`KVCacheManager`](file:///workspace/vllm/v1/core/kv_cache_manager.py#L110) 是另一个核心组件，负责管理 KV Cache 的分配、释放、查询。

### 4.1 块的抽象: KVCacheBlock

核心概念是**将显存分块（Paging）**：

```python
# 每个块是显存中的连续区域
@dataclass
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    block_hash: BlockHashWithGroupId | None = None
    is_null: bool = False
```

**设计决策思考**：为什么用 Paging？
- **内存利用率高**：不用预留连续空间
- **支持前缀共享**：相同前缀可以共享块
- **灵活抢占**：以块为单位抢占，简单高效

### 4.2 块分配策略: allocate_slots()

这是一个复杂但设计精巧的方法。让我们看关键部分：

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,
    ...
) -> KVCacheBlocks | None:
    
    # ─────────────────────────────────────────────────────────────
    # 第一步：先释放不需要的块（例如滑窗外的旧块）
    # ─────────────────────────────────────────────────────────────
    self.coordinator.remove_skipped_blocks(
        request.request_id, total_computed_tokens
    )
    
    # ─────────────────────────────────────────────────────────────
    # 第二步：计算需要多少新块
    # ─────────────────────────────────────────────────────────────
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=num_tokens_need_slot,
        ...
    )
    
    # ─────────────────────────────────────────────────────────────
    # 第三步：检查是否有足够的自由块
    # ─────────────────────────────────────────────────────────────
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # 不能分配
    
    # ─────────────────────────────────────────────────────────────
    # 第四步：真正分配 + 增加引用计数
    # ─────────────────────────────────────────────────────────────
    self.coordinator.allocate_new_computed_blocks(...)
    allocated_blocks = self.coordinator.allocate_new_blocks(...)
    
    return self.create_kv_cache_blocks(allocated_blocks)
```

### 4.3 前缀缓存的实现: find_longest_cache_hit()

这是 vLLM 最强大的优化之一：

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
    # 查找与当前请求前缀匹配的已缓存块
    
    if not self.enable_caching or request.skip_reading_prefix_cache:
        return self.empty_kv_cache_blocks, 0
    
    # NOTE: 最后一个 token 需要重新计算以得到 logits
    max_cache_hit_length = request.num_tokens - 1
    
    computed_blocks, num_new_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
    
    return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

**工作原理**：
- 每个请求的 token 序列被切分成块，每块计算一个哈希值（`block_hash`）
- 通过哈希值快速查找哪些块已经被其他请求计算过了
- 增加引用计数，直接复用，避免重新计算

### 4.4 BlockPool 的设计

[`BlockPool`](file:///workspace/vllm/v1/core/block_pool.py#L130) 管理所有物理块：

```python
class BlockPool:
    def __init__(self, num_gpu_blocks: int, ...):
        # ─────────────────────────────────────────────────────────
        # 物理块数组
        # ─────────────────────────────────────────────────────────
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(idx) for idx in range(num_gpu_blocks)
        ]
        
        # ─────────────────────────────────────────────────────────
        # 自由块队列（用于 LRU 风格的管理）
        # ─────────────────────────────────────────────────────────
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        
        # ─────────────────────────────────────────────────────────
        # 块哈希表：用于前缀缓存查找
        # ─────────────────────────────────────────────────────────
        self.cached_block_hash_to_block: BlockHashToBlockMap = (
            BlockHashToBlockMap()
        )
```

**设计亮点**：
1. **FreeKVCacheBlockQueue**：双链表，O(1) 获取和释放
2. **BlockHashToBlockMap**：哈希到块的映射，支持 O(1) 前缀查找
3. **null_block**：预留 block_id=0 作为占位符

---

## 五、关键数据结构与算法

### 5.1 RequestStatus 状态机

```python
class RequestStatus(enum.IntEnum):
    # ──────────────────────────────────────────
    # 活跃状态
    # ──────────────────────────────────────────
    WAITING = enum.auto()                      # 等待调度
    WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()       # 等远程 KV
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()                      # 正在运行
    PREEMPTED = enum.auto()                    # 被抢占了
    
    # ──────────────────────────────────────────
    # 完成状态 ( > PREEMPTED )
    # ──────────────────────────────────────────
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()
    FINISHED_ERROR = enum.auto()
    FINISHED_REPETITION = enum.auto()
```

### 5.2 KVCacheBlocks: 调度器与缓存管理器之间的接口

这是一个精心设计的边界抽象：

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # 外层 tuple 是 KV Cache Group
    # 内层 Sequence 是该组的块列表
    
    def get_block_ids(...) -> tuple[list[int], ...]:
        # 转换为 block_ids 传给 Worker
```

**设计决策思考**：为什么需要这个抽象？
- **解耦**：调度器不需要知道 KVCacheManager 的内部结构
- **可扩展**：支持不同的 KV cache 分组（不同层不同配置）
- **GC 友好**：有一个全局 `empty_kv_cache_blocks` 单例避免频繁创建

---

## 六、抢占机制的完整流程

当显存不够时，`_preempt_request()` 被调用：

```python
def _preempt_request(self, request: Request, timestamp: float) -> None:
    # ─────────────────────────────────────────────────────────────
    # 1. 释放 KV 块
    # ─────────────────────────────────────────────────────────────
    self.kv_cache_manager.free(request)
    self.encoder_cache_manager.free(request)
    
    # ─────────────────────────────────────────────────────────────
    # 2. 重置状态
    # ─────────────────────────────────────────────────────────────
    request.status = RequestStatus.PREEMPTED
    request.num_computed_tokens = 0  # ← 关键：从 0 开始
    if request.spec_token_ids:
        request.spec_token_ids = []
    request.num_preemptions += 1
    
    # ─────────────────────────────────────────────────────────────
    # 3. 放回等待队列头
    # ─────────────────────────────────────────────────────────────
    self.waiting.prepend_request(request)
```

**设计决策思考**：为什么设置 `num_computed_tokens = 0`？
- **简单正确**：不需要跟踪部分完成的状态
- **依靠 Prefix Cache**：下次调度时，`get_computed_blocks()` 会找回已计算的块
- **权衡**：虽然可能浪费一点计算，但相比复杂的状态跟踪，这是更好的工程选择

---

## 七、高级特性的实现

### 7.1 分块 Prefill (Chunked Prefill)

当一个请求的 Prompt 很长，超过了 `max_num_batched_tokens` 时：

```python
# 在 schedule() 中
num_new_tokens = request.num_tokens - num_computed_tokens
threshold = self.scheduler_config.long_prefill_token_threshold
if 0 < threshold < num_new_tokens:
    num_new_tokens = threshold  # ← 只调度一部分
```

下次调度时，这个请求还在 `running` 队列，`num_computed_tokens` 记录了进度，会继续算下一部分。

### 7.2 推测解码 (Speculative Decoding)

通过两个字段实现：
```python
request.num_tokens_with_spec = ...  # 包含草稿 token
request.spec_token_ids = [...]       # 实际的草稿
```

调度器调度时会多分配一些块给草稿：
```python
num_lookahead_tokens = self.num_lookahead_tokens
new_blocks = self.kv_cache_manager.allocate_slots(
    ..., num_lookahead_tokens=num_lookahead_tokens
)
```

### 7.3 多模态支持 (Encoder Inputs)

调度器同时管理 Encoder Cache：

```python
if request.has_encoder_inputs:
    (
        encoder_inputs_to_schedule,
        num_new_tokens,
        new_encoder_compute_budget,
        external_load_encoder_input,
    ) = self._try_schedule_encoder_inputs(...)
```

### 7.4 KV Connector 支持 (远程 KV / 混合内存)

vLLM v1 支持通过 Connector 扩展 KV 层：

```python
if self.connector is not None:
    ext_tokens, load_kv_async = (
        self.connector.get_num_new_matched_tokens(...)
    )
    self.connector.update_state_after_alloc(...)
```

---

## 八、为什么这样设计？(深入理解设计决策)

### 8.1 设计哲学

1. **统一抽象**：没有 Prefill/Decode，只有 `computed` vs `to_compute`
2. **以块为中心**：PagedAttention 是一切的基础
3. **按需分配**：Token 预算 + 并发数限制双重控制
4. **优雅降级**：抢占依靠 Prefix Cache，简单高效

### 8.2 关键权衡

| 设计选择 | 优势 | 劣势 | 为什么这样选 |
|---------|------|------|-------------|
| **统一调度** | 简单、通用、支持灵活批处理 | - | 收益远大于成本 |
| **抢占后重置 num_computed_tokens** | 实现简单，避免状态泄漏 | 可能重复计算 | Prefix Cache 弥补了这一点 |
| **FIFO 默认策略** | 公平，易于理解 | 不是最优延迟/吞吐 | 可配置，且够用 |
| **Block 粒度管理** | 灵活，内存利用率高 | 碎片化 | 可接受，且有滑窗回收缓解 |

---

## 九、vLLM v1 调度器的完整工作流

```
                    ┌───────────────┐
                    │ 新请求到达     │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │  WAITING 队列 │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
    ┌─────────▼─────────┐       ┌───────▼───────┐
    │  find_longest_    │       │  还有空间吗？ │
    │  cache_hit()?     │       └───┬───────┬───┘
    └─────────┬─────────┘           │       │
              │                     │ Yes   │ No
              │ Hit                 │       │
    ┌─────────▼─────────┐   ┌───────▼───────┐
    │  复用已有块       │   │   抢占一个    │
    └─────────┬─────────┘   │   运行请求    │
              │             └───────┬───────┘
    ┌─────────▼─────────┐           │
    │ allocate_slots()  │◄──────────┘
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │  加入 RUNNING     │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │  schedule() 循环  │◄──────────────────┐
    │  每次调度一点     │                   │
    └─────────┬─────────┘                   │
              │                             │
    ┌─────────▼─────────┐           ┌─────┴─────┐
    │  update_from_     │           │  继续生成  │
    │  outputs()        │           └───────────┘
    └─────────┬─────────┘
              │
              │ 完成？
    ┌─────────▼─────────┐
    │  Free KV Blocks   │
    └───────────────────┘
```

---

## 十、总结与关键亮点

vLLM v1 调度器的核心创新可总结为：

1. **范式革命**：消除 Prefill/Decode 二分法，统一为追赶问题
2. **以块为中心**：PagedAttention + Block Pool 是底层支柱
3. **前缀缓存**：通过哈希和引用计数实现零成本复用
4. **优雅抢占**：简单的 LIFO 策略 + 前缀缓存恢复
5. **高度可扩展**：支持 SpecDecoding、Multimodal、KV Offloading 等

**最后一个洞察**：vLLM v1 调度器之所以强大且简洁，在于它找到了一个恰当的抽象层次——不是 token，不是 sequence，而是 token 的**计算进度** + 显存的**块管理**。这是一个非常漂亮的设计！
