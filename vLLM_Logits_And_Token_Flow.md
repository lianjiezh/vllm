# vLLM Logits 计算与 Token 生成完整流程

本文档深入解析 vLLM 中 logits 如何从模型输出计算，经过采样最终生成 token，并返回到下一次 step 推理的完整流程。

---

## 一、整体数据流概览

```
┌─────────────────────────────────────────────────────────────────┐
│                      Step N: Model Forward Pass                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐                                              │
│   │ Hidden States │                                              │
│   │   [B, H]     │  ← 模型前向传播输出                          │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │ compute_logits│  ← 将 hidden states 转换为 logits           │
│   │   [B, V]     │     V = vocab_size                          │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │   Sampler    │  ← 采样层：应用各种约束和采样策略             │
│   └──────┬───────┘                                              │
│          │                                                      │
│          │  ┌──────────────────────────────────────┐           │
│          ├──│ 1. Logits Processors (重复惩罚等)      │           │
│          ├──│ 2. Temperature Scaling                 │           │
│          ├──│ 3. Top-K / Top-P Filtering            │           │
│          └──│ 4. 采样生成 Token                     │           │
│             └──────────────────────────────────────┘           │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │ SamplerOutput│  ← 包含 sampled_token_ids + logprobs        │
│   └──────┬───────┘                                              │
│          │                                                      │
├──────────┼──────────────────────────────────────────────────────┤
│          │         Step N+1: Token 反馈循环                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │ post_update() │  ← 更新请求状态：                             │
│   │              │     - 更新 all_token_ids                      │
│   │              │     - 更新 last_sampled_tokens                 │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │ combine_     │  ← 准备下一个 step 的输入:                     │
│   │ sampled_     │     - 获取 last_sampled_tokens                 │
│   │ and_draft_  │     - 作为 input_ids 喂回模型                  │
│   │ tokens()     │                                              │
│   └──────┬───────┘                                              │
│          │                                                      │
│          ▼                                                      │
│   ┌──────────────┐                                              │
│   │    模型      │──────────► 回到 Step N+1 的前向传播            │
│   │  Forward    │                                              │
│   └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、Logits 的计算过程

### 2.1 从 Hidden States 到 Logits

**代码位置**: [`model_runner.py:943`](file:///workspace/vllm/v1/worker/gpu/model_runner.py#L936-L943)

```python
def sample(
    self,
    hidden_states: torch.Tensor,
    input_batch: InputBatch,
    grammar_output: GrammarOutput | None,
) -> tuple[SamplerOutput, torch.Tensor, torch.Tensor]:
    # 1. 提取用于采样的 hidden states
    sample_hidden_states = hidden_states[input_batch.logits_indices]
    
    # 2. 通过 LM Head 计算 logits
    logits = self.model.compute_logits(sample_hidden_states)
    
    # 3. 如果有 grammar constraints，应用它们
    if grammar_output is not None:
        assert self.structured_outputs_worker is not None
        self.structured_outputs_worker.apply_grammar_bitmask(...)
    
    # 4. 传递给 Sampler 进行采样
    sampler_output = self.sampler(logits, input_batch)
    
    return sampler_output, num_sampled, num_rejected
```

**关键点**:
- `hidden_states`: 形状 `[num_tokens, hidden_size]`，包含所有 token 的隐藏状态
- `logits_indices`: 标记哪些 token 需要采样（通常是每个请求的最后一个 token）
- `logits`: 形状 `[num_requests, vocab_size]`，每个请求一个概率分布

### 2.2 Logits 的含义

```
Logits 本质上是未归一化的分数（logits 原文 = log-odds）

┌─────────────────────────────────────────────────────────────────┐
│                     Logits Vector [V]                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Token ID:     0      1      2      3     ...    V-1        │
│   Logits:     -3.2    8.7   -1.4    2.1   ...   -0.9         │
│                                                                  │
│   数值越大 → 该 token 被采样的概率越高                              │
│   （但还不是概率，需要 softmax 归一化）                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、Sampler 的完整采样流程

### 3.1 Sampler 的 9 步处理流程

**代码位置**: [`sampler.py:21-60`](file:///workspace/vllm/v1/sample/sampler.py#L21-L60)

```
┌─────────────────────────────────────────────────────────────────┐
│                     Sampler.forward() 执行流程                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1-2: 处理 Logprobs 请求 & 类型转换                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ if num_logprobs:                                          │   │
│  │   if mode == "raw_logprobs":                             │   │
│  │     raw_logprobs = log_softmax(logits)  ← 原文计算        │   │
│  │   else:  # raw_logits                                     │   │
│  │     raw_logprobs = logits.clone()  ← 直接克隆             │   │
│  │ logits = logits.to(torch.float32)  ← 转为 float32         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Step 3-4: 应用 Token 约束                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. 允许的 token IDs 白名单 (allowed_token_ids_mask)      │   │
│  │ 2. 禁止的 token IDs 黑名单 (bad_words)                    │   │
│  │    → 将这些 logits 设为 -∞                                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Step 5-6: 应用 Logits Processors                               │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ 1. Min Tokens Processor (最小 token 数约束)               │   │
│  │ 2. Logit Bias Processor (logit 偏置)                     │   │
│  │ 3. Repetition Penalty (重复惩罚)                          │   │
│  │ 4. Frequency Penalty (频率惩罚)                            │   │
│  │ 5. Presence Penalty (存在惩罚)                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Step 7: 采样 (核心)                                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ a) 如果 all_greedy:                                       │   │
│  │      sampled = argmax(logits)  ← 贪婪采样                 │   │
│  │                                                            │   │
│  │ b) 否则 (随机采样):                                        │   │
│  │      logits = logits / temperature  ← 温度缩放            │   │
│  │      apply_min_p(logits)  ← Min-P 采样                   │   │
│  │      apply_top_k(logits)   ← Top-K 过滤                   │   │
│  │      apply_top_p(logits)   ← Top-P 过滤                   │   │
│  │      sampled = multinomial(probs)  ← 多项式采样            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  Step 8-9: 收集 Top-K Logprobs & 返回                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ if num_logprobs:                                         │   │
│  │   topk_indices, topk_logprobs = topk(logprobs, k)        │   │
│  │   sampled_logprob = logprobs[sampled]                    │   │
│  │   token_rank = count(tokens > sampled_token)              │   │
│  │                                                            │   │
│  │ return SamplerOutput(                                     │   │
│  │   sampled_token_ids=[batch, 1],                          │   │
│  │   logprobs_tensors={indices, logprobs, ranks}            │   │
│  │ )                                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 重复惩罚的数学原理

**代码位置**: [`sampler.py:411-428`](file:///workspace/vllm/v1/sample/sampler.py#L411-L428)

```python
@staticmethod
def apply_penalties(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
    output_token_ids: list[list[int]],
) -> torch.Tensor:
    if sampling_metadata.no_penalties:
        return logits

    # 调用 Triton kernel 应用所有惩罚
    return apply_all_penalties(
        logits,
        sampling_metadata.prompt_token_ids,
        sampling_metadata.presence_penalties,  # 存在惩罚
        sampling_metadata.frequency_penalties, # 频率惩罚
        sampling_metadata.repetition_penalties, # 重复惩罚
        output_token_ids,
    )
```

**惩罚公式**:
```python
# Repetition Penalty (重复惩罚)
if token in generated_tokens:
    if logits[token] > 0:
        logits[token] /= repetition_penalty
    else:
        logits[token] *= repetition_penalty

# Frequency Penalty (频率惩罚)
if token in generated_tokens:
    logits[token] -= frequency_penalty * count(token)

# Presence Penalty (存在惩罚)
if token in generated_tokens:
    logits[token] -= presence_penalty
```

---

## 四、Token 如何返回到下一次 Step

### 4.1 核心机制：last_sampled_tokens

**代码位置**: [`states.py:64-66`](file:///workspace/vllm/v1/worker/gpu/states.py#L64-L66)

```python
class RequestState:
    def __init__(self, ...):
        # 每个请求维护一个 "上一次采样的 token"
        self.last_sampled_tokens = torch.zeros(
            self.max_num_reqs, 1, dtype=torch.int64, device=device
        )
```

这个 tensor 是整个循环的关键！它保存了每个请求最后一次采样的 token。

### 4.2 采样后的状态更新：post_update

**代码位置**: [`input_batch.py:477-505`](file:///workspace/vllm/v1/worker/gpu/input_batch.py#L477-L505)

```python
def post_update(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor | None,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> None:
    """更新请求状态，包括：
    1. 将采样的 token 写入 all_token_ids
    2. 更新 last_sampled_tokens
    3. 更新 num_computed_tokens
    4. 更新 total_len
    """
    num_reqs = idx_mapping.shape[0]
    _post_update_kernel[(num_reqs,)](
        idx_mapping,
        num_computed_tokens,
        last_sampled_tokens,
        output_bin_counts,
        sampled_tokens,
        num_sampled,
        num_rejected,
        query_start_loc,
        all_token_ids,
        total_len,
    )
```

### 4.3 Triton Kernel 的实际更新逻辑

**代码位置**: [`input_batch.py:424-475`](file:///workspace/vllm/v1/worker/gpu/input_batch.py#L424-L475)

```python
@triton.jit
def _post_update_kernel(
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    last_sampled_tokens_ptr,
    output_bin_counts_ptr,
    sampled_tokens_ptr,
    num_sampled_ptr,
    num_rejected_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    total_len_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    
    # ─────────────────────────────────────────────────────────────
    # 1. 获取采样信息
    # ─────────────────────────────────────────────────────────────
    num_sampled = tl.load(num_sampled_ptr + batch_idx)
    num_rejected = tl.load(num_rejected_ptr + batch_idx)
    
    # ─────────────────────────────────────────────────────────────
    # 2. 更新 last_sampled_tokens
    #    关键！这是下一次 step 的输入
    # ─────────────────────────────────────────────────────────────
    if num_sampled > 0:
        # 获取采样的 token
        sampled_token = tl.load(sampled_tokens_ptr + batch_idx)
        # 写入 last_sampled_tokens
        tl.store(last_sampled_tokens_ptr + req_state_idx, sampled_token)
    
    # ─────────────────────────────────────────────────────────────
    # 3. 将采样 token 添加到 all_token_ids
    # ─────────────────────────────────────────────────────────────
    total_len = tl.load(total_len_ptr + req_state_idx)
    all_token_ids_ptr_req = all_token_ids_ptr + req_state_idx
    
    # 写入新采样的 token
    sampled_token = tl.load(sampled_tokens_ptr + batch_idx)
    tl.store(all_token_ids_ptr_req + total_len, sampled_token)
    
    # ─────────────────────────────────────────────────────────────
    # 4. 更新 total_len
    # ─────────────────────────────────────────────────────────────
    new_total_len = total_len + num_sampled - num_rejected
    tl.store(total_len_ptr + req_state_idx, new_total_len)
```

---

## 五、下一次 Step 如何使用 last_sampled_tokens

### 5.1 combine_sampled_and_draft_tokens

**代码位置**: [`input_batch.py:279-360`](file:///workspace/vllm/v1/worker/gpu/input_batch.py#L279-L360)

这是准备下一次前向传播输入的核心函数：

```python
@triton.jit
def _combine_sampled_and_draft_tokens_kernel(
    input_ids_ptr,
    idx_mapping_ptr,
    last_sampled_tokens_ptr,  # ← 关键输入！
    query_start_loc_ptr,
    seq_lens_ptr,
    prefill_len_ptr,
    draft_tokens_ptr,
    ...
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    
    # ─────────────────────────────────────────────────────────────
    # 核心逻辑：使用 last_sampled_tokens 作为 decode step 的输入
    # ─────────────────────────────────────────────────────────────
    last_sampled = tl.load(last_sampled_tokens_ptr + req_state_idx)
    
    # 获取当前请求的 token 数量
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    num_tokens = query_end - query_start
    
    # ─────────────────────────────────────────────────────────────
    # 将 last_sampled_tokens 写入 input_ids
    # 这是下一个 token 的输入！
    # ─────────────────────────────────────────────────────────────
    if num_tokens == 1:
        # Decode case: 直接使用 last_sampled
        tl.store(input_ids_ptr + query_start, last_sampled)
    else:
        # Prefill + Decode case: 组合使用
        # ... (处理混合 prefills)
```

### 5.2 prepare_prefill_inputs: Prefill 阶段的 token 准备

**代码位置**: [`input_batch.py:161-218`](file:///workspace/vllm/v1/worker/gpu/input_batch.py#L161-L218)

```python
@triton.jit
def _prepare_prefill_inputs_kernel(
    input_ids_ptr,
    next_prefill_tokens_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    all_token_ids_ptr,
    ...
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)
    
    prefill_len = tl.load(prefill_lens_ptr + req_state_idx)
    num_computed = tl.load(num_computed_tokens_ptr + req_state_idx)
    
    # ─────────────────────────────────────────────────────────────
    # 从 all_token_ids 中提取需要计算的 token
    # 从 num_computed 开始，读取 (prefill_len - num_computed) 个 token
    # ─────────────────────────────────────────────────────────────
    request_ptr = all_token_ids_ptr + req_state_idx * all_token_ids_stride
    for i in range(0, query_len, BLOCK_SIZE):
        tokens = tl.load(request_ptr + num_computed + i)
        tl.store(input_ids_ptr + query_start + i, tokens)
    
    # 如果还有剩余，计算下一个 prefill token
    next_pos = num_computed + query_len
    if next_pos < prefill_len:
        next_token = tl.load(request_ptr + next_pos)
        tl.store(next_prefill_tokens_ptr + req_state_idx, next_token)
```

---

## 六、完整循环图解

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STEP N                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                         模型 Forward                               │    │
│  │                                                                    │    │
│  │   Prompt: "Hello" + all_token_ids[:-1]                           │    │
│  │                          │                                        │    │
│  │                          ▼                                        │    │
│  │   Hidden States [N, hidden]                                       │    │
│  │                          │                                        │    │
│  │                          ▼                                        │    │
│  │   compute_logits()                                                │    │
│  │                          │                                        │    │
│  │                          ▼                                        │    │
│  │   Logits [batch, vocab_size]                                      │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                          │                                             │
│                          ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                      Sampler                                      │    │
│  │                                                                    │    │
│  │   Logits Processing (Penalties, Temperature, Top-K/P)            │    │
│  │                          │                                        │    │
│  │                          ▼                                        │    │
│  │   Sampling → "world" (假设采样得到 "world")                     │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                          │                                             │
│                          ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                     post_update()                                 │    │
│  │                                                                    │    │
│  │   all_token_ids[4] = "world"  ← 添加到 token 序列               │    │
│  │   last_sampled_tokens = "world"  ← 保存为下一次输入             │    │
│  │   total_len += 1                                                │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
├─────────────────────────────────────────────────────────────────────────┤
│                          STEP N + 1                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │              prepare_inputs()                                    │    │
│  │                                                                    │    │
│  │   combine_sampled_and_draft_tokens()                            │    │
│  │                          │                                        │    │
│  │                          ▼                                        │    │
│  │   input_ids[0] = last_sampled_tokens  ← "world"                 │    │
│  │                          │                                        │    │
│  │                          ▼                                        │    │
│  │   positions[0] = 4  ← 位置编码                                  │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                          │                                             │
│                          ▼                                             │
│  ┌────────────────────────────────────────────────────────────────┐    │
│  │                         模型 Forward                               │    │
│  │                                                                    │    │
│  │   "world" → 计算 → Hidden State                                  │    │
│  │                          │                                        │    │
│  │                          ▼                                        │    │
│  │   Logits → Sampler → 下一个 Token                                │    │
│  │                                                                    │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│                          ... (重复直到结束)                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 七、关键数据结构的生命周期

### 7.1 all_token_ids

```python
# RequestState 中维护的完整 token 序列
self.all_token_ids = torch.zeros(
    (max_num_reqs, max_model_len),  # 预分配大矩阵
    dtype=torch.int32,
    device=device,
)

# 生命周期：
# 1. 新请求: all_token_ids[req_idx] = prompt_token_ids
# 2. 每个 step: all_token_ids[req_idx][total_len] = sampled_token
# 3. 请求结束: 等待复用或清理
```

### 7.2 last_sampled_tokens

```python
# 每个请求一个值，存储上一次采样的 token
self.last_sampled_tokens = torch.zeros(
    max_num_reqs, 1,
    dtype=torch.int64,
    device=device,
)

# 生命周期：
# 1. 新请求: 初始化为 0 或 prompt 的最后一个 token
# 2. 每个 step: last_sampled_tokens[req_idx] = sampled_token
# 3. 下一次 step: 作为 decode 的 input_ids 使用
```

### 7.3 num_computed_tokens

```python
# 跟踪每个请求已经计算了多少 token
self.num_computed_tokens = torch.zeros(
    max_num_reqs,
    dtype=torch.int32,
    device=device,
)

# 生命周期：
# 1. Prefill 开始: num_computed_tokens = 0
# 2. Prefill 中: num_computed_tokens += chunk_size
# 3. Prefill 结束: num_computed_tokens = prompt_len
# 4. Decode 中: 保持不变（每个 step 只生成 1 个 token）
```

---

## 八、推测解码（Speculative Decoding）的特殊处理

### 8.1 draft_tokens 的使用

**代码位置**: [`states.py:68-74`](file:///workspace/vllm/v1/worker/gpu/states.py#L68-L74)

```python
# 维护草稿 token（用于推测解码）
self.draft_tokens = torch.zeros(
    max_num_reqs,
    num_speculative_steps,  # 最多 N 个草稿 token
    dtype=torch.int64,
    device=device,
)
```

### 8.2 combine_sampled_and_draft_tokens 的完整逻辑

```python
# 对于推测解码，一个 batch 中可能有多个 logit 输出
# 格式: [req1_draft1, req1, req1_draft2, req2, req2_draft1, ...]

# decode tokens 来自 last_sampled_tokens
# draft tokens 来自 draft_tokens
# 最终合并为完整的 input_ids
```

---

## 九、总结

vLLM 中 logits 计算和 token 返回的核心流程可以总结为：

1. **Logits 计算**: 模型前向传播 → hidden states → LM Head → logits
2. **采样**: Logits processors → Temperature → Top-K/P → Multinomial → Token
3. **状态更新**: post_update() → 更新 all_token_ids + last_sampled_tokens
4. **下一轮输入**: combine_sampled_and_draft_tokens() → 使用 last_sampled_tokens 作为 input_ids

**关键设计**:
- `last_sampled_tokens` 是连接两个 step 的桥梁
- 所有状态都是 GPU-resident，避免 CPU-GPU 传输开销
- Triton kernels 实现高效的批量更新操作
- 支持推测解码、多模态、分块 Prefill 等高级特性
