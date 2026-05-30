# vLLM Benchmark 测试实现详解

## 目录
1. [Benchmark 架构概览](#benchmark-架构概览)
2. [CLI 接口设计](#cli-接口设计)
3. [核心测试模块](#核心测试模块)
4. [数据集模块](#数据集模块)
5. [服务端基准测试](#服务端基准测试)
6. [性能指标计算](#性能指标计算)

---

## Benchmark 架构概览

vLLM 的 benchmark 系统采用模块化设计，主要包含以下几个关键组件：

```
vllm/
├── entrypoints/cli/benchmark/        # CLI 入口
│   ├── base.py                      # 基准命令基类
│   ├── throughput.py                # 吞吐率测试
│   ├── latency.py                   # 延迟测试
│   ├── serve.py                     # 服务端测试
│   ├── startup.py                   # 启动时间测试
│   └── sweep/                       # 参数扫描
├── vllm/benchmarks/                 # 实际实现
│   ├── throughput.py                # 吞吐率核心实现
│   ├── latency.py                   # 延迟核心实现
│   ├── serve.py                     # 服务端核心实现
│   ├── datasets/                    # 数据集管理
│   └── lib/                         # 工具库
└── benchmarks/                       # 旧版（已废弃，重定向到CLI）
```

---

## CLI 接口设计

### 命令基类 - `BenchmarkSubcommandBase`

**位置：** [vllm/entrypoints/cli/benchmark/base.py](file:///workspace/vllm/entrypoints/cli/benchmark/base.py)

```python
class BenchmarkSubcommandBase(CLISubcommand):
    """基准测试子命令基类"""
    
    name: str           # 命令名 (如 "throughput", "latency")
    help: str           # 帮助信息
    
    @classmethod
    def add_cli_args(cls, parser):
        """添加命令行参数"""
        raise NotImplementedError
        
    @staticmethod
    def cmd(args):
        """运行基准测试"""
        raise NotImplementedError
```

### 具体命令实现

#### 1. Throughput 命令 - `BenchmarkThroughputSubcommand`

**位置：** [vllm/entrypoints/cli/benchmark/throughput.py](file:///workspace/vllm/entrypoints/cli/benchmark/throughput.py)

```python
class BenchmarkThroughputSubcommand(BenchmarkSubcommandBase):
    name = "throughput"
    help = "Benchmark offline inference throughput."
    
    @classmethod
    def add_cli_args(cls, parser):
        from vllm.benchmarks.throughput import add_cli_args
        add_cli_args(parser)
        
    @staticmethod
    def cmd(args):
        from vllm.benchmarks.throughput import main
        main(args)
```

#### 2. Latency 命令 - `BenchmarkLatencySubcommand`

**位置：** [vllm/entrypoints/cli/benchmark/latency.py](file:///workspace/vllm/entrypoints/cli/benchmark/latency.py)

```python
class BenchmarkLatencySubcommand(BenchmarkSubcommandBase):
    name = "latency"
    help = "Benchmark the latency of a single batch of requests."
    
    # 类似的结构...
```

### 使用方式

```bash
# 吞吐率测试
vllm bench throughput --model <model-path> --num-prompts 100

# 延迟测试
vllm bench latency --model <model-path> --input-len 32 --output-len 128

# 服务端测试
vllm bench serve --model <model-path>
```

---

## 核心测试模块

### 1. Throughput 基准测试

**位置：** [vllm/benchmarks/throughput.py](file:///workspace/vllm/benchmarks/throughput.py)

#### 核心函数：`run_vllm`

```python
def run_vllm(
    requests: list[SampleRequest],
    n: int,
    engine_args: EngineArgs,
    do_profile: bool,
    disable_detokenize: bool = False,
) -> tuple[float, list[RequestOutput] | None]:
    """
    运行 vLLM 离线推理吞吐率基准测试
    
    参数：
        requests: 测试请求列表
        n: 每个请求生成的序列数
        engine_args: vLLM 引擎参数
        do_profile: 是否进行性能分析
        disable_detokenize: 是否禁用 detokenization
    """
    # 1. 初始化 LLM 引擎
    llm = LLM.from_engine_args(engine_args)
    
    # 2. 验证请求长度
    assert all(
        llm.llm_engine.model_config.max_model_len
        >= (request.prompt_len + request.expected_output_len)
        for request in requests
    )
    
    # 3. 准备 prompts 和采样参数
    prompts = []
    sampling_params = []
    for request in requests:
        prompt = (
            TokensPrompt(prompt_token_ids=request.prompt["prompt_token_ids"])
            if "prompt_token_ids" in request.prompt
            else TextPrompt(prompt=request.prompt)
        )
        if request.multi_modal_data:
            prompt["multi_modal_data"] = request.multi_modal_data
        prompts.append(prompt)
        
        sampling_params.append(
            SamplingParams(
                n=n,
                temperature=1.0,
                top_p=1.0,
                ignore_eos=True,
                max_tokens=request.expected_output_len,
                detokenize=not disable_detokenize,
            )
        )
    
    # 4. 生成并计时
    start = time.perf_counter()
    if do_profile:
        llm.start_profile()
    outputs = llm.generate(
        prompts, sampling_params, lora_request=lora_requests, use_tqdm=True
    )
    if do_profile:
        llm.stop_profile()
    end = time.perf_counter()
    
    return end - start, outputs
```

#### 异步版本：`run_vllm_async`

```python
async def run_vllm_async(
    requests: list[SampleRequest],
    n: int,
    engine_args: AsyncEngineArgs,
    do_profile: bool,
    disable_detokenize: bool = False,
) -> float:
    """异步版本的基准测试"""
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client_from_engine_args,
    )
    
    async with build_async_engine_client_from_engine_args(
        engine_args,
    ) as llm:
        # 准备请求
        generators = []
        start = time.perf_counter()
        if do_profile:
            await llm.start_profile()
        
        # 异步生成多个请求
        for i, (prompt, sp, lr) in enumerate(
            zip(prompts, sampling_params, lora_requests)
        ):
            generator = llm.generate(
                prompt, sp, lora_request=lr, request_id=f"test{i}"
            )
            generators.append(generator)
        
        # 合并异步迭代器
        all_gens = merge_async_iterators(*generators)
        async for i, res in all_gens:
            pass
        
        if do_profile:
            await llm.stop_profile()
        end = time.perf_counter()
        return end - start
```

#### HF 对比版本：`run_hf`

```python
def run_hf(
    requests: list[SampleRequest],
    model: str,
    tokenizer: TokenizerLike,
    n: int,
    max_batch_size: int,
    trust_remote_code: bool,
    disable_detokenize: bool = False,
    dtype: torch.dtype | None = torch.float16,
    enable_torch_compile: bool = False,
) -> float:
    """运行 Hugging Face 模型的基准测试"""
    llm = AutoModelForCausalLM.from_pretrained(
        model, dtype=dtype, trust_remote_code=trust_remote_code
    )
    
    # 批量处理
    batch = []
    max_prompt_len = 0
    max_output_len = 0
    
    for i in range(len(requests)):
        batch.append(prompt)
        if len(batch) < max_batch_size and i != len(requests) - 1:
            continue
        
        # 生成序列
        input_ids = tokenizer(batch, return_tensors="pt", padding=True).input_ids
        llm_outputs = llm.generate(
            input_ids=input_ids.to(current_platform.device_type),
            do_sample=True,
            num_return_sequences=n,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            max_new_tokens=max_output_len,
        )
        
        batch = []
    
    return end - start
```

### 2. Latency 基准测试

**位置：** [vllm/benchmarks/latency.py](file:///workspace/vllm/benchmarks/latency.py)

```python
def main(args: argparse.Namespace):
    """延迟测试主函数"""
    engine_args = EngineArgs.from_cli_args(args)
    llm = LLM.from_engine_args(engine_args)
    
    # 准备采样参数
    sampling_params = SamplingParams(
        n=args.n,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
        detokenize=not args.disable_detokenize,
    )
    
    # 生成 dummy prompt token IDs
    dummy_prompt_token_ids = np.random.randint(
        10000, size=(args.batch_size, args.input_len)
    )
    dummy_prompts: list[PromptType] = [
        {"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()
    ]
    
    def llm_generate():
        if not args.use_beam_search:
            llm.generate(dummy_prompts, sampling_params=sampling_params, use_tqdm=False)
        else:
            llm.beam_search(
                dummy_prompts,
                BeamSearchParams(
                    beam_width=args.n,
                    max_tokens=args.output_len,
                    ignore_eos=True,
                ),
            )
    
    def run_to_completion(do_profile: bool = False):
        if do_profile:
            llm.start_profile()
            llm_generate()
            llm.stop_profile()
        else:
            start_time = time.perf_counter()
            llm_generate()
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency
    
    # Warmup
    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion(do_profile=False)
    
    if args.profile:
        run_to_completion(do_profile=True)
        return
    
    # 基准测试
    latencies = []
    for _ in tqdm(range(args.num_iters), desc="Bench iterations"):
        latencies.append(run_to_completion(do_profile=False))
    latencies = np.array(latencies)
    
    # 计算百分位数
    percentages = [10, 25, 50, 75, 90, 99]
    percentiles = np.percentile(latencies, percentages)
    print(f"Avg latency: {np.mean(latencies)} seconds")
    for percentage, percentile in zip(percentages, percentiles):
        print(f"{percentage}% percentile latency: {percentile} seconds")
```

---

## 数据集模块

**位置：** [vllm/benchmarks/datasets/datasets.py](file:///workspace/vllm/benchmarks/datasets/datasets.py)

### 数据集类型

vLLM 提供多种预定义数据集：

1. **RandomDataset** - 随机生成的数据集
2. **RandomMultiModalDataset** - 随机多模态数据集
3. **ShareGPTDataset** - ShareGPT 对话数据集
4. **SonnetDataset** - 基于 Sonnet 的数据集
5. **BurstGPTDataset** - BurstGPT 数据集
6. **ConversationDataset** - 通用对话数据集
7. **InstructCoderDataset** - 代码指令数据集

### 数据集基类

```python
from dataclasses import dataclass

@dataclass
class SampleRequest:
    """单个样本请求的数据类"""
    prompt: str | dict
    prompt_len: int
    expected_output_len: int
    multi_modal_data: dict | list[dict] | None = None
    lora_request: LoRARequest | None = None
```

### 请求获取函数

```python
def get_requests(args, tokenizer):
    """获取测试请求"""
    common_kwargs = {
        "dataset_path": args.dataset_path,
        "random_seed": args.seed,
    }
    sample_kwargs = {
        "tokenizer": tokenizer,
        "lora_path": args.lora_path,
        "max_loras": args.max_loras,
        "lora_assignment": getattr(args, "lora_assignment", "random"),
        "num_requests": args.num_prompts,
    }
    
    if args.dataset_name == "random":
        dataset_cls = RandomDataset
    elif args.dataset_name == "sharegpt":
        dataset_cls = ShareGPTDataset
    elif args.dataset_name == "hf":
        dataset_cls = HuggingFaceDataset
    # ... 更多数据集
    
    dataset = dataset_cls(**common_kwargs)
    requests = dataset.sample(**sample_kwargs)
    return requests
```

---

## 服务端基准测试

**位置：** [vllm/benchmarks/serve.py](file:///workspace/vllm/benchmarks/serve.py)

### 异步请求处理器

vLLM 支持多种后端的异步请求：

```python
# backend_request_func.py
@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: str | None = None
    logprobs: int | None = None
    extra_body: dict | None = None
    multi_modal_content: dict | list[dict] | None = None
    ignore_eos: bool = False
    language: str | None = None
    request_id: str | None = None

@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # Inter-token latency
    tpot: float = 0.0  # Token per output time
    prompt_len: int = 0
    error: str = ""

ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "openai-audio": async_request_openai_audio,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
    "llama.cpp": async_request_openai_completions,
}
```

### OpenAI 兼容后端示例

```python
async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: tqdm | None = None,
) -> RequestFuncOutput:
    """OpenAI 兼容后端的异步请求处理"""
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile"))
    
    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model_name if request_func_input.model_name 
                     else request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": request_func_input.output_len,
            "logprobs": request_func_input.logprobs,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id
        
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        
        generated_text = ""
        ttft = 0.0
        start = time.perf_counter()
        most_recent_timestamp = start
        
        async with session.post(
            url=api_url, json=payload, headers=headers
        ) as response:
            if response.status == 200:
                first_chunk_received = False
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    
                    chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                    if chunk != "[DONE]":
                        data = json.loads(chunk)
                        
                        if choices := data.get("choices"):
                            text = choices[0].get("text")
                            timestamp = time.perf_counter()
                            
                            # 记录首个 token 时间
                            if not first_chunk_received:
                                first_chunk_received = True
                                ttft = time.perf_counter() - start
                                output.ttft = ttft
                            else:
                                output.itl.append(timestamp - most_recent_timestamp)
                            
                            most_recent_timestamp = timestamp
                            generated_text += text or ""
                        
                        if usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens")
                
                output.generated_text = generated_text
                output.success = True
                output.latency = most_recent_timestamp - start
            else:
                output.error = response.reason or ""
                output.success = False
    
    if pbar:
        pbar.update(1)
    return output
```

---

## 性能指标计算

### 吞吐率指标

```python
# 在 throughput.py 的 main 函数中
elapsed_time = end - start
total_num_tokens = sum(
    len(output.outputs[0].token_ids) for output in outputs
)

results = {
    "elapsed_time": elapsed_time,
    "num_requests": len(requests),
    "total_num_tokens": total_num_tokens,
    "requests_per_second": len(requests) / elapsed_time,
    "tokens_per_second": total_num_tokens / elapsed_time,
}

print(f"Throughput: {results['tokens_per_second']:.2f} tokens/s")
print(f"Throughput: {results['requests_per_second']:.2f} requests/s")
```

### 延迟指标

```python
# 在 latency.py 中
percentages = [10, 25, 50, 75, 90, 99]
percentiles = np.percentile(latencies, percentages)
results = {
    "avg_latency": np.mean(latencies),
    "latencies": latencies.tolist(),
    "percentiles": dict(zip(percentages, percentiles.tolist())),
}
```

### TTFT 和 ITL 指标

```python
# 在服务端测试中收集
ttft = time_to_first_token
itl = inter_token_latency_list
tpot = avg_time_per_output_token

# 用于绘制性能曲线和分析抖动
```

---

## 时间收集工具

**位置：** [benchmarks/benchmark_utils.py](file:///workspace/benchmarks/benchmark_utils.py)

```python
class TimeCollector:
    """用于收集和分析时间数据的工具类"""
    
    NS: int = 1
    US: int = NS * 1000
    MS: int = US * 1000
    S: int = MS * 1000
    
    def __init__(self, scale: int) -> None:
        self.cnt: int = 0
        self._sum: int = 0
        self._max: int | None = None
        self.scale = scale
        self.start_time: int = time.monotonic_ns()
    
    def collect(self, value: int) -> None:
        """收集时间样本"""
        self.cnt += 1
        self._sum += value
        if self._max is None:
            self._max = value
        else:
            self._max = max(self._max, value)
    
    def avg(self) -> float | str:
        """计算平均时间"""
        return self._sum * 1.0 / self.cnt / self.scale if self.cnt > 0 else "N/A"
    
    def max(self) -> float | str:
        """获取最大时间"""
        return self._max / self.scale if self._max else "N/A"
    
    def __enter__(self) -> None:
        """开始计时"""
        self.start_time = time.monotonic_ns()
    
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> None:
        """结束计时并收集"""
        self.collect(time.monotonic_ns() - self.start_time)
```

---

## 完整 Benchmark 流程示例

### 运行 Throughput 测试

```python
# 示例使用流程
from vllm.benchmarks.throughput import main
import argparse

args = argparse.Namespace(
    model="meta-llama/Llama-2-7b-chat-hf",
    num_prompts=100,
    input_len=512,
    output_len=128,
    dataset_name="random",
    backend="vllm",
    disable_detokenize=False,
)

main(args)
```

### 运行 Latency 测试

```python
from vllm.benchmarks.latency import main
import argparse

args = argparse.Namespace(
    model="meta-llama/Llama-2-7b-chat-hf",
    input_len=512,
    output_len=128,
    batch_size=32,
    n=1,
    num_iters_warmup=10,
    num_iters=30,
    profile=False,
)

main(args)
```

---

## 总结

vLLM 的 Benchmark 系统具有以下特点：

1. **多后端支持** - 支持 vLLM、Hugging Face、TGI、DeepSpeed-MII、TensorRT-LLM 等
2. **丰富的数据集** - 包含多种预定义数据集，满足不同场景测试需求
3. **完整的性能指标** - 提供吞吐率、延迟、TTFT、ITL、百分位数等多维指标
4. **可扩展性** - 模块化设计，支持添加新的数据集和后端
5. **CLI 友好** - 提供简单的命令行接口，方便使用

这个系统是进行模型性能分析、优化验证和对比测试的强大工具。
