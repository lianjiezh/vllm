# vLLM 适配新模型完整流程指南

## 目录

1. [概述](#概述)
2. [准备工作](#准备工作)
3. [核心概念](#核心概念)
4. [适配流程](#适配流程)
   - [步骤1: 分析模型架构](#步骤1-分析模型架构)
   - [步骤2: 创建模型文件](#步骤2-创建模型文件)
   - [步骤3: 注册模型](#步骤3-注册模型)
   - [步骤4: 实现核心组件](#步骤4-实现核心组件)
   - [步骤5: 测试验证](#步骤5-测试验证)
5. [进阶优化](#进阶优化)
6. [常见问题](#常见问题)
7. [示例: 适配一个新模型](#示例-适配一个新模型)

---

## 概述

本指南详细介绍如何将新的 Hugging Face 模型适配到 vLLM 框架中。vLLM 通过优化的 CUDA kernel 和高效的 KV Cache 管理实现高吞吐的模型推理。

**关键优势**:
- 极高的 token 生成吞吐量
- 有效的内存管理
- 支持多种并行策略（TP/PP/DP）
- 支持 speculative decoding

---

## 准备工作

### 环境要求

```bash
# 基础环境
python >= 3.10
torch >= 2.1.0
transformers >= 4.38.0

# 推荐环境 (CUDA)
cuda >= 12.0
cudnn >= 8.9
```

### 代码仓库

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

### 工具准备

1. **Hugging Face Hub**: 下载目标模型
2. **NVIDIA Nsight Systems**: 性能分析
3. **PyTorch Profiler**: 性能分析

---

## 核心概念

### 模型注册机制

vLLM 使用模型注册表来管理支持的模型：

```python
# registry.py 中的模型映射
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "MiniMaxM2ForCausalLM": ("minimax_m2", "MiniMaxM2ForCausalLM"),
    # ... 更多模型
}
```

### 模型接口

所有 vLLM 模型必须实现以下接口：

| 接口 | 用途 | 必需方法 |
|------|------|---------|
| `VllmModel` | 基础模型接口 | `__init__`, `embed_input_ids`, `forward` |
| `VllmModelForTextGeneration` | 文本生成模型 | 额外需要 `compute_logits` |
| `VllmModelForPooling` | 嵌入模型 | 额外需要 `pooler` |

### 核心组件

一个典型的 decoder-only 模型包含：

1. **Embedding Layer** - 词嵌入
2. **Decoder Layer** - 解码器层（包含 Attention + FFN/MoE）
3. **Attention Layer** - 注意力机制（支持 MHA/GQA/MQA）
4. **RMSNorm/LayerNorm** - 归一化层
5. **LM Head** - 输出层

---

## 适配流程

### 步骤1: 分析模型架构

在开始实现前，需要深入分析目标模型的架构：

#### 1.1 收集模型信息

```python
from transformers import AutoConfig

# 加载模型配置
config = AutoConfig.from_pretrained("path/to/model")

# 关键参数
print(f"Hidden size: {config.hidden_size}")
print(f"Num heads: {config.num_attention_heads}")
print(f"Num kv heads: {config.num_key_value_heads}")
print(f"Hidden layers: {config.num_hidden_layers}")
print(f"Norm type: {config.normalization_type}")
print(f"Activation: {config.hidden_act}")
```

#### 1.2 识别关键特性

| 特性 | 说明 | 影响 |
|------|------|------|
| **Attention Type** | MHA/GQA/MQA/MLA | 决定 Attention 层实现 |
| **Norm Type** | RMSNorm/LayerNorm | 归一化层选择 |
| **Position Embedding** | RoPE/ALiBi/None | 位置编码实现 |
| **MoE** | 是否使用混合专家 | 需要 FusedMoE |
| **Parallelism** | TP/PP/DP 支持 | 分布式策略 |

#### 1.3 参考相似模型

查看 vLLM 中已有的相似模型实现：

```bash
# 查找相似模型
ls -la /workspace/vllm/model_executor/models/ | grep -i "llama\|mistral\|qwen"
```

---

### 步骤2: 创建模型文件

创建新的模型文件 `/workspace/vllm/model_executor/models/your_model.py`

#### 2.1 文件结构

```python
# your_model.py
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import make_layers
```

#### 2.2 模型类结构

```python
class YourModelAttention(nn.Module):
    """注意力层"""
    def __init__(self, hidden_size: int, num_heads: int, ...):
        pass
    
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        pass


class YourModelDecoderLayer(nn.Module):
    """解码器层"""
    def __init__(self, config: PretrainedConfig, prefix: str, ...):
        pass
    
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, residual: torch.Tensor) -> tuple:
        pass


@support_torch_compile
class YourModelModel(nn.Module):
    """基础模型"""
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        pass
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ...):
        pass


class YourModelForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    """完整的因果语言模型"""
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        pass
    
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, ...):
        pass
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass
```

---

### 步骤3: 注册模型

在 `registry.py` 中注册新模型：

```python
# vllm/model_executor/models/registry.py

_TEXT_GENERATION_MODELS = {
    # ... 现有模型 ...
    "YourModelForCausalLM": ("your_model", "YourModelForCausalLM"),
}
```

**注意**: 如果模型使用标准 Hugging Face 命名（如 `AutoModelForCausalLM`），vLLM 会自动检测。

---

### 步骤4: 实现核心组件

#### 4.1 Attention 层实现

```python
class YourModelAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_dim: int,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        
        # 计算维度
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.scaling = self.head_dim ** -0.5
        
        # QKV 投影
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        
        # 输出投影
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        
        # RoPE 位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=8192,
        )
        
        # 注意力层
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([
            self.num_heads * self.head_dim,
            self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim
        ], dim=-1)
        
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
```

#### 4.2 Decoder Layer 实现

```python
class YourModelDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        model_config: ModelConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        
        self.hidden_size = config.hidden_size
        
        # 自注意力层
        self.self_attn = YourModelAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rotary_dim=config.rotary_dim,
            head_dim=getattr(config, "head_dim", None),
            rms_norm_eps=config.rms_norm_eps,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        
        # MLP/FFN
        self.mlp = YourModelMLP(
            config=config,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        
        # 归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        
        # MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        
        return hidden_states, residual
```

#### 4.3 完整模型实现

```python
@support_torch_compile
class YourModelModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        
        # 词嵌入
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        
        # 解码器层
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda layer_prefix: YourModelDecoderLayer(
                config,
                layer_prefix,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers",
        )
        
        # 最终归一化
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 处理输入
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        
        # 逐层前向
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 最终归一化
        if get_pp_group().is_last_rank:
            hidden_states, _ = self.norm(hidden_states, residual)
        
        return hidden_states


class YourModelForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        
        self.model = YourModelModel(vllm_config=vllm_config, prefix=prefix)
        
        # LM Head
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.lm_head",
            )
        else:
            self.lm_head = PPMissingLayer()
        
        self.logits_processor = LogitsProcessor(config.vocab_size)
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)
    
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
```

---

### 步骤5: 测试验证

#### 5.1 基础测试

```python
# test_your_model.py
import torch
from vllm import LLM, SamplingParams

# 测试基本推理
llm = LLM(
    model="path/to/your/model",
    trust_remote_code=True,
    tensor_parallel_size=1,
)

sampling_params = SamplingParams(max_tokens=100)
outputs = llm.generate("Hello, world!", sampling_params=sampling_params)

for output in outputs:
    print(output.prompt)
    print(output.outputs[0].text)
```

#### 5.2 单元测试

创建测试文件 `/workspace/tests/models/test_your_model.py`：

```python
import pytest
import torch

from vllm.model_executor.models.your_model import YourModelForCausalLM
from vllm.config import VllmConfig, ModelConfig


class TestYourModel:
    def test_model_init(self):
        """测试模型初始化"""
        config = ModelConfig(
            model="path/to/model",
            dtype="bfloat16",
        )
        vllm_config = VllmConfig(model_config=config)
        
        model = YourModelForCausalLM(vllm_config=vllm_config)
        assert model is not None
    
    def test_forward(self):
        """测试前向传播"""
        config = ModelConfig(
            model="path/to/model",
            dtype="bfloat16",
        )
        vllm_config = VllmConfig(model_config=config)
        
        model = YourModelForCausalLM(vllm_config=vllm_config)
        model = model.to("cuda")
        
        input_ids = torch.randint(0, 1000, (2, 32)).cuda()
        positions = torch.arange(32).cuda().repeat(2)
        
        hidden_states = model(input_ids, positions)
        assert hidden_states is not None
    
    def test_compute_logits(self):
        """测试 logits 计算"""
        config = ModelConfig(
            model="path/to/model",
            dtype="bfloat16",
        )
        vllm_config = VllmConfig(model_config=config)
        
        model = YourModelForCausalLM(vllm_config=vllm_config)
        model = model.to("cuda")
        
        hidden_states = torch.randn(2, 32, 512).cuda()
        logits = model.compute_logits(hidden_states)
        
        assert logits is not None
        assert logits.shape[-1] == config.hf_config.vocab_size
```

#### 5.3 性能测试

```python
# benchmark_your_model.py
import torch
import time
from vllm import LLM, SamplingParams

llm = LLM(
    model="path/to/your/model",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.9,
)

# 预热
for _ in range(5):
    llm.generate("Hello", sampling_params=SamplingParams(max_tokens=32))

# 性能测试
num_iterations = 100
start = time.perf_counter()

for _ in range(num_iterations):
    llm.generate(
        "What is the meaning of life?",
        sampling_params=SamplingParams(max_tokens=128)
    )

elapsed = time.perf_counter() - start
tokens_per_sec = (num_iterations * 128) / elapsed

print(f"Throughput: {tokens_per_sec:.1f} tokens/sec")
```

---

## 进阶优化

### 5.1 量化支持

```python
# 添加量化支持
from vllm.model_executor.layers.quantization import QuantizationConfig

class YourModelAttention(nn.Module):
    def __init__(self, quant_config: QuantizationConfig | None = None, ...):
        self.qkv_proj = QKVParallelLinear(
            ...,
            quant_config=quant_config,
        )
```

### 5.2 混合专家 (MoE)

```python
from vllm.model_executor.layers.fused_moe import FusedMoE

class YourModelMoE(nn.Module):
    def __init__(self, config: PretrainedConfig, quant_config=None):
        self.experts = FusedMoE(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant_config=quant_config,
        )
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states.to(torch.float32))
        return self.experts(hidden_states=hidden_states, router_logits=router_logits)
```

### 5.3 CUDA Graph 优化

```python
# 在模型中启用 CUDA Graph
from vllm.compilation.decorators import support_torch_compile

@support_torch_compile
class YourModelModel(nn.Module):
    def __init__(self, ...):
        # 启用编译优化
        self.compile_config = vllm_config.compilation_config
```

### 5.4 自定义算子

```python
# 注册自定义算子
from vllm.model_executor.custom_op import CustomOp

@CustomOp.register("your_custom_op")
class YourCustomOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 前向计算
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播
        return grad_output
```

---

## 常见问题

### Q1: 模型加载失败

**原因**: 权重名称不匹配

**解决方案**:
```python
# 在模型类中实现 load_weights 方法
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    params_dict = dict(self.named_parameters())
    loaded_params = set()
    
    for name, loaded_weight in weights:
        # 权重名称映射
        name = name.replace("old_name", "new_name")
        
        if name in params_dict:
            param = params_dict[name]
            param.data.copy_(loaded_weight)
            loaded_params.add(name)
    
    return loaded_params
```

### Q2: 性能不佳

**检查项**:
1. 确认使用了 FlashAttention
2. 检查 KV Cache 布局
3. 验证 tensor parallel 设置

### Q3: 内存不足

**解决方案**:
1. 减少 `gpu_memory_utilization`
2. 启用 `swap_space`
3. 使用 FP8/INT8 量化

---

## 示例: 适配一个新模型

假设我们要适配一个名为 `MyModel` 的新模型：

### 1. 分析模型

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("my-org/my-model")
print(config)
```

### 2. 创建模型文件

```python
# vllm/model_executor/models/my_model.py
class MyModelAttention(nn.Module):
    # 实现注意力层
    pass

class MyModelDecoderLayer(nn.Module):
    # 实现解码器层
    pass

class MyModelModel(nn.Module):
    # 实现基础模型
    pass

class MyModelForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    # 实现完整模型
    pass
```

### 3. 注册模型

```python
# registry.py
_TEXT_GENERATION_MODELS = {
    "MyModelForCausalLM": ("my_model", "MyModelForCausalLM"),
}
```

### 4. 测试

```bash
python -m pytest tests/models/test_my_model.py -v
```

---

## 参考资源

### 官方文档
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Model Implementation Guide](https://docs.vllm.ai/en/latest/models/adding_new_model.html)

### 核心文件
| 文件 | 说明 |
|------|------|
| `model_executor/models/registry.py` | 模型注册表 |
| `model_executor/models/interfaces.py` | 模型接口定义 |
| `model_executor/layers/attention.py` | 注意力层 |
| `model_executor/layers/fused_moe/` | MoE 实现 |

### 示例模型
- [Llama](file:///workspace/vllm/model_executor/models/llama.py) - 标准 decoder-only
- [MiniMax-M2](file:///workspace/vllm/model_executor/models/minimax_m2.py) - GQA + MoE
- [Mixtral](file:///workspace/vllm/model_executor/models/mixtral.py) - MoE

---

## 总结

适配新模型到 vLLM 的关键步骤：

1. **分析架构** - 理解模型的核心特性
2. **实现组件** - Attention、Decoder Layer、完整模型
3. **注册模型** - 在 registry.py 中添加映射
4. **测试验证** - 功能测试 + 性能测试
5. **优化迭代** - 量化、并行、自定义算子

通过遵循这个流程，可以高效地将新模型适配到 vLLM 框架中，享受其高性能的推理能力。
