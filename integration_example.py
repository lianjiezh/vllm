"""
Integration Example: Replacing MiniMax-M2 Attention with Fused Operators

This example shows how to integrate the fused operators into the actual
MiniMax-M2 model implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    from minimax_m2_fused_ops import FusedQKRMSNormRoPE, FusedAttentionQKV
    FUSED_OPS_AVAILABLE = True
except ImportError:
    FUSED_OPS_AVAILABLE = False
    print("Warning: Fused operators not available. Using standard implementation.")


class MiniMaxM2AttentionFused(nn.Module):
    """
    Fused version of MiniMax-M2 Attention layer.
    
    This replaces the original attention implementation with fused operators
    for improved performance.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: int,
        max_position: int = 8192,
        base: float = 10000.0,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        
        if FUSED_OPS_AVAILABLE:
            self.use_fused = True
            self.fused_attention = FusedAttentionQKV(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rms_norm_eps=rms_norm_eps,
                dtype=dtype,
            )
        else:
            self.use_fused = False
            self.q_norm = nn.RMSNorm(num_heads * head_dim, eps=rms_norm_eps)
            self.k_norm = nn.RMSNorm(num_kv_heads * head_dim, eps=rms_norm_eps)
            self.qkv_proj = nn.Linear(hidden_size, num_heads * head_dim + 2 * num_kv_heads * head_dim)
            self.o_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
        self.scaling = head_dim ** -0.5
    
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with optional fused operators.
        
        Args:
            positions: Position indices [seq_len]
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            output: Attended hidden states [batch_size, seq_len, hidden_size]
        """
        if self.use_fused:
            return self._forward_fused(positions, hidden_states)
        else:
            return self._forward_standard(positions, hidden_states)
    
    def _forward_fused(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass using fused operators."""
        batch_size, seq_len, _ = hidden_states.shape
        
        q, k, v = self.fused_attention.forward_fused(
            hidden_states, 
            positions
        )
        
        q = q * self.scaling
        
        attn_output = self._flash_attention(q, k, v)
        
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output, _ = self.fused_attention.o_proj(attn_output)
        
        return output
    
    def _forward_standard(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Standard forward pass (fallback)."""
        batch_size, seq_len, _ = hidden_states.shape
        
        qkv, _ = self.qkv_proj(hidden_states)
        
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        q = q * self.scaling
        
        attn_output = self._flash_attention(q, k, v)
        
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        output, _ = self.o_proj(attn_output)
        
        return output
    
    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Simple flash attention implementation."""
        try:
            from flash_attn import flash_attn_func
            return flash_attn_func(
                query, key, value,
                causal=False,
                softmax_scale=self.scaling,
            )
        except ImportError:
            return self._naive_attention(query, key, value)
    
    def _naive_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Naive attention fallback."""
        scale = query.shape[-1] ** -0.5
        
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output.transpose(1, 2)


class MiniMaxM2MoEFused(nn.Module):
    """
    Fused version of MiniMax-M2 MoE layer.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        if FUSED_OPS_AVAILABLE:
            self.use_fused = True
            self.fused_gate = None
        else:
            self.use_fused = False
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
        self.w13 = nn.ModuleList([
            nn.Linear(hidden_size, intermediate_size, bias=False)
            for _ in range(num_experts)
        ])
        
        self.w2 = nn.ModuleList([
            nn.Linear(intermediate_size, hidden_size, bias=False)
            for _ in range(num_experts)
        ])
        
        self.act_fn = nn.SiLU()
        self.rms_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional fused gate + topk.
        """
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        
        if self.use_fused:
            topk_weights, topk_indices, _ = self.fused_gate.forward_fused(hidden_states)
        else:
            logits = self.gate(hidden_states)
            weights = torch.softmax(logits, dim=-1)
            topk_weights, topk_indices = torch.topk(weights, self.top_k, dim=-1)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        output = torch.zeros_like(hidden_states)
        
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            weight = topk_weights[:, i].unsqueeze(-1)
            
            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id
                if mask.any():
                    expert_input = hidden_states[mask]
                    expert_output = self.w2[expert_id](
                        self.act_fn(self.w13[expert_id](expert_input))
                    )
                    output[mask] += weight[mask] * expert_output
        
        output = output.view(batch_size, seq_len, -1)
        output = self.rms_norm(output)
        
        return output


def benchmark_integration():
    """
    Benchmark the fused integration vs standard implementation.
    """
    print("\n" + "=" * 80)
    print("Integration Benchmark: Fused vs Standard")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping benchmark.")
        return
    
    device = torch.device("cuda")
    
    batch_size = 4
    seq_len = 512
    hidden_size = 6144
    num_heads = 96
    num_kv_heads = 8
    head_dim = 64
    rotary_dim = 32
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Seq length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Num KV heads: {num_kv_heads}")
    print(f"  Head dim: {head_dim}")
    print()
    
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        device=device, dtype=torch.float32
    )
    positions = torch.randint(0, 8192, (seq_len,), device=device)
    
    standard_attn = MiniMaxM2AttentionFused(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        dtype=torch.float32,
    ).to(device)
    
    for _ in range(10):
        _ = standard_attn(positions, hidden_states.clone())
    torch.cuda.synchronize()
    
    import time
    
    num_iterations = 100
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = standard_attn(positions, hidden_states.clone())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    print(f"Standard Attention:")
    print(f"  Time per iteration: {elapsed * 1000 / num_iterations:.3f} ms")
    print(f"  Throughput: {batch_size * seq_len * num_iterations / elapsed / 1000:.1f}K tokens/sec")
    
    if FUSED_OPS_AVAILABLE:
        fused_attn = MiniMaxM2AttentionFused(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            dtype=torch.float32,
        ).to(device)
        
        for _ in range(10):
            _ = fused_attn(positions, hidden_states.clone())
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = fused_attn(positions, hidden_states.clone())
        torch.cuda.synchronize()
        elapsed_fused = time.perf_counter() - start
        
        print(f"\nFused Attention:")
        print(f"  Time per iteration: {elapsed_fused * 1000 / num_iterations:.3f} ms")
        print(f"  Throughput: {batch_size * seq_len * num_iterations / elapsed_fused / 1000:.1f}K tokens/sec")
        
        speedup = elapsed / elapsed_fused
        print(f"\nSpeedup: {speedup:.2f}x")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    if torch.cuda.is_available():
        benchmark_integration()
    else:
        print("CUDA not available. Running in CPU mode (limited functionality).")
        print("For full functionality, please run in a CUDA-enabled environment.")
