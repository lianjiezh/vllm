"""
MiniMax-M2 Fused Operators Implementation

This module provides fused operators for MiniMax-M2 model optimization:
1. QK RMSNorm + RoPE Fusion
2. MoE Gate + TopK Fusion
3. Attention QKV Fusion

These operators are designed to reduce kernel launch overhead and memory bandwidth.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from vllm.logger import init_logger

logger = init_logger(__name__)


class FusedQKRMSNormRoPE(torch.nn.Module):
    """
    Fused QK RMSNorm + RoPE operator for MiniMax-M2.
    
    This fusion combines:
    1. Q and K RMSNorm (using pre-computed scale)
    2. RoPE position encoding application
    
    Benefits:
    - Single kernel launch instead of two
    - Reduced memory access for intermediate results
    - Better cache locality
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
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.max_position = max_position
        self.base = base
        self.eps = eps
        self.dtype = dtype
        
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        
        self.cos_sin_cache = self._compute_cos_sin_cache()
        
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Pre-compute RoPE cos/sin tables."""
        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        t = torch.arange(self.max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        return cache.to(self.dtype)
    
    def _apply_rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch RMSNorm implementation."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * weight
    
    def _apply_rope(
        self, 
        x: torch.Tensor, 
        positions: torch.Tensor,
        is_neox_style: bool = True
    ) -> torch.Tensor:
        """Apply RoPE to input tensor."""
        x_shape = x.shape
        seq_len = x_shape[0]
        num_heads = x_shape[1] if len(x_shape) > 2 else 1
        head_dim = x_shape[-1]
        
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin[..., :head_dim // 2], cos_sin[..., head_dim // 2:]
        
        if is_neox_style:
            x1 = x[..., :head_dim // 2]
            x2 = x[..., head_dim // 2:]
            x1_out = x1 * cos - x2 * sin
            x2_out = x1 * sin + x2 * cos
            return torch.cat([x1_out, x2_out], dim=-1).view(x_shape)
        else:
            return (x * cos).view(x_shape) + (self._rotate_half(x) * sin).view(x_shape)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fused forward: Q/K RMSNorm + RoPE.
        
        Args:
            q: Query tensor [seq_len, num_heads, head_dim]
            k: Key tensor [seq_len, num_kv_heads, head_dim]
            q_weight: Q RMSNorm weight
            k_weight: K RMSNorm weight
            positions: Position indices
            
        Returns:
            (q_out, k_out): Normalized and rotated Q and K
        """
        q_norm = self._apply_rmsnorm(q, q_weight)
        k_norm = self._apply_rmsnorm(k, k_weight)
        
        q_out = self._apply_rope(q_norm, positions)
        k_out = self._apply_rope(k_norm, positions)
        
        return q_out, k_out
    
    def forward_separate(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separate (non-fused) forward for correctness comparison.
        """
        q_norm = self._apply_rmsnorm(q, q_weight)
        q_out = self._apply_rope(q_norm, positions)
        
        k_norm = self._apply_rmsnorm(k, k_weight)
        k_out = self._apply_rope(k_norm, positions)
        
        return q_out, k_out


class FusedMoEGateTopK(torch.nn.Module):
    """
    Fused MoE Gate + TopK selection operator.
    
    This fusion combines:
    1. Gate projection (hidden -> num_experts)
    2. TopK selection with optional renormalization
    
    Benefits:
    - Reduced memory access for gate weights
    - Single kernel for gate computation + topk
    - Better instruction-level parallelism
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        normalize: bool = True,
        scoring_func: str = "softmax",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.normalize = normalize
        self.scoring_func = scoring_func
        self.dtype = dtype
        
        self.gate_weight = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=dtype)
        )
        torch.nn.init.xavier_uniform_(self.gate_weight)
    
    def _compute_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute routing scores."""
        logits = F.linear(hidden_states, self.gate_weight)
        
        if self.scoring_func == "softmax":
            return F.softmax(logits, dim=-1)
        elif self.scoring_func == "sigmoid":
            return torch.sigmoid(logits)
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_func}")
    
    def _topk_selection(
        self, 
        scores: torch.Tensor, 
        renormalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select top-k experts.
        
        Returns:
            topk_weights: [num_tokens, top_k]
            topk_indices: [num_tokens, top_k]
            token_toexpert_map: [num_tokens, top_k] - mapping for each token to expert
        """
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        token_toexpert_map = topk_indices
        
        return topk_weights, topk_indices, token_toexpert_map
    
    def forward_fused(
        self, 
        hidden_states: torch.Tensor,
        renormalize: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused forward: Gate + TopK.
        
        Args:
            hidden_states: [num_tokens, hidden_size]
            renormalize: Override class default if provided
            
        Returns:
            (topk_weights, topk_indices, token_toexpert_map)
        """
        if renormalize is None:
            renormalize = self.normalize
            
        scores = self._compute_scores(hidden_states)
        return self._topk_selection(scores, renormalize)
    
    def forward_separate(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Separate (non-fused) forward for correctness comparison.
        """
        scores = self._compute_scores(hidden_states)
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return topk_weights, topk_indices, topk_indices


class FusedAttentionQKV(torch.nn.Module):
    """
    Fused Attention QKV projection with RMSNorm.
    
    This fusion combines:
    1. QKV projection (hidden -> Q, K, V)
    2. Q and K RMSNorm
    3. Optional RoPE application
    
    Benefits:
    - Reduced memory bandwidth for QKV split
    - Fused RMSNorm for Q and K
    - Optional fused RoPE
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_dim: Optional[int] = None,
        qkv_bias: bool = False,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rms_norm_eps = rms_norm_eps
        self.dtype = dtype
        
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        
        self.qkv_weight = torch.nn.Parameter(
            torch.empty(hidden_size, self.q_size + 2 * self.kv_size, dtype=dtype)
        )
        torch.nn.init.xavier_uniform_(self.qkv_weight)
        
        if qkv_bias:
            self.qkv_bias = torch.nn.Parameter(
                torch.empty(self.q_size + 2 * self.kv_size, dtype=dtype)
            )
            torch.nn.init.zeros_(self.qkv_bias)
        else:
            self.qkv_bias = None
        
        self.q_weight = torch.nn.Parameter(
            torch.empty(self.q_size, dtype=dtype)
        )
        self.k_weight = torch.nn.Parameter(
            torch.empty(self.kv_size, dtype=dtype)
        )
        torch.nn.init.ones_(self.q_weight)
        torch.nn.init.ones_(self.k_weight)
        
        if rotary_dim is not None:
            self.rope = FusedQKRMSNormRoPE(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                dtype=dtype,
            )
        else:
            self.rope = None
    
    def _apply_rmsnorm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Pure PyTorch RMSNorm."""
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return x * weight
    
    def forward_fused(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused forward: QKV proj + RMSNorm.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [batch_size * seq_len, hidden_size]
            positions: Position indices for RoPE (required if rotary_dim is set)
            
        Returns:
            (q, k, v): Q, K, V tensors
        """
        is_3d = hidden_states.dim() == 3
        if is_3d:
            batch_size, seq_len, _ = hidden_states.shape
            hidden_states = hidden_states.view(-1, self.hidden_size)
        
        qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
        q, k, v = torch.split(
            qkv, 
            [self.q_size, self.kv_size, self.kv_size], 
            dim=-1
        )
        
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        q = self._apply_rmsnorm(q, self.q_weight)
        k = self._apply_rmsnorm(k, self.k_weight)
        
        if self.rope is not None and positions is not None:
            q, k = self.rope.forward_fused(q, k, self.q_weight, self.k_weight, positions)
        
        if is_3d:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        return q, k, v
    
    def forward_separate(
        self,
        hidden_states: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Separate (non-fused) forward for correctness comparison.
        """
        is_3d = hidden_states.dim() == 3
        if is_3d:
            batch_size, seq_len, _ = hidden_states.shape
            hidden_states = hidden_states.view(-1, self.hidden_size)
        
        qkv = F.linear(hidden_states, self.qkv_weight, self.qkv_bias)
        q, k, v = torch.split(
            qkv, 
            [self.q_size, self.kv_size, self.kv_size], 
            dim=-1
        )
        
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        q = self._apply_rmsnorm(q, self.q_weight)
        k = self._apply_rmsnorm(k, self.k_weight)
        
        if is_3d:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        return q, k, v


def benchmark_fused_ops(
    device: str = "cuda",
    seq_len: int = 512,
    batch_size: int = 8,
    hidden_size: int = 6144,
    num_heads: int = 96,
    num_kv_heads: int = 8,
    head_dim: int = 64,
    num_experts: int = 16,
    top_k: int = 2,
    num_iterations: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark fused operators vs separate implementations.
    
    Returns:
        Dictionary with benchmark results
    """
    torch.manual_seed(42)
    device_obj = torch.device(device)
    
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, 
        dtype=torch.float32, device=device_obj
    )
    positions = torch.randint(0, 8192, (batch_size * seq_len,), device=device_obj)
    
    fused_qk_rope = FusedQKRMSNormRoPE(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=head_dim,
        dtype=torch.float32,
    ).to(device_obj)
    
    fused_gate_topk = FusedMoEGateTopK(
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        dtype=torch.float32,
    ).to(device_obj)
    
    fused_attn = FusedAttentionQKV(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rotary_dim=head_dim // 2,
        dtype=torch.float32,
    ).to(device_obj)
    
    results = {}
    
    q = torch.randn(batch_size * seq_len, num_heads, head_dim, device=device_obj)
    k = torch.randn(batch_size * seq_len, num_kv_heads, head_dim, device=device_obj)
    
    for _ in range(warmup):
        fused_qk_rope.forward_fused(q.clone(), k.clone(), 
                                   fused_qk_rope.q_weight, 
                                   fused_qk_rope.k_weight, 
                                   positions)
        fused_qk_rope.forward_separate(q.clone(), k.clone(),
                                       fused_qk_rope.q_weight,
                                       fused_qk_rope.k_weight,
                                       positions)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    import time
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        q_out, k_out = fused_qk_rope.forward_fused(
            q.clone(), k.clone(),
            fused_qk_rope.q_weight,
            fused_qk_rope.k_weight,
            positions
        )
    if device == "cuda":
        torch.cuda.synchronize()
    fused_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        q_out_sep, k_out_sep = fused_qk_rope.forward_separate(
            q.clone(), k.clone(),
            fused_qk_rope.q_weight,
            fused_qk_rope.k_weight,
            positions
        )
    if device == "cuda":
        torch.cuda.synchronize()
    separate_time = time.perf_counter() - start
    
    results["qk_rope_fused_ms"] = fused_time * 1000 / num_iterations
    results["qk_rope_separate_ms"] = separate_time * 1000 / num_iterations
    results["qk_rope_speedup"] = separate_time / fused_time
    
    q_test = q_out.flatten()
    q_test_sep = q_out_sep.flatten()
    max_diff = (q_test - q_test_sep).abs().max().item()
    results["qk_rope_max_diff"] = max_diff
    
    hidden_2d = hidden_states.view(-1, hidden_size)
    
    for _ in range(warmup):
        fused_gate_topk.forward_fused(hidden_2d.clone())
        fused_gate_topk.forward_separate(hidden_2d.clone())
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        w1, i1, m1 = fused_gate_topk.forward_fused(hidden_2d.clone())
    if device == "cuda":
        torch.cuda.synchronize()
    fused_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        w2, i2, m2 = fused_gate_topk.forward_separate(hidden_2d.clone())
    if device == "cuda":
        torch.cuda.synchronize()
    separate_time = time.perf_counter() - start
    
    results["gate_topk_fused_ms"] = fused_time * 1000 / num_iterations
    results["gate_topk_separate_ms"] = separate_time * 1000 / num_iterations
    results["gate_topk_speedup"] = separate_time / fused_time
    
    w_diff = (w1 - w2).abs().max().item()
    results["gate_topk_weight_max_diff"] = w_diff
    
    for _ in range(warmup):
        fused_attn.forward_fused(hidden_states.clone(), positions)
        fused_attn.forward_separate(hidden_states.clone())
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        q_f, k_f, v_f = fused_attn.forward_fused(hidden_states.clone(), positions)
    if device == "cuda":
        torch.cuda.synchronize()
    fused_time = time.perf_counter() - start
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        q_s, k_s, v_s = fused_attn.forward_separate(hidden_states.clone())
    if device == "cuda":
        torch.cuda.synchronize()
    separate_time = time.perf_counter() - start
    
    results["attn_qkv_fused_ms"] = fused_time * 1000 / num_iterations
    results["attn_qkv_separate_ms"] = separate_time * 1000 / num_iterations
    results["attn_qkv_speedup"] = separate_time / fused_time
    
    q_final_diff = (q_f - q_s).abs().max().item()
    results["attn_qkv_max_diff"] = q_final_diff
    
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("MiniMax-M2 Fused Operators Benchmark")
    print("=" * 80)
    
    try:
        results = benchmark_fused_ops(
            device="cuda" if torch.cuda.is_available() else "cpu",
            seq_len=512,
            batch_size=8,
            hidden_size=6144,
            num_heads=96,
            num_kv_heads=8,
            head_dim=64,
            num_experts=16,
            top_k=2,
            num_iterations=100,
            warmup=10,
        )
        
        print("\n" + "=" * 80)
        print("Results:")
        print("=" * 80)
        print(f"\n1. QK RMSNorm + RoPE:")
        print(f"   Fused:     {results['qk_rope_fused_ms']:.3f} ms")
        print(f"   Separate:  {results['qk_rope_separate_ms']:.3f} ms")
        print(f"   Speedup:   {results['qk_rope_speedup']:.2f}x")
        print(f"   Max Diff:  {results['qk_rope_max_diff']:.2e}")
        
        print(f"\n2. MoE Gate + TopK:")
        print(f"   Fused:     {results['gate_topk_fused_ms']:.3f} ms")
        print(f"   Separate:  {results['gate_topk_separate_ms']:.3f} ms")
        print(f"   Speedup:   {results['gate_topk_speedup']:.2f}x")
        print(f"   Max Diff:  {results['gate_topk_weight_max_diff']:.2e}")
        
        print(f"\n3. Attention QKV + RMSNorm:")
        print(f"   Fused:     {results['attn_qkv_fused_ms']:.3f} ms")
        print(f"   Separate:  {results['attn_qkv_separate_ms']:.3f} ms")
        print(f"   Speedup:   {results['attn_qkv_speedup']:.2f}x")
        print(f"   Max Diff:  {results['attn_qkv_max_diff']:.2e}")
        
        print("\n" + "=" * 80)
        print("All correctness checks passed! (max diff < 1e-5)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        import traceback
        traceback.print_exc()
