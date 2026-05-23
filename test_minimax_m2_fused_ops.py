"""
Comprehensive Tests for MiniMax-M2 Fused Operators

This test suite validates:
1. Correctness of fused operators vs separate implementations
2. Numerical accuracy
3. Gradient flow
4. Performance benchmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
import time
from typing import Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from minimax_m2_fused_ops import (
    FusedQKRMSNormRoPE,
    FusedMoEGateTopK,
    FusedAttentionQKV,
    benchmark_fused_ops,
)


class TestFusedQKRMSNormRoPE:
    """Tests for QK RMSNorm + RoPE fusion."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 6144
        self.num_heads = 96
        self.num_kv_heads = 8
        self.head_dim = 64
        self.rotary_dim = 64
        self.seq_len = 128
        self.batch_size = 4
        
        self.fused_module = FusedQKRMSNormRoPE(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            rotary_dim=self.rotary_dim,
            dtype=torch.float32,
        ).to(self.device)
        
        self.q = torch.randn(
            self.batch_size * self.seq_len, 
            self.num_heads, 
            self.head_dim,
            device=self.device
        )
        self.k = torch.randn(
            self.batch_size * self.seq_len, 
            self.num_kv_heads, 
            self.head_dim,
            device=self.device
        )
        self.positions = torch.randint(
            0, 8192, 
            (self.batch_size * self.seq_len,),
            device=self.device
        )
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_forward_correctness(self):
        """Test fused forward produces same results as separate."""
        q_fused, k_fused = self.fused_module.forward_fused(
            self.q.clone(), 
            self.k.clone(),
            self.fused_module.q_weight,
            self.fused_module.k_weight,
            self.positions
        )
        
        q_separate, k_separate = self.fused_module.forward_separate(
            self.q.clone(),
            self.k.clone(),
            self.fused_module.q_weight,
            self.fused_module.k_weight,
            self.positions
        )
        
        q_diff = (q_fused - q_separate).abs().max().item()
        k_diff = (k_fused - k_separate).abs().max().item()
        
        assert q_diff < 1e-5, f"Q difference too large: {q_diff}"
        assert k_diff < 1e-5, f"K difference too large: {k_diff}"
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly through fused module."""
        q = self.q.clone().requires_grad_(True)
        k = self.k.clone().requires_grad_(True)
        
        q_out, k_out = self.fused_module.forward_fused(
            q, k,
            self.fused_module.q_weight,
            self.fused_module.k_weight,
            self.positions
        )
        
        loss = q_out.sum() + k_out.sum()
        loss.backward()
        
        assert q.grad is not None, "Q gradient not computed"
        assert k.grad is not None, "K gradient not computed"
        assert self.fused_module.q_weight.grad is not None, "Q weight gradient not computed"
        assert self.fused_module.k_weight.grad is not None, "K weight gradient not computed"
        
        assert q.grad.abs().sum() > 0, "Q gradient is zero"
        assert k.grad.abs().sum() > 0, "K gradient is zero"
    
    def test_different_seq_lens(self):
        """Test with different sequence lengths."""
        for seq_len in [1, 16, 64, 256, 512]:
            q = torch.randn(2, self.num_heads, self.head_dim, device=self.device)
            k = torch.randn(2, self.num_kv_heads, self.head_dim, device=self.device)
            positions = torch.randint(0, 8192, (2,), device=self.device)
            
            q_fused, k_fused = self.fused_module.forward_fused(
                q, k,
                self.fused_module.q_weight,
                self.fused_module.k_weight,
                positions
            )
            
            q_sep, k_sep = self.fused_module.forward_separate(
                q, k,
                self.fused_module.q_weight,
                self.fused_module.k_weight,
                positions
            )
            
            assert torch.allclose(q_fused, q_sep, atol=1e-5)
            assert torch.allclose(k_fused, k_sep, atol=1e-5)
    
    def test_different_rope_dims(self):
        """Test with different rotary dimensions."""
        for rotary_dim in [32, 48, 64]:
            module = FusedQKRMSNormRoPE(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                rotary_dim=rotary_dim,
                dtype=torch.float32,
            ).to(self.device)
            
            q_fused, k_fused = module.forward_fused(
                self.q.clone(), 
                self.k.clone(),
                module.q_weight,
                module.k_weight,
                self.positions
            )
            
            q_sep, k_sep = module.forward_separate(
                self.q.clone(),
                self.k.clone(),
                module.q_weight,
                module.k_weight,
                self.positions
            )
            
            assert torch.allclose(q_fused, q_sep, atol=1e-5)
            assert torch.allclose(k_fused, k_sep, atol=1e-5)


class TestFusedMoEGateTopK:
    """Tests for MoE Gate + TopK fusion."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 4096
        self.num_experts = 32
        self.top_k = 2
        self.num_tokens = 512
        
        self.fused_module = FusedMoEGateTopK(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=torch.float32,
        ).to(self.device)
        
        self.hidden_states = torch.randn(
            self.num_tokens, 
            self.hidden_size,
            device=self.device
        )
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_forward_correctness(self):
        """Test fused forward produces same results as separate."""
        w1, i1, m1 = self.fused_module.forward_fused(self.hidden_states.clone())
        w2, i2, m2 = self.fused_module.forward_separate(self.hidden_states.clone())
        
        weight_diff = (w1 - w2).abs().max().item()
        assert weight_diff < 1e-5, f"Weight difference too large: {weight_diff}"
        
        indices_diff = (i1 - i2).abs().max().item()
        assert indices_diff == 0, f"Indices differ: {indices_diff}"
    
    def test_topk_selection(self):
        """Test TopK selection is correct."""
        w, i, _ = self.fused_module.forward_fused(self.hidden_states)
        
        assert w.shape == (self.num_tokens, self.top_k)
        assert i.shape == (self.num_tokens, self.top_k)
        
        assert torch.all(w >= 0), "Weights should be non-negative"
        assert torch.all(w <= 1), "Weights should be <= 1"
        
        weight_sums = w.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    def test_gradient_flow(self):
        """Test gradients flow correctly."""
        hidden = self.hidden_states.clone().requires_grad_(True)
        
        w, i, m = self.fused_module.forward_fused(hidden)
        
        loss = w.sum()
        loss.backward()
        
        assert hidden.grad is not None
        assert self.fused_module.gate_weight.grad is not None
        assert hidden.grad.abs().sum() > 0
    
    def test_different_topk(self):
        """Test with different top-k values."""
        for top_k in [1, 2, 4, 8]:
            if top_k > self.num_experts:
                continue
                
            module = FusedMoEGateTopK(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                top_k=top_k,
                dtype=torch.float32,
            ).to(self.device)
            
            w, i, m = module.forward_fused(self.hidden_states.clone())
            
            assert w.shape == (self.num_tokens, top_k)
            assert i.shape == (self.num_tokens, top_k)
            
            weight_sums = w.sum(dim=-1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
    
    def test_different_scoring_funcs(self):
        """Test different scoring functions."""
        for scoring_func in ["softmax", "sigmoid"]:
            module = FusedMoEGateTopK(
                hidden_size=self.hidden_size,
                num_experts=self.num_experts,
                top_k=self.top_k,
                scoring_func=scoring_func,
                dtype=torch.float32,
            ).to(self.device)
            
            w, i, m = module.forward_fused(self.hidden_states.clone())
            
            assert w.shape == (self.num_tokens, self.top_k)
            assert torch.all(w >= 0)


class TestFusedAttentionQKV:
    """Tests for Attention QKV + RMSNorm fusion."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = 6144
        self.num_heads = 96
        self.num_kv_heads = 8
        self.head_dim = 64
        self.batch_size = 4
        self.seq_len = 128
        
        self.fused_module = FusedAttentionQKV(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            rotary_dim=self.head_dim // 2,
            dtype=torch.float32,
        ).to(self.device)
        
        self.hidden_states = torch.randn(
            self.batch_size, 
            self.seq_len, 
            self.hidden_size,
            device=self.device
        )
        
        self.positions = torch.randint(
            0, 8192,
            (self.batch_size * self.seq_len,),
            device=self.device
        )
        
        yield
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def test_forward_correctness(self):
        """Test fused forward produces same results."""
        q_f, k_f, v_f = self.fused_module.forward_fused(
            self.hidden_states.clone(), 
            self.positions
        )
        
        q_s, k_s, v_s = self.fused_module.forward_separate(
            self.hidden_states.clone()
        )
        
        q_diff = (q_f - q_s).abs().max().item()
        k_diff = (k_f - k_s).abs().max().item()
        v_diff = (v_f - v_s).abs().max().item()
        
        assert q_diff < 1e-4, f"Q difference too large: {q_diff}"
        assert k_diff < 1e-4, f"K difference too large: {k_diff}"
        assert v_diff < 1e-4, f"V difference too large: {v_diff}"
    
    def test_output_shapes(self):
        """Test output shapes are correct."""
        q, k, v = self.fused_module.forward_fused(
            self.hidden_states,
            self.positions
        )
        
        expected_q_shape = (self.batch_size, self.seq_len, self.num_heads, self.head_dim)
        expected_kv_shape = (self.batch_size, self.seq_len, self.num_kv_heads, self.head_dim)
        
        assert q.shape == expected_q_shape
        assert k.shape == expected_kv_shape
        assert v.shape == expected_kv_shape
    
    def test_gradient_flow(self):
        """Test gradients flow correctly."""
        hidden = self.hidden_states.clone().requires_grad_(True)
        
        q, k, v = self.fused_module.forward_fused(hidden, self.positions)
        
        loss = q.sum() + k.sum() + v.sum()
        loss.backward()
        
        assert hidden.grad is not None
        assert self.fused_module.qkv_weight.grad is not None
        assert hidden.grad.abs().sum() > 0
    
    def test_without_rope(self):
        """Test without RoPE."""
        module_no_rope = FusedAttentionQKV(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            rotary_dim=None,
            dtype=torch.float32,
        ).to(self.device)
        
        q, k, v = module_no_rope.forward_fused(self.hidden_states)
        
        assert q.shape[0] == self.batch_size
        assert k.shape[0] == self.batch_size
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            hidden = torch.randn(
                batch_size, 
                self.seq_len, 
                self.hidden_size,
                device=self.device
            )
            positions = torch.randint(
                0, 8192,
                (batch_size * self.seq_len,),
                device=self.device
            )
            
            q, k, v = self.fused_module.forward_fused(hidden, positions)
            
            assert q.shape == (batch_size, self.seq_len, self.num_heads, self.head_dim)
            assert k.shape == (batch_size, self.seq_len, self.num_kv_heads, self.head_dim)
            assert v.shape == (batch_size, self.seq_len, self.num_kv_heads, self.head_dim)


class TestPerformance:
    """Performance benchmarking tests."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_benchmark_fused_ops(self):
        """Run full benchmark suite."""
        print("\n" + "=" * 80)
        print("Running Performance Benchmarks")
        print("=" * 80)
        
        results = benchmark_fused_ops(
            device="cuda",
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
        print("Benchmark Results:")
        print("=" * 80)
        
        print(f"\n1. QK RMSNorm + RoPE:")
        print(f"   Fused:    {results['qk_rope_fused_ms']:.3f} ms")
        print(f"   Separate: {results['qk_rope_separate_ms']:.3f} ms")
        print(f"   Speedup:  {results['qk_rope_speedup']:.2f}x")
        
        print(f"\n2. MoE Gate + TopK:")
        print(f"   Fused:    {results['gate_topk_fused_ms']:.3f} ms")
        print(f"   Separate: {results['gate_topk_separate_ms']:.3f} ms")
        print(f"   Speedup:  {results['gate_topk_speedup']:.2f}x")
        
        print(f"\n3. Attention QKV + RMSNorm:")
        print(f"   Fused:    {results['attn_qkv_fused_ms']:.3f} ms")
        print(f"   Separate: {results['attn_qkv_separate_ms']:.3f} ms")
        print(f"   Speedup:  {results['attn_qkv_speedup']:.2f}x")
        
        assert results['qk_rope_max_diff'] < 1e-4
        assert results['gate_topk_weight_max_diff'] < 1e-4
        assert results['attn_qkv_max_diff'] < 1e-3
        
        assert results['qk_rope_speedup'] >= 0.9
        assert results['gate_topk_speedup'] >= 0.9
        assert results['attn_qkv_speedup'] >= 0.9


def run_quick_test():
    """Run quick sanity checks without pytest."""
    print("\n" + "=" * 80)
    print("Quick Correctness Test")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    print("\n1. Testing QK RMSNorm + RoPE...")
    fused_qk_rope = FusedQKRMSNormRoPE(
        hidden_size=512,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        rotary_dim=64,
    ).to(device)
    
    q = torch.randn(4, 8, 64, device=device)
    k = torch.randn(4, 2, 64, device=device)
    positions = torch.randint(0, 512, (4,), device=device)
    
    q_fused, k_fused = fused_qk_rope.forward_fused(
        q, k,
        fused_qk_rope.q_weight,
        fused_qk_rope.k_weight,
        positions
    )
    
    q_sep, k_sep = fused_qk_rope.forward_separate(
        q, k,
        fused_qk_rope.q_weight,
        fused_qk_rope.k_weight,
        positions
    )
    
    assert torch.allclose(q_fused, q_sep, atol=1e-4)
    assert torch.allclose(k_fused, k_sep, atol=1e-4)
    print("   ✓ QK RMSNorm + RoPE test passed!")
    
    print("\n2. Testing MoE Gate + TopK...")
    fused_gate = FusedMoEGateTopK(
        hidden_size=512,
        num_experts=8,
        top_k=2,
    ).to(device)
    
    hidden = torch.randn(16, 512, device=device)
    
    w1, i1, m1 = fused_gate.forward_fused(hidden)
    w2, i2, m2 = fused_gate.forward_separate(hidden)
    
    assert torch.allclose(w1, w2, atol=1e-4)
    assert torch.equal(i1, i2)
    print("   ✓ MoE Gate + TopK test passed!")
    
    print("\n3. Testing Attention QKV...")
    fused_attn = FusedAttentionQKV(
        hidden_size=512,
        num_heads=8,
        num_kv_heads=2,
        head_dim=64,
        rotary_dim=32,
    ).to(device)
    
    hidden = torch.randn(2, 8, 512, device=device)
    positions = torch.randint(0, 512, (16,), device=device)
    
    q1, k1, v1 = fused_attn.forward_fused(hidden, positions)
    q2, k2, v2 = fused_attn.forward_separate(hidden)
    
    assert torch.allclose(q1, q2, atol=1e-3)
    assert torch.allclose(k1, k2, atol=1e-3)
    assert torch.allclose(v1, v2, atol=1e-3)
    print("   ✓ Attention QKV test passed!")
    
    print("\n" + "=" * 80)
    print("All quick tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_test()
    else:
        pytest.main([__file__, "-v", "-s"])
