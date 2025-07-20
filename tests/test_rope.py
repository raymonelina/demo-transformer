"""
Tests for Rotary Position Embedding (RoPE) implementation.
"""

import torch
import pytest
from demo_transformer.rotary_positional_encoding import RotaryPositionalEncoding
from demo_transformer.rope_attention import RoPEMultiHeadAttention


def test_rotary_positional_encoding_initialization():
    """Test that RotaryPositionalEncoding initializes correctly."""
    embed_dim = 64
    max_seq_len = 128
    
    rope = RotaryPositionalEncoding(embed_dim, max_seq_len)
    
    # Check that the frequency tensor was created correctly
    assert hasattr(rope, "_freqs_cis")
    assert rope._freqs_cis.shape == (max_seq_len, embed_dim // 2)
    assert rope._freqs_cis.dtype == torch.complex64


def test_rotary_positional_encoding_forward():
    """Test that RotaryPositionalEncoding forward pass works correctly."""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    max_seq_len = 128
    
    rope = RotaryPositionalEncoding(embed_dim, max_seq_len)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Forward should return input unchanged (actual rotation happens in attention)
    output = rope(x)
    assert torch.allclose(output, x)


def test_rotary_positional_encoding_apply_rotary():
    """Test that apply_rotary_pos_emb works correctly."""
    batch_size = 2
    seq_len = 10
    num_heads = 4
    head_dim = 16  # Must be divisible by 2 for RoPE
    embed_dim = num_heads * head_dim
    max_seq_len = 128
    
    rope = RotaryPositionalEncoding(head_dim, max_seq_len)
    
    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    q_rot, k_rot = rope.apply_rotary_pos_emb(q, k, seq_len)
    
    # Check that shapes are preserved
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    
    # Check that the rotated tensors are different from the originals
    assert not torch.allclose(q_rot, q)
    assert not torch.allclose(k_rot, k)


def test_rope_multi_head_attention():
    """Test that RoPEMultiHeadAttention works correctly."""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4
    
    attention = RoPEMultiHeadAttention(embed_dim, num_heads)
    
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)
    
    output = attention(q, k, v)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_rope_multi_head_attention_with_mask():
    """Test that RoPEMultiHeadAttention works correctly with a mask."""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4
    
    attention = RoPEMultiHeadAttention(embed_dim, num_heads)
    
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create a causal mask (lower triangular)
    mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len))
    
    output = attention(q, k, v, mask)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, embed_dim)


def test_rope_multi_head_attention_store_attention():
    """Test that RoPEMultiHeadAttention stores attention weights correctly."""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4
    
    attention = RoPEMultiHeadAttention(embed_dim, num_heads, store_attention=True)
    
    q = torch.randn(batch_size, seq_len, embed_dim)
    k = torch.randn(batch_size, seq_len, embed_dim)
    v = torch.randn(batch_size, seq_len, embed_dim)
    
    attention(q, k, v)
    
    # Check that attention weights were stored
    assert attention.last_attn_weights is not None
    assert attention.last_attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)