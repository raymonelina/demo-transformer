import torch
import pytest
from demo_transformer.attention import MultiHeadAttention


def test_multi_head_attention_forward():
    """Test the forward pass of MultiHeadAttention with simple inputs."""
    print("\n\nRunning test_multi_head_attention_forward\n")
    # Setup
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    num_heads = 2

    # Create a simple attention module with debug printing enabled
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, debug_mode=True)

    # Create simple input tensors
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # Execute forward pass
    output = attention(query, key, value)

    # Basic assertions
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_multi_head_attention_with_mask():
    """Test the forward pass of MultiHeadAttention with a mask."""
    print("\n\nRunning test_multi_head_attention_with_mask\n")
    # Setup
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    num_heads = 2

    # Create a simple attention module with debug printing enabled
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, debug_mode=True)

    # Create simple input tensors
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # Create a simple mask (mask out the last token)
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool)
    mask[:, :, :, -1] = True

    # Execute forward pass with mask
    output = attention(query, key, value, mask)

    # Basic assertions
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"
