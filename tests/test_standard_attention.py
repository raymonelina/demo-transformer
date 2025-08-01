"""
Tests for standard Multi-Head Attention implementation.
"""

import torch
import pytest
from demo_transformer.multi_head_attention import MultiHeadAttention


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
    """Test the forward pass of MultiHeadAttention with a mask and verify mask is applied."""
    print("\n\nRunning test_multi_head_attention_with_mask\n")
    # Setup
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    num_heads = 2

    # Create attention module with attention storage enabled to verify masking
    attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, store_attention=True)

    # Create simple input tensors
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # Create a mask that masks out the last token (position 3)
    mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool)
    mask[:, :, :, -1] = True  # Mask out attention to the last position

    # Execute forward pass with mask
    output = attention(query, key, value, mask)

    # Basic assertions
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"

    # Verify mask is working: attention weights to masked positions should be 0
    attention_weights = (
        attention.last_attention_weights
    )  # [batch_size, num_heads, seq_len, seq_len]
    assert attention_weights is not None, "Attention weights should be stored"

    # Check that attention to the last position (index 3) is zero for all queries
    masked_attention = attention_weights[:, :, :, -1]  # [batch_size, num_heads, seq_len]
    assert torch.allclose(
        masked_attention, torch.zeros_like(masked_attention), atol=1e-6
    ), "Attention weights to masked positions should be zero"

    # Verify that attention weights still sum to 1 across non-masked positions
    # Sum over the key dimension (last dim), excluding the masked position
    unmasked_attention_sum = attention_weights[:, :, :, :-1].sum(
        dim=-1
    )  # [batch_size, num_heads, seq_len]
    assert torch.allclose(
        unmasked_attention_sum, torch.ones_like(unmasked_attention_sum), atol=1e-5
    ), "Attention weights should sum to 1 across unmasked positions"
