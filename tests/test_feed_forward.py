import torch
import pytest
from demo_transformer.feed_forward import FeedForwardBlock


def test_feed_forward_block_dimensions():
    """Test that the FeedForwardBlock preserves input/output dimensions."""
    print("\n\nRunning test_feed_forward_block_dimensions\n")
    # Setup
    batch_size = 2
    seq_len = 4
    embed_dim = 8
    ff_dim = 16
    
    # Create a feed forward block
    ff_block = FeedForwardBlock(embed_dim=embed_dim, ff_dim=ff_dim)
    
    # Create input tensor
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # Execute forward pass
    output = ff_block(x)
    
    # Verify output dimensions match input dimensions
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_feed_forward_block_with_different_dimensions():
    """Test the FeedForwardBlock with different input dimensions."""
    print("\n\nRunning test_feed_forward_block_with_different_dimensions\n")
    # Setup
    embed_dim = 16
    ff_dim = 64
    
    # Create a feed forward block
    ff_block = FeedForwardBlock(embed_dim=embed_dim, ff_dim=ff_dim)
    
    # Test with different batch sizes and sequence lengths
    test_cases = [
        (1, 10),  # batch_size=1, seq_len=10
        (4, 5),   # batch_size=4, seq_len=5
        (8, 1),   # batch_size=8, seq_len=1
    ]
    
    for batch_size, seq_len in test_cases:
        # Create input tensor
        x = torch.randn(batch_size, seq_len, embed_dim)
        
        # Execute forward pass
        output = ff_block(x)
        
        # Verify output dimensions
        assert output.shape == (batch_size, seq_len, embed_dim)
        assert not torch.isnan(output).any(), f"Output contains NaN values for batch_size={batch_size}, seq_len={seq_len}"
        assert not torch.isinf(output).any(), f"Output contains infinite values for batch_size={batch_size}, seq_len={seq_len}"