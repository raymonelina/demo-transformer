import torch
import pytest
from demo_transformer.encoder import TransformerEncoder


def test_transformer_encoder_dimensions():
    """Test that the TransformerEncoder preserves sequence length and outputs correct embedding dimension."""
    print("\n\nRunning test_transformer_encoder_dimensions\n")
    # Setup
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    embed_dim = 32
    num_heads = 4
    ff_dim = 64
    num_layers = 2
    max_seq_len = 100
    
    # Create encoder with debug mode enabled
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        debug_mode=True
    )
    
    # Create input tensor
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Execute forward pass
    output = encoder(input_ids)
    
    # Verify output dimensions
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_transformer_encoder_with_padding_mask():
    """Test the TransformerEncoder with a padding mask."""
    print("\n\nRunning test_transformer_encoder_with_padding_mask\n")
    # Setup
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    embed_dim = 32
    num_heads = 4
    ff_dim = 64
    num_layers = 2
    max_seq_len = 100
    
    # Create encoder with debug mode enabled
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        debug_mode=True
    )
    
    # Create input tensor
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create padding mask (mask out the last 3 tokens)
    src_padding_mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool)
    src_padding_mask[:, :, :, -3:] = True
    
    # Execute forward pass with padding mask
    output = encoder(input_ids, src_padding_mask)
    
    # Verify output dimensions
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_transformer_encoder_with_gradient_checkpointing():
    """Test the TransformerEncoder with gradient checkpointing enabled."""
    print("\n\nRunning test_transformer_encoder_with_gradient_checkpointing\n")
    # Setup
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    embed_dim = 32
    num_heads = 4
    ff_dim = 64
    num_layers = 2
    max_seq_len = 100
    
    # Create encoder with gradient checkpointing and debug mode enabled
    encoder = TransformerEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        use_gradient_checkpointing=True,
        debug_mode=True
    )
    
    # Set to training mode to activate gradient checkpointing
    encoder.train()
    
    # Create input tensor
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Execute forward pass
    output = encoder(input_ids)
    
    # Verify output dimensions
    assert output.shape == (batch_size, seq_len, embed_dim)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"