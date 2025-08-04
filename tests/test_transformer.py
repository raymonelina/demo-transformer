import torch
import pytest
from demo_transformer.transformer import Transformer
from demo_transformer.config import TransformerConfig


def test_transformer_standard_attention():
    """Test the Transformer with standard attention."""
    print("\n\nRunning test_transformer_standard_attention\n")
    # Setup
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    vocab_size = 1000
    
    # Create a configuration
    config = TransformerConfig(
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=100,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        use_relative_pos=False,
        use_rope=False,
        debug_mode=True
    )
    
    # Create transformer model
    transformer = Transformer(config)
    
    # Create input tensors
    src_ids = torch.randint(1, vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
    tgt_ids[:, 0] = 1  # SOS token
    
    # Execute forward pass
    output = transformer(src_ids, tgt_ids)
    
    # Verify output dimensions
    assert output.shape == (batch_size, tgt_seq_len, vocab_size)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_transformer_relative_attention():
    """Test the Transformer with relative positional attention."""
    print("\n\nRunning test_transformer_relative_attention\n")
    # Setup
    batch_size = 2
    # For relative attention, use the same sequence length for source and target
    seq_len = 10
    vocab_size = 1000
    
    # Create a configuration
    config = TransformerConfig(
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=100,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        use_relative_pos=True,
        use_rope=False,
        debug_mode=True
    )
    
    # Create transformer model
    transformer = Transformer(config)
    
    # Create input tensors with the same sequence length
    src_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt_ids[:, 0] = 1  # SOS token
    
    # Execute forward pass
    output = transformer(src_ids, tgt_ids)
    
    # Verify output dimensions
    assert output.shape == (batch_size, seq_len, vocab_size)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_transformer_rope_attention():
    """Test the Transformer with RoPE attention."""
    print("\n\nRunning test_transformer_rope_attention\n")
    # Setup
    batch_size = 2
    seq_len = 10
    vocab_size = 1000
    num_heads = 4
    head_dim = 16  # Must be divisible by 2 for RoPE
    embed_dim = num_heads * head_dim  # 64
    
    # Create a configuration
    config = TransformerConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=100,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        use_relative_pos=False,
        use_rope=True,
        debug_mode=True
    )
    
    # Create transformer model
    transformer = Transformer(config)
    
    # Create input tensors with the same sequence length
    src_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    tgt_ids[:, 0] = 1  # SOS token
    
    # Execute forward pass
    output = transformer(src_ids, tgt_ids)
    
    # Verify output dimensions
    assert output.shape == (batch_size, seq_len, vocab_size)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_transformer_with_padding_masks():
    """Test the Transformer with source and target padding masks."""
    print("\n\nRunning test_transformer_with_padding_masks\n")
    # Setup
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    vocab_size = 1000
    
    # Create a configuration
    config = TransformerConfig(
        embed_dim=32,
        num_heads=4,
        ff_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=100,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        debug_mode=True
    )
    
    # Create transformer model
    transformer = Transformer(config)
    
    # Create input tensors
    src_ids = torch.randint(1, vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
    tgt_ids[:, 0] = 1  # SOS token
    
    # Create source padding mask (mask out the last 2 tokens)
    src_padding_mask = torch.zeros(batch_size, 1, 1, src_seq_len, dtype=torch.bool)
    src_padding_mask[:, :, :, -2:] = True
    
    # Create target padding mask (mask out the last token)
    tgt_padding_mask = torch.zeros(batch_size, 1, tgt_seq_len, tgt_seq_len, dtype=torch.bool)
    tgt_padding_mask[:, :, :, -1] = True
    
    # Execute forward pass with padding masks
    output = transformer(src_ids, tgt_ids, src_padding_mask, tgt_padding_mask)
    
    # Verify output dimensions
    assert output.shape == (batch_size, tgt_seq_len, vocab_size)
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_transformer_encode_decode_methods():
    """Test the separate encode and decode methods of the Transformer."""
    print("\n\nRunning test_transformer_encode_decode_methods\n")
    # Setup
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    vocab_size = 1000
    embed_dim = 32
    
    # Create a configuration
    config = TransformerConfig(
        embed_dim=embed_dim,
        num_heads=4,
        ff_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=100,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        debug_mode=True
    )
    
    # Create transformer model
    transformer = Transformer(config)
    
    # Create input tensors
    src_ids = torch.randint(1, vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
    tgt_ids[:, 0] = 1  # SOS token
    
    # Test encode method
    encoder_output = transformer.encode(src_ids)
    assert encoder_output.shape == (batch_size, src_seq_len, embed_dim)
    
    # Test decode method
    output = transformer.decode(tgt_ids, encoder_output)
    assert output.shape == (batch_size, tgt_seq_len, vocab_size)
    
    # Verify no NaN or infinite values
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


def test_transformer_weight_tying():
    """Test the weight tying feature of the Transformer."""
    print("\n\nRunning test_transformer_weight_tying\n")
    # Setup
    vocab_size = 1000
    embed_dim = 32
    
    # Create a configuration with weight tying enabled
    config = TransformerConfig(
        embed_dim=embed_dim,
        num_heads=4,
        ff_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=100,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        weight_tying=True,
        debug_mode=True
    )
    
    # Create transformer model
    transformer = Transformer(config)
    
    # Check that the embedding weights are tied
    assert transformer.decoder.output_projection.weight is transformer.decoder.token_embedding.weight
    
    # Create a configuration with different vocab sizes (weight tying should be disabled)
    config = TransformerConfig(
        embed_dim=embed_dim,
        num_heads=4,
        ff_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=100,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size + 100,  # Different vocab size
        weight_tying=True,
        debug_mode=True
    )
    
    # Create transformer model
    transformer = Transformer(config)
    
    # Check that the embedding weights are not tied
    assert transformer.decoder.output_projection.weight is not transformer.decoder.token_embedding.weight


def test_transformer_save_load(tmp_path):
    """Test saving and loading a Transformer model."""
    print("\n\nRunning test_transformer_save_load\n")
    # Setup
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    vocab_size = 1000
    embed_dim = 32
    
    # Create a configuration
    config = TransformerConfig(
        embed_dim=embed_dim,
        num_heads=4,
        ff_dim=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=100,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        debug_mode=True
    )
    
    # Create transformer model
    transformer = Transformer(config)
    
    # Create input tensors
    src_ids = torch.randint(1, vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(1, vocab_size, (batch_size, tgt_seq_len))
    tgt_ids[:, 0] = 1  # SOS token
    
    # Save the model
    model_path = tmp_path / "transformer_test.pt"
    transformer.save_pretrained(str(model_path))
    
    # Load the model
    loaded_transformer = Transformer.from_pretrained(str(model_path))
    
    # Verify model parameters are the same
    for param1, param2 in zip(transformer.parameters(), loaded_transformer.parameters()):
        assert torch.allclose(param1, param2), "Model parameters differ after loading"
    
    # Verify config was preserved
    assert loaded_transformer.config.embed_dim == config.embed_dim
    assert loaded_transformer.config.num_heads == config.num_heads
    assert loaded_transformer.config.ff_dim == config.ff_dim
    assert loaded_transformer.config.num_encoder_layers == config.num_encoder_layers
    assert loaded_transformer.config.num_decoder_layers == config.num_decoder_layers
    
    # Verify outputs are the same with the same input
    transformer.eval()  # Set to eval mode to disable dropout
    loaded_transformer.eval()
    
    with torch.no_grad():
        output_before = transformer(src_ids, tgt_ids)
        output_after = loaded_transformer(src_ids, tgt_ids)
        assert torch.allclose(output_before, output_after), "Model outputs differ after loading"