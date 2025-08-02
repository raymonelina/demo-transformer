"""
Example script demonstrating the Rotary Position Embedding (RoPE) implementation.
"""

import torch
from demo_transformer import Transformer, TransformerConfig


def main():
    """Run a simple demonstration of RoPE."""
    print("Creating transformer model with RoPE...")
    
    # Create a configuration with RoPE enabled
    config = TransformerConfig(
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_seq_len=512,
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        use_rope=True,  # Enable RoPE
        debug_mode=True,  # Enable debug printing
    )
    
    # Initialize the model
    model = Transformer(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create some dummy data
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    src_ids = torch.randint(1, config.src_vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(1, config.tgt_vocab_size, (batch_size, tgt_seq_len))
    tgt_ids[:, 0] = 1  # SOS token
    
    # Create padding masks (False = attend, True = mask)
    src_padding_mask = torch.zeros(batch_size, 1, 1, src_seq_len, dtype=torch.bool)
    tgt_padding_mask = torch.zeros(batch_size, 1, tgt_seq_len, tgt_seq_len, dtype=torch.bool)
    
    print("\nRunning forward pass with RoPE...")
    output = model(src_ids, tgt_ids, src_padding_mask, tgt_padding_mask)
    
    print(f"\nOutput shape: {output.shape}")
    print("Forward pass completed successfully!")


if __name__ == "__main__":
    main()