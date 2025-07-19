"""
Example script demonstrating the debug printing functionality.
"""

import torch
from demo_transformer import Transformer, TransformerConfig

def main():
    # Create a configuration with debug mode enabled
    config = TransformerConfig(
        embed_dim=64,  # Using smaller dimensions for cleaner debug output
        num_heads=2,
        ff_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=16,
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        debug_mode=True,  # Enable debug printing
    )
    
    print("Creating transformer model with debug mode enabled...")
    model = Transformer(config)
    
    # Create some dummy data
    batch_size = 2
    src_seq_len = 8
    tgt_seq_len = 6
    
    src_ids = torch.randint(0, config.src_vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(0, config.tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Create padding masks
    src_padding_mask = torch.zeros(batch_size, 1, 1, src_seq_len).bool()
    # Add some padding in the source sequence
    src_padding_mask[0, 0, 0, -2:] = True  # Pad last 2 positions in first batch
    
    print("\n" + "="*80)
    print("Running forward pass with debug printing enabled...")
    print("="*80 + "\n")
    
    # Run the model with debug printing
    with torch.no_grad():
        output = model(src_ids, tgt_ids, src_padding_mask)
    
    print("\n" + "="*80)
    print("Forward pass completed!")
    print("="*80)
    
    # Show output shape
    print(f"\nOutput shape: {output.shape}")
    
    print("\nTo disable debug printing, set debug_mode=False in the TransformerConfig.")

if __name__ == "__main__":
    main()