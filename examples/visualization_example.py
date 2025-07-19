"""
Example script demonstrating the visualization capabilities of the transformer package.
"""

import torch
import matplotlib.pyplot as plt
from demo_transformer import Transformer, TransformerConfig

def main():
    # Create a configuration with store_attention enabled
    config = TransformerConfig(
        embed_dim=64,  # Using smaller dimensions for cleaner visualization
        num_heads=4,
        ff_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=16,
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        store_attention=True,  # Enable attention storage for visualization
    )
    
    print("Creating transformer model with attention visualization...")
    model = Transformer(config)
    
    # Create some dummy data
    batch_size = 1
    src_seq_len = 8
    tgt_seq_len = 6
    
    src_ids = torch.randint(0, config.src_vocab_size, (batch_size, src_seq_len))
    tgt_ids = torch.randint(0, config.tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # Create padding masks
    src_padding_mask = torch.zeros(batch_size, 1, 1, src_seq_len).bool()
    # Add some padding in the source sequence
    src_padding_mask[0, 0, 0, -2:] = True  # Pad last 2 positions in first batch
    
    # Sample token names for visualization
    src_tokens = [f"src_{i}" for i in range(src_seq_len)]
    tgt_tokens = [f"tgt_{i}" for i in range(tgt_seq_len)]
    
    print("\nRunning forward pass to generate attention weights...")
    
    # Run the model to generate attention weights
    with torch.no_grad():
        output = model(src_ids, tgt_ids, src_padding_mask)
    
    print("\nVisualizing attention weights...")
    
    # Visualize encoder self-attention
    print("Plotting encoder self-attention...")
    fig1 = model.visualize_encoder_attention(
        layer_idx=0, 
        head_idx=None,  # Average across heads
        tokens=src_tokens,
        figsize=(8, 6)
    )
    plt.savefig("encoder_self_attention.png")
    plt.close(fig1)
    
    # Visualize decoder self-attention
    print("Plotting decoder self-attention...")
    fig2 = model.visualize_decoder_self_attention(
        layer_idx=0,
        head_idx=None,  # Average across heads
        tokens=tgt_tokens,
        figsize=(8, 6)
    )
    plt.savefig("decoder_self_attention.png")
    plt.close(fig2)
    
    # Visualize decoder cross-attention
    print("Plotting decoder cross-attention...")
    fig3 = model.visualize_decoder_cross_attention(
        layer_idx=0,
        head_idx=None,  # Average across heads
        src_tokens=src_tokens,
        tgt_tokens=tgt_tokens,
        figsize=(8, 6)
    )
    plt.savefig("decoder_cross_attention.png")
    plt.close(fig3)
    
    # Visualize encoder embeddings
    print("Plotting encoder embeddings PCA...")
    fig4 = model.visualize_encoder_embeddings(
        input_ids=src_ids,
        tokens=src_tokens,
        figsize=(8, 6)
    )
    plt.savefig("encoder_embeddings_pca.png")
    plt.close(fig4)
    
    # Visualize all attention heads in a grid
    print("Plotting all attention heads...")
    fig5 = model.visualize_attention_heads(
        attention_type="encoder",
        layer_idx=0,
        tokens=src_tokens,
        figsize=(12, 10)
    )
    plt.savefig("encoder_attention_heads.png")
    plt.close(fig5)
    
    print("\nVisualization complete! Images saved to:")
    print("- encoder_self_attention.png")
    print("- decoder_self_attention.png")
    print("- decoder_cross_attention.png")
    print("- encoder_embeddings_pca.png")
    print("- encoder_attention_heads.png")

if __name__ == "__main__":
    main()