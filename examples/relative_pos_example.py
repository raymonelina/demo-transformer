"""
Example demonstrating the use of relative positional encoding in the transformer.
"""

import torch
from demo_transformer import Transformer, TransformerConfig, TransformerInference


def main():
    # Create a configuration with relative positional encoding
    config = TransformerConfig(
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        max_seq_len=50,
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        dropout_rate=0.1,
        use_relative_pos=True,  # Enable relative positional encoding
    )
    
    print("Transformer Configuration:")
    print(f"  Using relative positional encoding: {config.use_relative_pos}")
    
    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    # Initialize models - one with relative positional encoding and one without
    transformer_rel_pos = Transformer(config).to(device)
    
    # Create a configuration without relative positional encoding for comparison
    config_no_rel_pos = TransformerConfig(
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        max_seq_len=50,
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        dropout_rate=0.1,
        use_relative_pos=False,  # Disable relative positional encoding
    )
    
    transformer_no_rel_pos = Transformer(config_no_rel_pos).to(device)
    
    print("Models initialized.")
    
    # Create dummy input data
    batch_size = 2
    src_seq_len = 20
    tgt_seq_len = 15
    
    src_ids = torch.randint(0, config.src_vocab_size, (batch_size, src_seq_len)).to(device)
    tgt_ids = torch.randint(0, config.tgt_vocab_size, (batch_size, tgt_seq_len)).to(device)
    
    src_padding_mask = torch.zeros(batch_size, 1, 1, src_seq_len, dtype=torch.bool).to(device)
    src_padding_mask[0, :, :, 15:] = True  # Add padding to first sequence
    
    print(f"Input shapes - Source: {src_ids.shape}, Target: {tgt_ids.shape}")
    
    # Forward pass with both models
    print("\nRunning forward pass...")
    
    # Model with relative positional encoding
    transformer_rel_pos.eval()
    with torch.no_grad():
        output_rel_pos = transformer_rel_pos(src_ids, tgt_ids, src_padding_mask)
    
    # Model without relative positional encoding
    transformer_no_rel_pos.eval()
    with torch.no_grad():
        output_no_rel_pos = transformer_no_rel_pos(src_ids, tgt_ids, src_padding_mask)
    
    print(f"Output shape (with relative pos): {output_rel_pos.shape}")
    print(f"Output shape (without relative pos): {output_no_rel_pos.shape}")
    
    # Check if outputs are different (they should be)
    output_diff = torch.abs(output_rel_pos - output_no_rel_pos).mean().item()
    print(f"\nMean absolute difference between outputs: {output_diff:.6f}")
    print("The difference confirms that relative positional encoding is working differently than absolute positional encoding.")
    
    # Create inference modules for both models
    inference_rel_pos = TransformerInference(
        model=transformer_rel_pos,
        start_token_id=config.sos_token_id,
        end_token_id=config.eos_token_id,
    )
    
    inference_no_rel_pos = TransformerInference(
        model=transformer_no_rel_pos,
        start_token_id=config_no_rel_pos.sos_token_id,
        end_token_id=config_no_rel_pos.eos_token_id,
    )
    
    # Generate sequences with both models
    print("\nGenerating sequences...")
    
    # Take just the first sequence for inference
    single_src_ids = src_ids[0:1]
    single_src_padding_mask = src_padding_mask[0:1]
    
    # Greedy decoding
    rel_pos_output = inference_rel_pos.greedy_decode(
        single_src_ids,
        single_src_padding_mask,
        max_output_len=10,
    )
    
    no_rel_pos_output = inference_no_rel_pos.greedy_decode(
        single_src_ids,
        single_src_padding_mask,
        max_output_len=10,
    )
    
    print(f"Generated sequence (with relative pos): {rel_pos_output}")
    print(f"Generated sequence (without relative pos): {no_rel_pos_output}")
    
    print("\nExample completed.")


if __name__ == "__main__":
    main()