"""
Basic usage example for the transformer package.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from demo_transformer import (
    Transformer,
    TransformerConfig,
    TransformerTrainer,
    TransformerInference,
    get_transformer_scheduler,
)


def main():
    # Define model parameters using the configuration system
    config = TransformerConfig(
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_encoder_layers=3,
        num_decoder_layers=3,
        max_seq_len=50,
        src_vocab_size=1000,
        tgt_vocab_size=1200,
        dropout_rate=0.1,
        label_smoothing=0.1,
        weight_tying=False,  # Different vocab sizes, so no weight tying
        pre_norm=True,
    )
    
    # Print configuration
    print("Transformer Configuration:")
    for key, value in config.__dict__.items():
        print(f"  {key}: {value}")
    
    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    # Initialize the transformer model
    transformer_model = Transformer(config).to(device)
    print("Transformer model initialized.")
    
    # Create optimizer with learning rate scheduler
    optimizer = optim.Adam(transformer_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_transformer_scheduler(optimizer, config.embed_dim, warmup_steps=4000)
    
    # Create trainer
    trainer = TransformerTrainer(
        model=transformer_model,
        optimizer=optimizer,
        label_smoothing=config.label_smoothing,
        pad_token_id=config.pad_token_id,
        device=device,
    )
    
    # Create inference module
    inference = TransformerInference(
        model=transformer_model,
        start_token_id=config.sos_token_id,
        end_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        device=device,
    )
    
    # --- Simulate Inputs ---
    batch_size = 2
    source_seq_len = 15
    target_seq_len = 10
    
    # Create dummy source data
    dummy_src_ids = torch.randint(0, config.src_vocab_size, (batch_size, source_seq_len)).to(device)
    dummy_src_padding_mask = torch.zeros(batch_size, 1, 1, source_seq_len, dtype=torch.bool).to(device)
    dummy_src_padding_mask[1, :, :, 10:] = True  # Add padding to second sequence
    
    # Create dummy target data with padding mask
    dummy_tgt_ids = torch.randint(0, config.tgt_vocab_size, (batch_size, target_seq_len)).to(device)
    dummy_tgt_ids[:, 0] = config.sos_token_id  # Start with SOS token
    dummy_tgt_ids[0, 5:] = config.pad_token_id  # Add padding
    dummy_tgt_ids[1, 7:] = config.pad_token_id  # Add padding
    
    # Create target padding mask
    dummy_tgt_padding_mask = torch.zeros(batch_size, 1, target_seq_len, target_seq_len, dtype=torch.bool).to(device)
    # Mask padding positions for first sequence (positions 5-9)
    dummy_tgt_padding_mask[0, :, 5:, :] = True  # Mask rows (queries)
    dummy_tgt_padding_mask[0, :, :, 5:] = True  # Mask columns (keys)
    # Mask padding positions for second sequence (positions 7-9)
    dummy_tgt_padding_mask[1, :, 7:, :] = True  # Mask rows (queries)
    dummy_tgt_padding_mask[1, :, :, 7:] = True  # Mask columns (keys)
    
    print(f"Dummy Source IDs shape: {dummy_src_ids.shape}")
    print(f"Dummy Target IDs shape: {dummy_tgt_ids.shape}")
    
    # --- Simulate Training ---
    print("\n--- Simulating a training step ---")
    transformer_model.train()
    
    # Create a batch dictionary
    batch = {
        "src_ids": dummy_src_ids,
        "tgt_ids": dummy_tgt_ids,
        "src_padding_mask": dummy_src_padding_mask,
        "tgt_padding_mask": dummy_tgt_padding_mask,
    }
    
    # Perform a training step with both masks
    metrics = trainer.train_step(
        dummy_src_ids, dummy_tgt_ids, 
        src_padding_mask=dummy_src_padding_mask,
        tgt_padding_mask=dummy_tgt_padding_mask
    )
    print(f"Training loss: {metrics['loss']:.4f}")
    
    # Update learning rate
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Current learning rate: {current_lr:.8f}")
    
    # --- Simulate Inference ---
    print("\n--- Simulating Inference ---")
    transformer_model.eval()
    
    # Take just the first sequence for inference
    single_src_ids = dummy_src_ids[0:1]
    single_src_padding_mask = dummy_src_padding_mask[0:1]
    
    # Greedy decoding
    print("\n1. Greedy Decoding:")
    greedy_output = inference.greedy_decode(
        single_src_ids,
        single_src_padding_mask,
        max_output_len=20,
    )
    print(f"Greedy output: {greedy_output}")
    
    # Beam search
    print("\n2. Beam Search Decoding:")
    beam_size = 3
    beam_outputs = inference.beam_search_decode(
        single_src_ids,
        single_src_padding_mask,
        max_output_len=20,
        beam_size=beam_size,
    )
    
    print(f"Top {beam_size} beam search results:")
    for i, (sequence, score) in enumerate(beam_outputs[:beam_size]):
        print(f"  Beam {i+1}: Score={score:.4f}, Sequence={sequence}")
    
    # Sampling
    print("\n3. Sampling Decoding:")
    sampled_output = inference.sample_decode(
        single_src_ids,
        single_src_padding_mask,
        max_output_len=20,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )
    print(f"Sampled output: {sampled_output}")
    
    # Save the model
    print("\n--- Saving and Loading Model ---")
    model_path = "transformer_model.pt"
    transformer_model.save_pretrained(model_path)
    
    # Load the model
    loaded_model = Transformer.from_pretrained(model_path)
    print("Model loaded successfully")
    
    print("\nSimulation completed.")


if __name__ == "__main__":
    main()