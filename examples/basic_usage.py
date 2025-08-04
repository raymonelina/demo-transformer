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
        "cuda"
        if torch.cuda.is_available()
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

    # Define actual sequence lengths for each sequence in batch
    src_lengths = [12, 8]  # Source sequences: lengths 12 and 8
    tgt_lengths = [6, 9]  # Target sequences: lengths 6 and 9

    # Padded lengths (max in batch)
    source_seq_len = max(src_lengths)
    target_seq_len = max(tgt_lengths)

    # Create source sequences and masks (NO SOS/EOS for encoder!)
    dummy_src_ids = torch.randint(1, config.src_vocab_size, (batch_size, source_seq_len)).to(device)
    dummy_src_padding_mask = torch.zeros(batch_size, 1, 1, source_seq_len, dtype=torch.bool).to(
        device
    )

    for i, length in enumerate(src_lengths):
        dummy_src_ids[i, length:] = config.pad_token_id  # Add padding
        dummy_src_padding_mask[i, :, :, length:] = True  # Mask padding

    # Create target sequences and masks (WITH SOS/EOS for decoder!)
    dummy_tgt_ids = torch.randint(1, config.tgt_vocab_size, (batch_size, target_seq_len)).to(device)
    dummy_tgt_ids[:, 0] = config.sos_token_id  # Start with SOS token
    dummy_tgt_padding_mask = torch.zeros(
        batch_size, 1, target_seq_len, target_seq_len, dtype=torch.bool
    ).to(device)

    for i, length in enumerate(tgt_lengths):
        dummy_tgt_ids[i, length:] = config.pad_token_id  # Add padding
        dummy_tgt_padding_mask[i, :, length:, :] = True  # Mask rows (queries from padding)
        dummy_tgt_padding_mask[i, :, :, length:] = True  # Mask columns (keys to padding)

    # Example of resulting token arrays:
    # dummy_src_ids[0]: [45, 123, 67, 89, 234, 12, 456, 78, 90, 345, 23, 567] (length 12)
    # dummy_src_ids[1]: [12, 34, 56, 78, 90, 123, 45, 67, PAD, PAD, PAD, PAD] (length 8, padded)
    # dummy_tgt_ids[0]: [SOS, 234, 567, 89, 123, 45, PAD, PAD, PAD] (length 6, padded)
    # dummy_tgt_ids[1]: [SOS, 45, 678, 90, 234, 567, 89, 12, 345] (length 9)

    print(f"Source lengths: {src_lengths} -> padded to {source_seq_len}")
    print(f"Target lengths: {tgt_lengths} -> padded to {target_seq_len}")
    print(f"Dummy Source IDs shape: {dummy_src_ids.shape}")
    print(f"Dummy Target IDs shape: {dummy_tgt_ids.shape}")
    print(f"\nActual Source IDs:")
    print(f"  Sequence 0: {dummy_src_ids[0].tolist()}")
    print(f"  Sequence 1: {dummy_src_ids[1].tolist()}")
    print(f"\nActual Target IDs:")
    print(f"  Sequence 0: {dummy_tgt_ids[0].tolist()}")
    print(f"  Sequence 1: {dummy_tgt_ids[1].tolist()}")
    print(f"\nPAD token ID: {config.pad_token_id}, SOS token ID: {config.sos_token_id}")
    print(f"\nSource Padding Masks (T=masked, F=attend):")
    print(f"  Sequence 0: {['T' if x else 'F' for x in dummy_src_padding_mask[0, 0, 0].tolist()]}")
    print(f"  Sequence 1: {['T' if x else 'F' for x in dummy_src_padding_mask[1, 0, 0].tolist()]}")
    print(f"\nTarget Padding Masks (T=masked, F=attend, showing diagonal):")
    print(
        f"  Sequence 0 diagonal: {['T' if x else 'F' for x in torch.diag(dummy_tgt_padding_mask[0, 0]).tolist()]}"
    )
    print(
        f"  Sequence 1 diagonal: {['T' if x else 'F' for x in torch.diag(dummy_tgt_padding_mask[1, 0]).tolist()]}"
    )

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

    # Check for potential issues before training
    print(
        f"Model parameters finite: {all(torch.isfinite(p).all() for p in transformer_model.parameters())}"
    )
    print(
        f"Input ranges - src: [{dummy_src_ids.min()}, {dummy_src_ids.max()}], tgt: [{dummy_tgt_ids.min()}, {dummy_tgt_ids.max()}]"
    )

    # Perform a training step with both masks
    metrics = trainer.train_step(
        dummy_src_ids,
        dummy_tgt_ids,
        src_padding_mask=dummy_src_padding_mask,
        tgt_padding_mask=dummy_tgt_padding_mask,
    )
    print(f"Training loss: {metrics['loss']:.4f}")

    # Check model parameters after training step
    print(
        f"Model parameters finite after step: {all(torch.isfinite(p).all() for p in transformer_model.parameters())}"
    )

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

    # Note: Since this is an untrained model, inference may produce NaN/inf values
    print(
        "\nNote: Inference with untrained model may produce unstable results due to random weights"
    )

    try:
        # Greedy decoding
        print("\n1. Greedy Decoding:")
        greedy_output = inference.greedy_decode(
            single_src_ids,
            single_src_padding_mask,
            max_output_len=20,
        )
        print(f"Greedy output: {greedy_output}")
    except RuntimeError as e:
        print(f"Greedy decoding failed: {e}")

    try:
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
    except RuntimeError as e:
        print(f"Beam search failed: {e}")

    try:
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
    except RuntimeError as e:
        print(f"Sampling failed: {e}")

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
