# demo.py

import torch
import torch.nn as nn
from demo_transformer.transformer import Transformer
from demo_transformer.inference_utils import greedy_decode, beam_search_decode

if __name__ == "__main__":
    # Define model parameters
    embed_dim = 256
    num_heads = 8
    ff_dim = 512
    num_encoder_layers = 3
    num_decoder_layers = 3
    max_seq_len = 50
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    dropout_rate = 0.1

    # Determine device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # --- 1. Initialize Full Transformer Model ---
    transformer_model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_seq_len=max_seq_len,
        dropout_rate=dropout_rate,
    ).to(device)
    print("Full Transformer (Encoder-Decoder) initialized.")

    # --- 2. Simulate Inputs ---
    batch_size = 2

    source_seq_len = 15
    dummy_src_ids = torch.randint(0, src_vocab_size, (batch_size, source_seq_len)).to(
        device
    )
    dummy_src_padding_mask = torch.zeros(
        batch_size, 1, 1, source_seq_len, dtype=torch.bool
    ).to(device)
    dummy_src_padding_mask[1, :, :, 10:] = True
    print(f"\nDummy Source IDs shape: {dummy_src_ids.shape}")
    print(f"Dummy Source Padding Mask shape: {dummy_src_padding_mask.shape}")

    target_seq_len = 10
    dummy_tgt_ids = torch.randint(0, tgt_vocab_size, (batch_size, target_seq_len)).to(
        device
    )
    SOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    PAD_TOKEN_ID = 0
    dummy_tgt_ids[:, 0] = SOS_TOKEN_ID
    dummy_tgt_ids[0, 5:] = PAD_TOKEN_ID
    dummy_tgt_ids[1, 7:] = PAD_TOKEN_ID
    print(f"Dummy Target IDs shape: {dummy_tgt_ids.shape}")

    # --- 3. Forward Pass (Conceptual Training Step) ---
    print("\n--- Simulating a training forward pass ---")
    transformer_model.train()
    decoder_logits = transformer_model(
        dummy_src_ids, dummy_tgt_ids, src_padding_mask=dummy_src_padding_mask
    )
    print(f"Decoder Logits shape: {decoder_logits.shape}")

    expected_decoder_logits_shape = (batch_size, target_seq_len, tgt_vocab_size)
    assert (
        decoder_logits.shape == expected_decoder_logits_shape
    ), "Decoder logits shape mismatch!"
    print("Full Transformer forward pass successful (simulated training step).")

    # --- 4. Simulate Greedy Inference (Decoding) ---
    print("\n--- Simulating Greedy Inference (token-by-token generation) ---")

    single_src_ids = dummy_src_ids[0:1]
    single_src_padding_mask = dummy_src_padding_mask[0:1]

    MAX_GENERATION_LEN = 20

    print(f"Source IDs for inference: {single_src_ids.cpu().numpy()}")
    print(f"Starting decoding with SOS token (ID: {SOS_TOKEN_ID})...")

    generated_ids = greedy_decode(
        transformer_model,
        single_src_ids,
        single_src_padding_mask,
        start_token_id=SOS_TOKEN_ID,
        end_token_id=EOS_TOKEN_ID,
        max_output_len=MAX_GENERATION_LEN,
        device=device,
    )

    print(f"\nFinal generated sequence (IDs, excluding SOS): {generated_ids}")
    print(f"Generated sequence length: {len(generated_ids)}")
    print("Greedy inference simulation completed.")
    
    # --- 5. Simulate Beam Search Inference ---
    print("\n--- Simulating Beam Search Inference ---")
    
    BEAM_SIZE = 3
    print(f"Using beam size: {BEAM_SIZE}")
    
    beam_results = beam_search_decode(
        transformer_model,
        single_src_ids,
        single_src_padding_mask,
        start_token_id=SOS_TOKEN_ID,
        end_token_id=EOS_TOKEN_ID,
        max_output_len=MAX_GENERATION_LEN,
        beam_size=BEAM_SIZE,
        device=device,
    )
    
    print("\nBeam search results (top sequences and their scores):")
    for i, (sequence, score) in enumerate(beam_results[:BEAM_SIZE]):
        print(f"  Beam {i+1}: Score={score:.4f}, Sequence={sequence}")
    
    print("\nComparing top beam search result with greedy search:")
    if beam_results:
        top_beam_sequence = beam_results[0][0]
        print(f"  Greedy: {generated_ids}")
        print(f"  Beam:   {top_beam_sequence}")
        
        if generated_ids == top_beam_sequence:
            print("  Results match! (This can happen with random initialization)")
        else:
            print("  Results differ (beam search typically finds better sequences)")
    
    print("Inference simulations completed.")
