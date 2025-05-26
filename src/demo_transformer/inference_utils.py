# src/demo_transformer/inference_utils.py

import torch
from .transformer import Transformer


def greedy_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    src_padding_mask: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    max_output_len: int,
    device: torch.device,
):
    """
    Performs greedy decoding to generate a sequence from the Transformer.
    """
    model.eval()

    with torch.no_grad():
        encoder_output = model.encoder(src_ids, src_padding_mask)

    generated_sequence = [start_token_id]

    with torch.no_grad():
        for _ in range(max_output_len):
            current_tgt_ids = (
                torch.tensor(generated_sequence, dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )

            logits = model.decoder(current_tgt_ids, encoder_output, src_padding_mask)

            next_token_logits = logits[:, -1, :]

            predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()

            generated_sequence.append(predicted_token_id)

            if predicted_token_id == end_token_id:
                break

    return generated_sequence[1:]  # Return sequence without SOS token
