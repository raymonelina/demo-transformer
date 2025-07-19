# src/demo_transformer/inference_utils.py

import torch
import torch.nn.functional as F
from .transformer import Transformer
from typing import List, Tuple


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


def beam_search_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    src_padding_mask: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    max_output_len: int,
    beam_size: int = 5,
    device: torch.device = None,
) -> List[Tuple[List[int], float]]:
    """
    Performs beam search decoding to generate sequences from the Transformer.
    
    Args:
        model: The transformer model
        src_ids: Source token IDs [batch_size, src_seq_len]
        src_padding_mask: Source padding mask
        start_token_id: ID of the start token
        end_token_id: ID of the end token
        max_output_len: Maximum length of the output sequence
        beam_size: Beam size (number of hypotheses to maintain)
        device: Device to run on
        
    Returns:
        List of tuples (sequence, score) sorted by score in descending order
    """
    if device is None:
        device = src_ids.device
        
    model.eval()
    batch_size = src_ids.size(0)
    assert batch_size == 1, "Beam search only supports batch size 1"
    
    # Encode the source sequence
    with torch.no_grad():
        encoder_output = model.encoder(src_ids, src_padding_mask)
    
    # Initialize the beam
    # Each beam item is (sequence, score)
    beam = [(torch.tensor([[start_token_id]], device=device), 0.0)]
    finished_hypotheses = []
    
    # Beam search
    with torch.no_grad():
        for _ in range(max_output_len):
            if len(beam) == 0:
                break
                
            # Expand all current beam items
            all_candidates = []
            
            # For each hypothesis in the beam
            for seq, score in beam:
                # If the last token is EOS, add to finished hypotheses
                if seq[0, -1].item() == end_token_id:
                    finished_hypotheses.append((seq.squeeze(0).tolist(), score))
                    continue
                    
                # Get the next token probabilities
                logits = model.decoder(seq, encoder_output, src_padding_mask)
                next_token_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get top-k tokens and their probabilities
                topk_log_probs, topk_indices = log_probs.topk(beam_size)
                
                # Create new candidates
                for i in range(beam_size):
                    token_id = topk_indices[0, i].item()
                    log_prob = topk_log_probs[0, i].item()
                    
                    # Create new sequence by appending the token
                    new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)
                    
                    # Update score: average log probability
                    new_score = (score * (seq.size(1) - 1) + log_prob) / (seq.size(1))
                    
                    all_candidates.append((new_seq, new_score))
            
            # Select top-k candidates
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = all_candidates[:beam_size]
            
            # Check if all beam items are finished
            if all(seq[0, -1].item() == end_token_id for seq, _ in beam):
                finished_hypotheses.extend([(seq.squeeze(0).tolist(), score) for seq, score in beam])
                break
    
    # Add any unfinished hypotheses to the finished list
    for seq, score in beam:
        if seq[0, -1].item() != end_token_id:
            finished_hypotheses.append((seq.squeeze(0).tolist(), score))
    
    # Sort by score and return
    finished_hypotheses.sort(key=lambda x: x[1], reverse=True)
    
    # Remove start token from sequences
    return [(seq[1:], score) for seq, score in finished_hypotheses]
