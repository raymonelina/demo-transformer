"""Inference utilities for Transformer sequence generation.

Implements three main decoding strategies from academic literature:
1. Greedy Decoding - Deterministic, always picks most likely token
2. Beam Search - Maintains multiple hypotheses (Bahdanau et al., 2014)
3. Sampling - Stochastic with temperature/top-k/top-p (Holtzman et al., 2019)
"""

import torch
import torch.nn.functional as F
from .transformer import Transformer
from typing import List, Tuple, Optional


def greedy_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    src_padding_mask: Optional[torch.Tensor],
    start_token_id: int,
    end_token_id: int,
    max_output_len: int,
    device: torch.device,
):
    """Greedy Decoding: y_t = argmax P(y_t | y_<t, x)

    Deterministic generation - always picks most probable token.
    Fast but can produce repetitive/suboptimal sequences.
    Used as baseline in "Attention Is All You Need" (Vaswani et al., 2017).
    """
    model.eval()

    with torch.no_grad():
        encoder_output = model.encoder(src_ids, src_padding_mask)

    generated_sequence = [start_token_id]

    with torch.no_grad():
        for _ in range(max_output_len):
            current_tgt_ids = (
                torch.tensor(generated_sequence, dtype=torch.long).unsqueeze(0).to(device)
            )

            # Get decoder output for current sequence
            logits = model.decoder(current_tgt_ids, encoder_output, src_padding_mask)
            next_token_logits = logits[:, -1, :]  # Last position predictions
            # Greedy choice: argmax over vocabulary
            predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()

            generated_sequence.append(predicted_token_id)

            if predicted_token_id == end_token_id:
                break

    return generated_sequence[1:]  # Remove SOS token


def beam_search_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    src_padding_mask: Optional[torch.Tensor],
    start_token_id: int,
    end_token_id: int,
    max_output_len: int,
    beam_size: int = 5,
    device: torch.device = None,
) -> List[Tuple[List[int], float]]:
    """Beam Search: Maintains top-k hypotheses to find high-probability sequences.

    Classic search algorithm adapted for neural text generation. Explores multiple
    paths simultaneously to find higher-quality sequences than greedy decoding.

    Uses running average for length normalization: prevents bias toward shorter sequences.

    Returns: List of (sequence, score) tuples sorted by score.
    """
    if device is None:
        device = src_ids.device

    model.eval()

    with torch.no_grad():
        encoder_output = model.encoder(src_ids, src_padding_mask)

    # Initialize beam with start token, score=0 (log prob)
    beam = [(torch.tensor([[start_token_id]], device=device), 0.0)]
    finished_hypotheses = []

    with torch.no_grad():
        for step in range(max_output_len):
            if len(beam) == 0:
                break

            all_candidates = []

            for seq, score in beam:
                # Move finished sequences (ending with EOS) to results
                if seq[0, -1].item() == end_token_id:
                    finished_hypotheses.append((seq.squeeze(0).tolist(), score))
                    continue

                # Get next token probabilities for this hypothesis
                logits = model.decoder(seq, encoder_output, src_padding_mask)
                next_token_logits = logits[:, -1, :]
                log_probs = F.log_softmax(next_token_logits, dim=-1)

                # Get top-k most likely next tokens
                topk_log_probs, topk_indices = log_probs.topk(beam_size)

                for i in range(beam_size):
                    token_id = topk_indices[0, i].item()
                    log_prob = topk_log_probs[0, i].item()

                    new_seq = torch.cat([seq, torch.tensor([[token_id]], device=device)], dim=1)

                    # Length normalization: running average of log probabilities
                    # Prevents bias against longer sequences
                    seq_len = new_seq.size(1) - 1  # Exclude SOS token
                    new_score = (
                        (score * (seq_len - 1) + log_prob) / seq_len if seq_len > 0 else log_prob
                    )

                    all_candidates.append((new_seq, new_score))

            # Keep top beam_size candidates by score
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beam = all_candidates[:beam_size]

            # Continue until max_output_len or all beams finished
            if all(seq[0, -1].item() == end_token_id for seq, _ in beam):
                break

    # Add remaining beam items (handle case where max_output_len reached)
    for seq, score in beam:
        finished_hypotheses.append((seq.squeeze(0).tolist(), score))

    finished_hypotheses.sort(key=lambda x: x[1], reverse=True)
    return [(seq[1:], score) for seq, score in finished_hypotheses[:beam_size]]


def sample_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    src_padding_mask: Optional[torch.Tensor],
    start_token_id: int,
    end_token_id: int,
    max_output_len: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device = None,
) -> List[int]:
    """Sampling: y_t ~ P(y_t | y_<t, x) with filtering techniques.

    Combines three methods:
    - Temperature: P(y) = softmax(logits/T) - controls randomness
    - Top-k: Keep only k most likely tokens (Fan et al., 2018)
    - Top-p (Nucleus): Remove tokens where cumulative prob > p (Holtzman et al., 2019)

    More diverse than greedy, avoids repetition.
    """
    if device is None:
        device = src_ids.device

    model.eval()

    with torch.no_grad():
        encoder_output = model.encoder(src_ids, src_padding_mask)

    generated_sequence = [start_token_id]

    with torch.no_grad():
        for _ in range(max_output_len):
            current_tgt_ids = torch.tensor([generated_sequence], dtype=torch.long, device=device)

            logits = model.decoder(current_tgt_ids, encoder_output, src_padding_mask)
            # Temperature scaling: higher T = more random, lower T = more focused
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-7)

            # Top-k filtering: keep only k most likely tokens
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits, min(top_k, next_token_logits.size(-1))
                )
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)

            # Top-p (Nucleus) filtering: dynamic vocabulary based on cumulative probability
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens where cumulative prob > top_p
                # But always keep at least the most likely token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                # Map back to original token order and mask
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample from filtered distribution using multinomial sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            generated_sequence.append(next_token_id)

            if next_token_id == end_token_id:
                break

    return generated_sequence[1:]


class TransformerInference:
    """Unified interface for Transformer inference with multiple decoding strategies.

    Provides clean API for greedy, beam search, and sampling decoding.
    Handles device management and token configuration automatically.
    """

    def __init__(self, model: Transformer, start_token_id: int = 1, end_token_id: int = 2):
        self.model = model
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.device = next(model.parameters()).device

    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor],
        max_output_len: int = 100,
    ):
        return greedy_decode(
            self.model,
            src_ids,
            src_padding_mask,
            self.start_token_id,
            self.end_token_id,
            max_output_len,
            self.device,
        )

    def beam_search_decode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor],
        max_output_len: int = 100,
        beam_size: int = 5,
    ):
        return beam_search_decode(
            self.model,
            src_ids,
            src_padding_mask,
            self.start_token_id,
            self.end_token_id,
            max_output_len,
            beam_size,
            self.device,
        )

    def sample_decode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor],
        max_output_len: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ):
        return sample_decode(
            self.model,
            src_ids,
            src_padding_mask,
            self.start_token_id,
            self.end_token_id,
            max_output_len,
            temperature,
            top_k,
            top_p,
            self.device,
        )
