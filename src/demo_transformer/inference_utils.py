# src/demo_transformer/inference_utils.py
"""
Inference utilities for Transformer models with multiple decoding strategies.

This module implements three main decoding approaches for sequence generation:
1. Greedy Decoding - Deterministic, always picks most likely token
2. Beam Search - Maintains multiple hypotheses, finds high-probability sequences
3. Sampling - Stochastic generation with temperature, top-k, and top-p filtering

Academic References:
- "Attention Is All You Need" (Vaswani et al., 2017) - Original Transformer
- "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019) - Top-p sampling
- "Hierarchical Neural Story Generation" (Fan et al., 2018) - Top-k sampling
- "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2014) - Beam search
"""

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
    Greedy Decoding: Deterministic sequence generation strategy.
    
    At each time step, selects the token with the highest probability:
    y_t = argmax P(y_t | y_<t, x)
    
    Academic Background:
    - Most basic decoding strategy, widely used in early neural MT systems
    - Mentioned in "Attention Is All You Need" (Vaswani et al., 2017)
    - Simple but can lead to repetitive or suboptimal sequences
    
    Characteristics:
    - Deterministic: same input always produces same output
    - Fast: O(T) time complexity where T is sequence length
    - Local optimality: optimal choice at each step, but not globally optimal
    - Can suffer from "exposure bias" - training uses teacher forcing but inference is autoregressive
    
    Use Cases:
    - When deterministic output is required
    - Fast inference scenarios
    - Baseline for comparing other decoding methods
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

            # Greedy selection: always pick the most probable token
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
    Beam Search Decoding: Approximate search for high-probability sequences.
    
    Maintains top-k hypotheses (beam) at each step, expanding each hypothesis
    and keeping the k best candidates based on cumulative log probability:
    
    Score(y_1...y_t) = (1/t) * Σ log P(y_i | y_<i, x)
    
    Academic Background:
    - "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2014)
    - "Sequence to Sequence Learning with Neural Networks" (Sutskever et al., 2014)
    - Standard in neural machine translation and text generation
    
    Algorithm:
    1. Start with beam containing only start token
    2. For each step:
       - Expand each hypothesis in beam with all possible next tokens
       - Score all candidates using average log probability
       - Keep top-k candidates as new beam
    3. Return finished hypotheses sorted by score
    
    Characteristics:
    - Better than greedy: explores multiple paths simultaneously
    - Approximate: not guaranteed to find globally optimal sequence
    - Time complexity: O(T * K * V) where T=length, K=beam_size, V=vocab_size
    - Length bias: longer sequences have lower scores (addressed by length normalization)
    
    Use Cases:
    - Machine translation (most common)
    - Text summarization
    - When quality is more important than speed
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
                    
                    # Update score: average log probability (length normalization)
                    # This prevents bias against longer sequences
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


def sample_decode(
    model: Transformer,
    src_ids: torch.Tensor,
    src_padding_mask: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
    max_output_len: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    device: torch.device = None,
) -> List[int]:
    """
    Sampling Decoding: Stochastic generation with advanced filtering techniques.
    
    Samples from the probability distribution instead of taking the argmax:
    y_t ~ P(y_t | y_<t, x)
    
    Combines three techniques:
    1. Temperature Scaling: Controls randomness by scaling logits
    2. Top-k Sampling: Restricts sampling to k most likely tokens
    3. Top-p (Nucleus) Sampling: Dynamic vocabulary based on cumulative probability
    
    Academic References:
    
    Temperature Sampling:
    - Concept from statistical mechanics, applied to neural language models
    - P(y_t) = softmax(logits / temperature)
    - temperature = 1.0: original distribution
    - temperature > 1.0: more uniform (creative)
    - temperature < 1.0: more peaked (conservative)
    
    Top-k Sampling:
    - "Hierarchical Neural Story Generation" (Fan et al., 2018)
    - Keep only k most probable tokens, redistribute probability mass
    - Fixed vocabulary size regardless of probability distribution shape
    - Typical values: k = 40-50
    
    Top-p (Nucleus) Sampling:
    - "The Curious Case of Neural Text Degeneration" (Holtzman et al., 2019)
    - https://arxiv.org/abs/1904.09751
    - Dynamic vocabulary: keep tokens until cumulative probability ≥ p
    - Adapts to probability distribution shape (peaked vs flat)
    - Typical values: p = 0.9-0.95
    - Superior to top-k for avoiding repetition and maintaining coherence
    
    Algorithm:
    1. Apply temperature scaling to logits
    2. Apply top-k filtering (if k > 0)
    3. Apply top-p filtering (if p < 1.0)
    4. Sample from filtered distribution using multinomial sampling
    
    Characteristics:
    - Stochastic: different outputs for same input
    - More diverse and creative than deterministic methods
    - Can avoid repetition and generic responses
    - Quality depends heavily on hyperparameter tuning
    
    Use Cases:
    - Creative text generation (stories, poetry)
    - Dialogue systems
    - When diversity is more important than accuracy
    """
    if device is None:
        device = src_ids.device
        
    model.eval()
    batch_size = src_ids.size(0)
    assert batch_size == 1, "Sampling only supports batch size 1"
    
    # Encode the source sequence
    with torch.no_grad():
        encoder_output = model.encoder(src_ids, src_padding_mask)
    
    generated_sequence = [start_token_id]
    
    with torch.no_grad():
        for _ in range(max_output_len):
            current_tgt_ids = torch.tensor([generated_sequence], device=device)
            
            # Get logits for next token
            logits = model.decoder(current_tgt_ids, encoder_output, src_padding_mask)
            # Apply temperature scaling: higher temp = more random, lower temp = more focused
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering (Fan et al., 2018)
            # Keep only the k most likely tokens, set others to -inf
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering (Holtzman et al., 2019)
            # Keep tokens with cumulative probability <= top_p
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least the first token (highest probability)
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Convert back to original token order and mask
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution using multinomial sampling
            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            generated_sequence.append(next_token_id)
            
            if next_token_id == end_token_id:
                break
    
    return generated_sequence[1:]  # Return sequence without start token


class TransformerInference:
    """
    Unified inference interface for Transformer models with multiple decoding strategies.
    
    This class provides a clean API for all three main decoding approaches:
    - Greedy decoding for fast, deterministic generation
    - Beam search for high-quality, structured generation
    - Sampling for diverse, creative generation
    
    The class handles device management, token ID configuration, and provides
    consistent interfaces across all decoding methods.
    
    Example Usage:
        config = TransformerConfig(...)
        model = Transformer(config)
        inference = TransformerInference(model)
        
        # Greedy decoding
        output = inference.greedy_decode(src_ids, src_mask, max_len=50)
        
        # Beam search
        beams = inference.beam_search_decode(src_ids, src_mask, beam_size=5)
        
        # Creative sampling
        output = inference.sample_decode(
            src_ids, src_mask, temperature=0.8, top_p=0.9
        )
    """
    
    def __init__(
        self,
        model: Transformer,
        start_token_id: int = 1,
        end_token_id: int = 2,
        pad_token_id: int = 0,
        device: torch.device = None,
    ):
        """
        Initialize the inference wrapper.
        
        Args:
            model: Transformer model
            start_token_id: Start-of-sequence token ID
            end_token_id: End-of-sequence token ID
            pad_token_id: Padding token ID
            device: Device to use for inference
        """
        self.model = model
        self.start_token_id = start_token_id
        self.end_token_id = end_token_id
        self.pad_token_id = pad_token_id
        
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device
            
        self.model.to(self.device)
    
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: torch.Tensor,
        max_output_len: int = 100,
    ) -> List[int]:
        """
        Greedy decoding: always pick the most likely next token.
        
        Fast, deterministic generation suitable for tasks requiring consistency.
        """
        return greedy_decode(
            self.model, src_ids, src_padding_mask,
            self.start_token_id, self.end_token_id,
            max_output_len, self.device
        )
    
    def beam_search_decode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: torch.Tensor,
        max_output_len: int = 100,
        beam_size: int = 5,
    ) -> List[Tuple[List[int], float]]:
        """
        Beam search decoding: maintain top-k hypotheses.
        
        Higher quality generation by exploring multiple paths simultaneously.
        Returns multiple candidates ranked by score.
        """
        return beam_search_decode(
            self.model, src_ids, src_padding_mask,
            self.start_token_id, self.end_token_id,
            max_output_len, beam_size, self.device
        )
    
    def sample_decode(
        self,
        src_ids: torch.Tensor,
        src_padding_mask: torch.Tensor,
        max_output_len: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> List[int]:
        """
        Sampling decoding with temperature, top-k, and top-p (nucleus) sampling.
        
        Stochastic generation for diverse, creative outputs. Combines multiple
        filtering techniques for controlled randomness.
        
        Args:
            temperature: Controls randomness (higher = more random)
            top_k: Keep only top k tokens (0 = no filtering)
            top_p: Keep tokens with cumulative probability <= top_p (1.0 = no filtering)
        """
        return sample_decode(
            self.model, src_ids, src_padding_mask,
            self.start_token_id, self.end_token_id,
            max_output_len, temperature, top_k, top_p, self.device
        )
