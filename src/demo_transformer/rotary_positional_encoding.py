"""
Rotary Positional Encoding (RoPE) implementation for transformer models.

Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class RotaryPositionalEncoding(nn.Module):
    """
    Implements Rotary Position Embedding (RoPE) for transformer models.
    
    RoPE performs rotation in the complex space to encode relative positions,
    which helps the model generalize better to sequences of different lengths.
    It directly modifies the query and key matrices in the attention mechanism
    rather than being added to the input embeddings.
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 512):
        """
        Initialize rotary positional encoding.

        Args:
            embed_dim: Dimension of embeddings (must be divisible by 2)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        if embed_dim % 2 != 0:
            raise ValueError(f"Embedding dimension {embed_dim} must be divisible by 2 for RoPE")
        
        # Generate the frequency bands
        self.freqs_cis = self._precompute_freqs_cis(embed_dim, max_seq_len)
    
    def _precompute_freqs_cis(self, dim: int, max_seq_len: int) -> torch.Tensor:
        """
        Precompute the frequency tensor for complex rotation.
        
        Args:
            dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
            
        Returns:
            Complex tensor with precomputed frequencies
        """
        # Compute the frequencies for each dimension
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # Create position indices
        positions = torch.arange(max_seq_len).float()
        
        # Outer product of positions and frequencies
        freqs = torch.outer(positions, freqs)
        
        # Convert to complex representation: [max_seq_len, dim/2]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        
        # Register as buffer (not a parameter)
        self.register_buffer("_freqs_cis", freqs_cis)
        
        return freqs_cis
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotate half of the dimensions of the input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            Rotated tensor with same shape
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(
        self, q: torch.Tensor, k: torch.Tensor, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key tensors.
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len, num_heads, head_dim]
            seq_len: Sequence length
            
        Returns:
            Tuple of (rotated_q, rotated_k) with same shapes as inputs
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )
        
        # Get the relevant frequencies for the current sequence length
        freqs_cis = self._freqs_cis[:seq_len]
        
        # Reshape for broadcasting
        # [seq_len, dim/2] -> [1, seq_len, 1, dim/2]
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        
        # Convert to complex representation
        q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        
        # Apply complex rotation
        q_out = torch.view_as_real(q_complex * freqs_cis).flatten(-2)
        k_out = torch.view_as_real(k_complex * freqs_cis).flatten(-2)
        
        # Convert back to the original dtype if needed
        q_out = q_out.type_as(q)
        k_out = k_out.type_as(k)
        
        return q_out, k_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        For compatibility with other positional encoding classes.
        Note: RoPE is typically applied inside the attention mechanism,
        not directly to the input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            Input tensor unchanged (RoPE is applied in attention)
        """
        # RoPE is applied in the attention mechanism, not here
        # This method exists for API compatibility with other positional encodings
        return x