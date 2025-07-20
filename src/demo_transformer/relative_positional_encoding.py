"""
Relative positional encoding implementation for transformer models.
"""

import torch
import torch.nn as nn
import math


class RelativePositionalEncoding(nn.Module):
    """
    Implements relative positional encoding for transformer models.
    Based on "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 512):
        """
        Initialize relative positional encoding.

        Args:
            embed_dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Create relative position embeddings
        # We need 2*max_seq_len-1 positions (-max_seq_len+1 to max_seq_len-1)
        self.relative_positions_embeddings = nn.Parameter(
            torch.zeros(2 * max_seq_len - 1, embed_dim)
        )

        # Initialize with sinusoidal pattern
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with sinusoidal pattern."""
        positions = torch.arange(-(self.max_seq_len - 1), self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-math.log(10000.0) / self.embed_dim)
        )
        
        # Create a temporary buffer with sinusoidal pattern
        embeddings = torch.zeros(2 * self.max_seq_len - 1, self.embed_dim)
        embeddings[:, 0::2] = torch.sin(positions * div_term)
        embeddings[:, 1::2] = torch.cos(positions * div_term)
        
        # Copy to parameter (non-in-place operation)
        with torch.no_grad():
            self.relative_positions_embeddings.copy_(embeddings)

    def forward(self, length: int) -> torch.Tensor:
        """
        Get relative positional embeddings for a given sequence length.

        Args:
            length: Sequence length

        Returns:
            Relative positional embeddings [2*length-1, embed_dim]
        """
        if length > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({length}) exceeds max_seq_len ({self.max_seq_len})"
            )

        # Return embeddings for positions from -(length-1) to (length-1)
        start_pos = self.max_seq_len - length
        end_pos = start_pos + 2 * length - 1
        return self.relative_positions_embeddings[start_pos:end_pos]
