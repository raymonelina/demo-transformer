# src/demo_transformer/positional_encoding.py

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Applies sinusoidal positional encoding to input embeddings.
    """

    def __init__(self, embed_dim: int, max_seq_len: int = 5000):
        super().__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(
                f"Input sequence length ({seq_len}) exceeds max_seq_len ({self.pe.size(1)}) for positional encoding."
            )

        return x + self.pe[:, :seq_len, :]
