# src/demo_transformer/feed_forward.py

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    A simple two-layer feed-forward network with GELU activation and dropout.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
