# src/demo_transformer/feed_forward.py

import torch
import torch.nn as nn


class FeedForwardBlock(nn.Module):
    """
    A simple two-layer feed-forward network with GELU activation and dropout.
    """

    def __init__(self, embed_dim: int, ff_dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x
