# src/demo_transformer/attention.py

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Can be used for self-attention (query=key=value) or cross-attention.
    """

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()

        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))

        attention_probs = torch.softmax(attention_scores, dim=-1)

        context = torch.matmul(attention_probs, V)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len_q, self.embed_dim)
        )

        output = self.out_proj(context)

        return output
