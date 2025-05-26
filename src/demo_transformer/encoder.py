# src/demo_transformer/encoder.py

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForwardBlock
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None):
        norm_x = self.norm1(x)
        self_attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=src_mask)
        x = x + self.dropout1(self_attn_output)

        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_output)

        return x


class TransformerEncoder(nn.Module):
    """
    A simplified Transformer Encoder, stacking multiple EncoderLayers.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_layers: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids: torch.Tensor, src_padding_mask: torch.Tensor = None):
        embeddings = self.token_embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)
        x = self.dropout(embeddings)

        for layer in self.encoder_layers:
            x = layer(x, src_padding_mask)

        return x
