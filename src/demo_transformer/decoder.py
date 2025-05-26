# src/demo_transformer/decoder.py

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForwardBlock
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    A single Transformer Decoder Layer.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout_rate)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(
        self,
        target_input: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ):
        norm_tgt = self.norm1(target_input)
        self_attn_output = self.self_attn(norm_tgt, norm_tgt, norm_tgt, mask=tgt_mask)
        x = target_input + self.dropout1(self_attn_output)

        norm_x = self.norm2(x)
        cross_attn_output = self.cross_attn(
            norm_x, encoder_output, encoder_output, mask=src_mask
        )
        x = x + self.dropout2(cross_attn_output)

        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout3(ff_output)

        return x


class TransformerDecoder(nn.Module):
    """
    A simplified Transformer Decoder.
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
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, ff_dim, dropout_rate)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        mask = torch.ones(sz, sz, device=device).bool()
        mask = ~torch.tril(mask)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        target_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        src_padding_mask: torch.Tensor = None,
    ):
        target_seq_len = target_ids.size(1)

        tgt_mask = self.generate_square_subsequent_mask(
            target_seq_len, target_ids.device
        )

        target_embeddings = self.token_embedding(target_ids)
        x = self.positional_encoding(target_embeddings)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask, src_mask=src_padding_mask)

        logits = self.output_projection(x)

        return logits
