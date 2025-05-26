# src/demo_transformer/transformer.py

import torch
import torch.nn as nn
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class Transformer(nn.Module):
    """
    A complete Encoder-Decoder Transformer model for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        max_seq_len: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_encoder_layers,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate,
        )
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_layers=num_decoder_layers,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_padding_mask: torch.Tensor = None,
    ):
        encoder_output = self.encoder(src_ids, src_padding_mask)
        decoder_logits = self.decoder(tgt_ids, encoder_output, src_padding_mask)
        return decoder_logits
