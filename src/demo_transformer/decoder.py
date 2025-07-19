"""Transformer decoder implementation."""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .relative_positional_encoding import RelativeMultiHeadAttention
from .feed_forward import FeedForwardBlock
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    """
    A single Transformer Decoder Layer.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1,
        pre_norm: bool = True, use_relative_pos: bool = False, max_seq_len: int = 512
    ):
        """
        Initialize a decoder layer.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout_rate: Dropout probability
            pre_norm: Whether to use pre-layer normalization (True) or post-layer normalization (False)
        """
        super().__init__()
        if use_relative_pos:
            self.self_attn = RelativeMultiHeadAttention(embed_dim, num_heads, max_seq_len)
            self.cross_attn = RelativeMultiHeadAttention(embed_dim, num_heads, max_seq_len)
        else:
            self.self_attn = MultiHeadAttention(embed_dim, num_heads)
            self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout_rate)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.pre_norm = pre_norm

    def forward(
        self,
        target_input: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder layer.
        
        Args:
            target_input: Target input tensor [batch_size, tgt_seq_len, embed_dim]
            encoder_output: Encoder output tensor [batch_size, src_seq_len, embed_dim]
            tgt_mask: Target mask tensor [batch_size, 1, tgt_seq_len, tgt_seq_len]
            src_mask: Source mask tensor [batch_size, 1, 1, src_seq_len]
            
        Returns:
            Output tensor [batch_size, tgt_seq_len, embed_dim]
        """
        if self.pre_norm:
            # Pre-layer normalization
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
        else:
            # Post-layer normalization (original transformer)
            self_attn_output = self.self_attn(target_input, target_input, target_input, mask=tgt_mask)
            x = self.norm1(target_input + self.dropout1(self_attn_output))
            
            cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, mask=src_mask)
            x = self.norm2(x + self.dropout2(cross_attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout3(ff_output))

        return x


class TransformerDecoder(nn.Module):
    """
    A Transformer Decoder.
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
        pre_norm: bool = True,
        use_relative_pos: bool = False,
    ):
        """
        Initialize a transformer decoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            num_layers: Number of decoder layers
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout probability
            pre_norm: Whether to use pre-layer normalization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(embed_dim, num_heads, ff_dim, dropout_rate, pre_norm, use_relative_pos, max_seq_len)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        
        # Final layer norm for pre-norm architecture
        self.final_norm = nn.LayerNorm(embed_dim) if pre_norm else None

    def generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate a square mask for the sequence to prevent attending to future positions.
        
        Args:
            sz: Size of the square mask
            device: Device to create the mask on
            
        Returns:
            Mask tensor [1, 1, sz, sz]
        """
        mask = torch.ones(sz, sz, device=device).bool()
        mask = ~torch.tril(mask)  # Lower triangular part is False (not masked)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        target_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the decoder.
        
        Args:
            target_ids: Target token IDs [batch_size, tgt_seq_len]
            encoder_output: Encoder output tensor [batch_size, src_seq_len, embed_dim]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            tgt_padding_mask: Target padding mask [batch_size, 1, tgt_seq_len, tgt_seq_len]
            
        Returns:
            Decoder logits [batch_size, tgt_seq_len, vocab_size]
        """
        target_seq_len = target_ids.size(1)

        # Generate causal mask to prevent attending to future positions
        causal_mask = self.generate_square_subsequent_mask(
            target_seq_len, target_ids.device
        )
        
        # Combine causal mask with padding mask if provided
        tgt_mask = causal_mask
        if tgt_padding_mask is not None:
            tgt_mask = tgt_mask | tgt_padding_mask

        target_embeddings = self.token_embedding(target_ids)
        x = self.positional_encoding(target_embeddings)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, tgt_mask=tgt_mask, src_mask=src_padding_mask)
            
        # Apply final normalization if using pre-norm
        if self.final_norm is not None:
            x = self.final_norm(x)

        logits = self.output_projection(x)

        return logits
