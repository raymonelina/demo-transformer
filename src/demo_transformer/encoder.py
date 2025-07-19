"""Transformer encoder implementation."""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Optional, Tuple

from .attention import MultiHeadAttention
from .relative_positional_encoding import RelativeMultiHeadAttention
from .feed_forward import FeedForwardBlock
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer.
    """

    def __init__(
        self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1,
        pre_norm: bool = True, use_relative_pos: bool = False, max_seq_len: int = 512
    ):
        """
        Initialize an encoder layer.
        
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
        else:
            self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            src_mask: Source mask tensor [batch_size, 1, 1, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        if self.pre_norm:
            # Pre-layer normalization
            norm_x = self.norm1(x)
            self_attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=src_mask)
            x = x + self.dropout1(self_attn_output)
            
            norm_x = self.norm2(x)
            ff_output = self.feed_forward(norm_x)
            x = x + self.dropout2(ff_output)
        else:
            # Post-layer normalization (original transformer)
            self_attn_output = self.self_attn(x, x, x, mask=src_mask)
            x = self.norm1(x + self.dropout1(self_attn_output))
            
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """
    A Transformer Encoder, stacking multiple EncoderLayers.
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
        use_gradient_checkpointing: bool = False,
    ):
        """
        Initialize a transformer encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            num_layers: Number of encoder layers
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout probability
            pre_norm: Whether to use pre-layer normalization
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(embed_dim, num_heads, ff_dim, dropout_rate, pre_norm, use_relative_pos, max_seq_len)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final layer norm for pre-norm architecture
        self.final_norm = nn.LayerNorm(embed_dim) if pre_norm else None
        
        # Gradient checkpointing to save memory
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def _layer_forward(self, layer: nn.Module, x: torch.Tensor, src_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Helper function for gradient checkpointing."""
        return layer(x, src_padding_mask)
    
    def forward(self, input_ids: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            src_padding_mask: Source padding mask [batch_size, 1, 1, seq_len]
            
        Returns:
            Encoder output [batch_size, seq_len, embed_dim]
        """
        embeddings = self.token_embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)
        x = self.dropout(embeddings)

        for i, layer in enumerate(self.encoder_layers):
            if self.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, layer, x, src_padding_mask
                )
            else:
                x = layer(x, src_padding_mask)
            
        # Apply final normalization if using pre-norm
        if self.final_norm is not None:
            x = self.final_norm(x)

        return x
