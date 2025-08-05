"""Transformer encoder implementation."""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Optional, Tuple

from .debug_utils import debug_print

from .multi_head_attention import MultiHeadAttention
from .relative_attention import RelativeMultiHeadAttention
from .rope_attention import RoPEMultiHeadAttention
from .feed_forward import FeedForwardBlock
from .positional_encoding import PositionalEncoding
from .relative_positional_encoding import RelativePositionalEncoding
from .rotary_positional_encoding import RotaryPositionalEncoding


class EncoderLayer(nn.Module):
    """
    A single Transformer Encoder Layer.
    
    NORMALIZATION PLACEMENT:
    - Post-norm (original): LayerNorm applied AFTER residual connection
      From "Attention Is All You Need" (Vaswani et al., 2017)
      Formula: LayerNorm(x + Sublayer(x))
      
    - Pre-norm (modern): LayerNorm applied BEFORE sublayer
      From "Learning Deep Transformer Models for Machine Translation" (Wang et al., 2019)
      and "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
      Formula: x + Sublayer(LayerNorm(x))
      
    Pre-norm advantages:
    - Better gradient flow and training stability
    - Enables training much deeper models (100+ layers)
    - Faster convergence and less sensitive to learning rate
    - Used in GPT-2, T5, and most modern large language models
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        pre_norm: bool = True,
        use_relative_pos: bool = False,
        use_rope: bool = False,
        max_seq_len: int = 512,
        debug_mode: bool = False,
        store_attention: bool = False,
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
        self.debug_mode = debug_mode
        self.store_attention = store_attention
        
        # Choose the appropriate attention mechanism based on parameters
        if use_relative_pos:
            self.self_attn = RelativeMultiHeadAttention(
                embed_dim,
                num_heads,
                max_seq_len,
                debug_mode=debug_mode,
                store_attention=store_attention,
            )
        elif use_rope:
            self.self_attn = RoPEMultiHeadAttention(
                embed_dim,
                num_heads,
                max_seq_len,
                debug_mode=debug_mode,
                store_attention=store_attention,
            )
        else:
            self.self_attn = MultiHeadAttention(
                embed_dim, num_heads, debug_mode=debug_mode, store_attention=store_attention
            )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.feed_forward = FeedForwardBlock(embed_dim, ff_dim, dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.pre_norm = pre_norm
        
        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.debug_mode:
            debug_print(x, "layer_input", "Input to encoder layer", "EncoderLayer: ")
        """
        Forward pass of the encoder layer.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            src_padding_mask: Source padding mask tensor [batch_size, 1, 1, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, embed_dim]
        """
        if self.pre_norm:
            # Pre-layer normalization: x + Sublayer(LayerNorm(x))
            # Better gradient flow, enables deeper models (Wang et al., 2019)
            # Note: Numerical stability is handled by the robust attention mechanism
            norm_x = self.norm1(x)
            self_attn_output = self.self_attn(norm_x, norm_x, norm_x, mask=src_padding_mask)
            x = x + self.dropout1(self_attn_output)

            norm_x = self.norm2(x)
            ff_output = self.feed_forward(norm_x)
            x = x + self.dropout2(ff_output)
        else:
            # Post-layer normalization: LayerNorm(x + Sublayer(x))
            # Original Transformer architecture (Vaswani et al., 2017)
            # Note: Numerical stability is handled by the robust attention mechanism
            self_attn_output = self.self_attn(x, x, x, mask=src_padding_mask)
            x = self.norm1(x + self.dropout1(self_attn_output))

            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))

        return x
    
    def _init_weights(self):
        """Initialize layer norm weights."""
        nn.init.ones_(self.norm1.weight)
        nn.init.zeros_(self.norm1.bias)
        nn.init.ones_(self.norm2.weight)
        nn.init.zeros_(self.norm2.bias)


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
        use_rope: bool = False,
        use_gradient_checkpointing: bool = False,
        debug_mode: bool = False,
        store_attention: bool = False,
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
        
        # Choose the appropriate positional encoding based on parameters
        if use_rope:
            # For RoPE, we still create a positional encoding object for API compatibility,
            # but the actual rotation is applied in the attention mechanism
            self.positional_encoding = RotaryPositionalEncoding(embed_dim, max_seq_len)
        elif use_relative_pos:
            # For relative attention, use learnable relative positional encodings
            # These are used by RelativeMultiHeadAttention for position-aware attention
            self.positional_encoding = RelativePositionalEncoding(embed_dim, max_seq_len)
        else:
            # Standard sinusoidal positional encoding (original Transformer)
            self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)
            
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim,
                    num_heads,
                    ff_dim,
                    dropout_rate,
                    pre_norm,
                    use_relative_pos,
                    use_rope,
                    max_seq_len,
                    debug_mode,
                    store_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

        # Final layer norm for pre-norm architecture
        self.final_norm = nn.LayerNorm(embed_dim) if pre_norm else None

        # Gradient checkpointing: Trade computation for memory
        # Academic foundation: "Training Deep Nets with Sublinear Memory Cost" (Chen et al., 2016)
        # Engineering optimization that enables training much larger models on limited GPU memory
        # 
        # How it works:
        # - Forward pass: Only store activations at checkpoints, discard intermediate ones
        # - Backward pass: Recompute discarded activations on-demand during backpropagation
        # - Memory reduction: O(√n) instead of O(n) for n layers
        # 
        # Benefits:
        # - Memory savings: ~50-80% reduction in activation memory
        # - Enables larger batch sizes or deeper models on same hardware
        # - Cost: ~33% increase in training time due to recomputation
        # - Used in GPT-3, T5, and other large language models
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Debug mode and attention storage
        self.debug_mode = debug_mode
        self.store_attention = store_attention
        
        # Initialize weights
        self._init_weights()

    def _layer_forward(
        self, layer: nn.Module, x: torch.Tensor, src_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Helper function for gradient checkpointing."""
        return layer(x, src_padding_mask)

    def forward(
        self, input_ids: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the encoder.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            src_padding_mask: Source padding mask [batch_size, 1, 1, seq_len]

        Returns:
            Encoder output [batch_size, seq_len, embed_dim]
        """
        if self.debug_mode:
            debug_print(input_ids, "input_ids", "Input token IDs", "Encoder: ")
            if src_padding_mask is not None:
                debug_print(
                    src_padding_mask, "src_padding_mask", "Source padding mask", "Encoder: "
                )

        embeddings = self.token_embedding(input_ids)
        if self.debug_mode:
            debug_print(
                embeddings,
                "token_embeddings",
                "Token embeddings before positional encoding",
                "Encoder: ",
            )

        # Apply positional encoding based on the type
        if isinstance(self.positional_encoding, RelativePositionalEncoding):
            # For relative attention, positional encoding is handled inside the attention mechanism
            # Don't modify embeddings here - just pass them through
            if self.debug_mode:
                debug_print(
                    embeddings, "pos_embeddings", "Embeddings (using RelativePositionalEncoding - handled in attention)", "Encoder: "
                )
        elif isinstance(self.positional_encoding, RotaryPositionalEncoding):
            # For RoPE, positional encoding is applied in attention mechanism
            embeddings = self.positional_encoding(embeddings)
            if self.debug_mode:
                debug_print(
                    embeddings, "pos_embeddings", "Embeddings (using RotaryPositionalEncoding - applied in attention)", "Encoder: "
                )
        else:
            # For standard sinusoidal positional encoding, apply to embeddings
            embeddings = self.positional_encoding(embeddings)
            if self.debug_mode:
                debug_print(
                    embeddings, "pos_embeddings", "Embeddings (using standard PositionalEncoding - added to embeddings)", "Encoder: "
                )

        x = self.dropout(embeddings)

        for i, layer in enumerate(self.encoder_layers):
            if self.debug_mode:
                debug_print(
                    x, f"encoder_layer_{i}_input", f"Input to encoder layer {i}", "Encoder: "
                )

            if self.use_gradient_checkpointing and self.training:
                # Apply gradient checkpointing: save memory by recomputing activations
                # Only during training - inference doesn't need gradients so no benefit
                x = torch.utils.checkpoint.checkpoint(
                    self._layer_forward, layer, x, src_padding_mask, use_reentrant=False
                )
            else:
                # Standard forward pass: store all intermediate activations
                x = layer(x, src_padding_mask)

            if self.debug_mode:
                debug_print(
                    x, f"encoder_layer_{i}_output", f"Output from encoder layer {i}", "Encoder: "
                )

        # Apply final normalization if using pre-norm
        if self.final_norm is not None:
            x = self.final_norm(x)
            if self.debug_mode:
                debug_print(
                    x, "encoder_final_norm", "Output after final layer normalization", "Encoder: "
                )

        if self.debug_mode:
            debug_print(x, "encoder_output", "Final encoder output", "Encoder: ")

        return x
    
    def _init_weights(self):
        """Initialize weights for embeddings and layer norms."""
        # Initialize token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize final layer norm if present
        if self.final_norm is not None:
            nn.init.ones_(self.final_norm.weight)
            nn.init.zeros_(self.final_norm.bias)
