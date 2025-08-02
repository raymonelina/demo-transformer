"""Transformer model implementation."""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List, Tuple
import matplotlib.pyplot as plt

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .config import TransformerConfig
from .debug_utils import debug_print
from .visualization import plot_attention_weights, plot_embeddings_pca, plot_attention_heads


class Transformer(nn.Module):
    """
    A complete Encoder-Decoder Transformer model for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        config: Union[TransformerConfig, Dict[str, Any]],
    ):
        """
        Initialize a Transformer model.

        Args:
            config: A TransformerConfig object or a dictionary with configuration parameters
        """
        super().__init__()

        # Convert dict to config if needed
        if isinstance(config, dict):
            config = TransformerConfig(**config)

        self.config = config

        # Store attention weights for visualization
        self.store_attention = getattr(config, "store_attention", False)

        # Create encoder and decoder
        self.encoder = TransformerEncoder(
            vocab_size=config.src_vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.num_encoder_layers,
            max_seq_len=config.max_seq_len,
            dropout_rate=config.dropout_rate,
            pre_norm=config.pre_norm,
            use_relative_pos=config.use_relative_pos,
            use_rope=config.use_rope,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            debug_mode=config.debug_mode,
            store_attention=self.store_attention,
        )

        self.decoder = TransformerDecoder(
            vocab_size=config.tgt_vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.num_decoder_layers,
            max_seq_len=config.max_seq_len,
            dropout_rate=config.dropout_rate,
            pre_norm=config.pre_norm,
            use_relative_pos=config.use_relative_pos,
            use_rope=config.use_rope,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            debug_mode=config.debug_mode,
            store_attention=self.store_attention,
        )

        # Implement weight tying if configured
        if config.weight_tying and config.src_vocab_size == config.tgt_vocab_size:
            self.decoder.output_projection.weight = self.decoder.token_embedding.weight

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model - USED FOR TRAINING with teacher forcing.
        
        TEACHER FORCING explained:
        During training, we feed the correct target sequence as input, rather than using
        the model's own predictions. This allows parallel processing of all positions.
        
        Example - Training to translate "I love cats" → "J'aime les chats":
        
        ENCODER:
        - Input (src_ids):  ["I", "love", "cats"]     # English source
        - Output: encoder_output                      # Understanding of English meaning
        
        DECODER (with teacher forcing):
        - Input (tgt_ids):  [SOS, "J'aime", "les", "chats"]  # Correct French as input
        - Uses: encoder_output + tgt_ids                     # English meaning + French context
        - Target output: ["J'aime", "les", "chats", EOS]     # What we want to predict
        
        KEY DIFFERENCE - Where do the French words come from?
        
        TRAINING (Teacher Forcing - PARALLEL):
        - We GIVE the model the correct French: [SOS, "J'aime", "les", "chats"]
        - Model processes ALL positions at once:
          Position 0: English + SOS → predict "J'aime" (we know answer is "J'aime")
          Position 1: English + [SOS, "J'aime"] → predict "les" (we know answer is "les")
          Position 2: English + [SOS, "J'aime", "les"] → predict "chats" (we know answer is "chats")
          Position 3: English + [SOS, "J'aime", "les", "chats"] → predict EOS (we know answer is EOS)
        
        INFERENCE (No Teacher Forcing - SEQUENTIAL):
        - Model must GENERATE its own French words step by step:
          Step 1: English + [SOS] → predict "J'aime" (model's guess)
          Step 2: English + [SOS, "J'aime"] → predict "les" (using its own "J'aime")
          Step 3: English + [SOS, "J'aime", "les"] → predict "chats" (using its own words)
          Step 4: English + [SOS, "J'aime", "les", "chats"] → predict EOS (using its own words)
        
        Training: Uses CORRECT French words (fast, parallel)
        Inference: Uses MODEL'S OWN French words (slow, sequential, can make mistakes)
        
        MASKING'S ROLE in Teacher Forcing:
        Even though we give the model ALL French words [SOS, "J'aime", "les", "chats"],
        masking prevents cheating by blocking future tokens:
        
        Position 0 (predicting "J'aime"):
        - Can see: [SOS] ✓
        - MASKED: ["J'aime", "les", "chats"] ✗ (blocked by causal mask)
        
        Position 1 (predicting "les"):
        - Can see: [SOS, "J'aime"] ✓
        - MASKED: ["les", "chats"] ✗ (blocked by causal mask)
        
        Position 2 (predicting "chats"):
        - Can see: [SOS, "J'aime", "les"] ✓
        - MASKED: ["chats"] ✗ (blocked by causal mask)
        
        Without masking, the model would cheat by looking at future answers!
        Teacher forcing makes training faster (parallel) and more stable (uses correct tokens).

        Args:
            src_ids: Source token IDs [batch_size, src_seq_len]
            tgt_ids: Complete target token IDs [batch_size, tgt_seq_len]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            tgt_padding_mask: Target padding mask [batch_size, 1, tgt_seq_len, tgt_seq_len]

        Returns:
            Decoder logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        if hasattr(self.config, "debug_mode") and self.config.debug_mode:
            debug_print(src_ids, "src_ids", "Source token IDs", "Transformer: ")
            debug_print(tgt_ids, "tgt_ids", "Target token IDs", "Transformer: ")
            if src_padding_mask is not None:
                debug_print(
                    src_padding_mask, "src_padding_mask", "Source padding mask", "Transformer: "
                )
            if tgt_padding_mask is not None:
                debug_print(
                    tgt_padding_mask, "tgt_padding_mask", "Target padding mask", "Transformer: "
                )

        encoder_output = self.encoder(src_ids, src_padding_mask)

        if hasattr(self.config, "debug_mode") and self.config.debug_mode:
            debug_print(encoder_output, "encoder_output", "Encoder output tensor", "Transformer: ")

        decoder_logits = self.decoder(tgt_ids, encoder_output, src_padding_mask, tgt_padding_mask)

        if hasattr(self.config, "debug_mode") and self.config.debug_mode:
            debug_print(decoder_logits, "decoder_logits", "Decoder output logits", "Transformer: ")

        return decoder_logits

    def encode(
        self, src_ids: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode the source sequence - USED FOR INFERENCE (Step 1 of 2).
        
        This method encodes the source sequence once and returns the encoder output,
        which can be reused for multiple decoding steps during autoregressive generation.
        
        Inference usage:
        encoder_output = model.encode(src_ids, src_padding_mask)  # Call once
        # Then use encoder_output multiple times in decode() for token generation
        
        Args:
            src_ids: Source token IDs [batch_size, src_seq_len]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            
        Returns:
            Encoder output [batch_size, src_seq_len, embed_dim]
        """
        return self.encoder(src_ids, src_padding_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode with target sequence and encoder output - USED FOR INFERENCE (Step 2 of 2).
        
        This method takes a partial target sequence and generates predictions for the next token.
        Called iteratively during autoregressive generation, where the sequence grows one token
        at a time: [SOS] → [SOS, token1] → [SOS, token1, token2] → ...
        
        Inference usage:
        for step in range(max_len):
            current_seq = [SOS, generated_token1, generated_token2, ...]  # Growing sequence
            logits = model.decode(current_seq, encoder_output, src_padding_mask)
            next_token = argmax(logits[:, -1, :])  # Only use prediction for last position
            
        Args:
            tgt_ids: Partial target sequence [batch_size, current_seq_len]
            encoder_output: Pre-computed encoder output [batch_size, src_seq_len, embed_dim]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            
        Returns:
            Decoder logits [batch_size, current_seq_len, tgt_vocab_size]
        """
        return self.decoder(tgt_ids, encoder_output, src_padding_mask, tgt_padding_mask)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Transformer":
        """Load a pretrained model from a checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu")
        config = TransformerConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, model_path: str) -> None:
        """Save the model to a checkpoint."""
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        checkpoint = {"config": config_dict, "model_state_dict": self.state_dict()}
        torch.save(checkpoint, model_path)
        
        # Print file size
        file_size = os.path.getsize(model_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"Model saved to {model_path} (Size: {file_size_mb:.2f} MB)")

    def get_encoder_attention_weights(self) -> List[torch.Tensor]:
        """Get attention weights from all encoder layers.

        Returns:
            List of attention weight tensors, one per layer
        """
        if not self.store_attention:
            raise ValueError("Attention weights not stored. Initialize with store_attention=True")

        attention_weights = []
        for i, layer in enumerate(self.encoder.encoder_layers):
            attention_weights.append(layer.self_attn.last_attention_weights)

        return attention_weights

    def get_decoder_self_attention_weights(self) -> List[torch.Tensor]:
        """Get self-attention weights from all decoder layers.

        Returns:
            List of attention weight tensors, one per layer
        """
        if not self.store_attention:
            raise ValueError("Attention weights not stored. Initialize with store_attention=True")

        attention_weights = []
        for i, layer in enumerate(self.decoder.decoder_layers):
            attention_weights.append(layer.self_attn.last_attention_weights)

        return attention_weights

    def get_decoder_cross_attention_weights(self) -> List[torch.Tensor]:
        """Get cross-attention weights from all decoder layers.

        Returns:
            List of attention weight tensors, one per layer
        """
        if not self.store_attention:
            raise ValueError("Attention weights not stored. Initialize with store_attention=True")

        attention_weights = []
        for i, layer in enumerate(self.decoder.decoder_layers):
            attention_weights.append(layer.cross_attn.last_attention_weights)

        return attention_weights

    def visualize_encoder_attention(
        self,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize encoder self-attention weights.

        Args:
            layer_idx: Index of the encoder layer to visualize
            head_idx: Index of the attention head to visualize (None for average)
            tokens: Optional list of token strings for axis labels
            **kwargs: Additional arguments to pass to plot_attention_weights

        Returns:
            Matplotlib figure object
        """
        attention_weights = self.get_encoder_attention_weights()
        return plot_attention_weights(
            attention_weights[layer_idx],
            tokens=tokens,
            title="Encoder Self-Attention",
            layer_idx=layer_idx,
            head_idx=head_idx,
            **kwargs,
        )

    def visualize_decoder_self_attention(
        self,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize decoder self-attention weights.

        Args:
            layer_idx: Index of the decoder layer to visualize
            head_idx: Index of the attention head to visualize (None for average)
            tokens: Optional list of token strings for axis labels
            **kwargs: Additional arguments to pass to plot_attention_weights

        Returns:
            Matplotlib figure object
        """
        attention_weights = self.get_decoder_self_attention_weights()
        return plot_attention_weights(
            attention_weights[layer_idx],
            tokens=tokens,
            title="Decoder Self-Attention",
            layer_idx=layer_idx,
            head_idx=head_idx,
            **kwargs,
        )

    def visualize_decoder_cross_attention(
        self,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        src_tokens: Optional[List[str]] = None,
        tgt_tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize decoder cross-attention weights.

        Args:
            layer_idx: Index of the decoder layer to visualize
            head_idx: Index of the attention head to visualize (None for average)
            src_tokens: Optional list of source token strings for x-axis labels
            tgt_tokens: Optional list of target token strings for y-axis labels
            **kwargs: Additional arguments to pass to plot_attention_weights

        Returns:
            Matplotlib figure object
        """
        attention_weights = self.get_decoder_cross_attention_weights()
        # For cross-attention, we need both source and target tokens for proper visualization
        # Source tokens should be on x-axis (keys) and target tokens on y-axis (queries)
        return plot_attention_weights(
            attention_weights[layer_idx],
            tokens=src_tokens,  # Source tokens for cross-attention keys
            title="Decoder Cross-Attention",
            layer_idx=layer_idx,
            head_idx=head_idx,
            **kwargs,
        )

    def visualize_encoder_embeddings(
        self, input_ids: torch.Tensor, tokens: Optional[List[str]] = None, **kwargs
    ) -> plt.Figure:
        """Visualize encoder token embeddings using PCA.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            tokens: Optional list of token strings for labels
            **kwargs: Additional arguments to pass to plot_embeddings_pca

        Returns:
            Matplotlib figure object
        """
        with torch.no_grad():
            embeddings = self.encoder.token_embedding(input_ids[0])  # Take first batch

        return plot_embeddings_pca(
            embeddings, tokens=tokens, title="Encoder Token Embeddings", **kwargs
        )

    def visualize_attention_heads(
        self,
        attention_type: str = "encoder",
        layer_idx: int = 0,
        tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize multiple attention heads in a grid.

        Args:
            attention_type: Type of attention to visualize ("encoder", "decoder_self", or "decoder_cross")
            layer_idx: Index of the layer to visualize
            tokens: Optional list of token strings for axis labels
            **kwargs: Additional arguments to pass to plot_attention_heads

        Returns:
            Matplotlib figure object
        """
        if attention_type == "encoder":
            attention_weights = self.get_encoder_attention_weights()[layer_idx]
        elif attention_type == "decoder_self":
            attention_weights = self.get_decoder_self_attention_weights()[layer_idx]
        elif attention_type == "decoder_cross":
            attention_weights = self.get_decoder_cross_attention_weights()[layer_idx]
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        return plot_attention_heads(attention_weights, tokens=tokens, layer_idx=layer_idx, **kwargs)
