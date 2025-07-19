"""Transformer model implementation."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List, Tuple

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
        self.store_attention = getattr(config, 'store_attention', False)
        
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
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            debug_mode=config.debug_mode,
            store_attention=store_attention,
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
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            debug_mode=config.debug_mode,
            store_attention=store_attention,
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
        Forward pass of the transformer model.
        
        Args:
            src_ids: Source token IDs [batch_size, src_seq_len]
            tgt_ids: Target token IDs [batch_size, tgt_seq_len]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            tgt_padding_mask: Target padding mask [batch_size, 1, tgt_seq_len, tgt_seq_len]
            
        Returns:
            Decoder logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
            debug_print(src_ids, "src_ids", "Source token IDs", "Transformer: ")
            debug_print(tgt_ids, "tgt_ids", "Target token IDs", "Transformer: ")
            if src_padding_mask is not None:
                debug_print(src_padding_mask, "src_padding_mask", "Source padding mask", "Transformer: ")
            if tgt_padding_mask is not None:
                debug_print(tgt_padding_mask, "tgt_padding_mask", "Target padding mask", "Transformer: ")
                
        encoder_output = self.encoder(src_ids, src_padding_mask)
        
        if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
            debug_print(encoder_output, "encoder_output", "Encoder output tensor", "Transformer: ")
            
        decoder_logits = self.decoder(tgt_ids, encoder_output, src_padding_mask)
        
        if hasattr(self.config, 'debug_mode') and self.config.debug_mode:
            debug_print(decoder_logits, "decoder_logits", "Decoder output logits", "Transformer: ")
            
        return decoder_logits
    
    def encode(self, src_ids: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode the source sequence."""
        return self.encoder(src_ids, src_padding_mask)
    
    def decode(self, tgt_ids: torch.Tensor, encoder_output: torch.Tensor, 
               src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode with target sequence and encoder output."""
        return self.decoder(tgt_ids, encoder_output, src_padding_mask)
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'Transformer':
        """Load a pretrained model from a checkpoint."""
        checkpoint = torch.load(model_path, map_location='cpu')
        config = TransformerConfig(**checkpoint['config'])
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def save_pretrained(self, model_path: str) -> None:
        """Save the model to a checkpoint."""
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        checkpoint = {
            'config': config_dict,
            'model_state_dict': self.state_dict()
        }
        torch.save(checkpoint, model_path)
        
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
    
    def visualize_encoder_attention(self, layer_idx: int = 0, head_idx: Optional[int] = None, 
                                  tokens: Optional[List[str]] = None, **kwargs) -> plt.Figure:
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
            **kwargs
        )
    
    def visualize_decoder_self_attention(self, layer_idx: int = 0, head_idx: Optional[int] = None, 
                                        tokens: Optional[List[str]] = None, **kwargs) -> plt.Figure:
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
            **kwargs
        )
    
    def visualize_decoder_cross_attention(self, layer_idx: int = 0, head_idx: Optional[int] = None, 
                                         src_tokens: Optional[List[str]] = None, 
                                         tgt_tokens: Optional[List[str]] = None, 
                                         **kwargs) -> plt.Figure:
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
        return plot_attention_weights(
            attention_weights[layer_idx],
            tokens=src_tokens if tgt_tokens is None else tgt_tokens,  # Use one set if the other is missing
            title="Decoder Cross-Attention",
            layer_idx=layer_idx,
            head_idx=head_idx,
            **kwargs
        )
    
    def visualize_encoder_embeddings(self, input_ids: torch.Tensor, 
                                   tokens: Optional[List[str]] = None, **kwargs) -> plt.Figure:
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
            embeddings,
            tokens=tokens,
            title="Encoder Token Embeddings",
            **kwargs
        )
    
    def visualize_attention_heads(self, attention_type: str = "encoder", layer_idx: int = 0, 
                                tokens: Optional[List[str]] = None, **kwargs) -> plt.Figure:
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
        
        return plot_attention_heads(
            attention_weights,
            tokens=tokens,
            layer_idx=layer_idx,
            **kwargs
        )
