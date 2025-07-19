"""Transformer model implementation."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .config import TransformerConfig
from .debug_utils import debug_print


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
