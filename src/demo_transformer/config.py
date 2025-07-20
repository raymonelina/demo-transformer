"""
Configuration module for the transformer package.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration class for the Transformer model."""
    
    # Model architecture
    embed_dim: int = 512
    num_heads: int = 8
    ff_dim: int = 2048
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    
    # Vocabulary
    src_vocab_size: int = 32000
    tgt_vocab_size: Optional[int] = None  # If None, will use src_vocab_size (shared vocab)
    
    # Special tokens
    pad_token_id: int = 0
    sos_token_id: int = 1
    eos_token_id: int = 2
    
    # Training
    label_smoothing: float = 0.1
    weight_tying: bool = True
    pre_norm: bool = True  # Pre-layer normalization
    use_relative_pos: bool = False  # Whether to use relative positional encoding
    use_rope: bool = False  # Whether to use rotary positional encoding (RoPE)
    use_gradient_checkpointing: bool = False  # Whether to use gradient checkpointing to save memory
    debug_mode: bool = False  # Whether to print debug information about tensor shapes and values
    store_attention: bool = False  # Whether to store attention weights for visualization
    
    def __post_init__(self):
        """Validate and set default values after initialization."""
        if self.tgt_vocab_size is None:
            self.tgt_vocab_size = self.src_vocab_size