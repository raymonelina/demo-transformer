"""
Transformer package for sequence-to-sequence tasks.
"""

from .transformer import Transformer
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .multi_head_attention import MultiHeadAttention
from .relative_attention import RelativeMultiHeadAttention
from .rope_attention import RoPEMultiHeadAttention
from .feed_forward import FeedForwardBlock
from .positional_encoding import PositionalEncoding
from .relative_positional_encoding import RelativePositionalEncoding
from .rotary_positional_encoding import RotaryPositionalEncoding
from .config import TransformerConfig
from .training import TransformerTrainer, LabelSmoothingLoss, get_transformer_scheduler
from .inference_utils import TransformerInference
from .data import TransformerDataset, TransformerCollator, create_dataloaders
from .debug_utils import debug_print
from .visualization import plot_attention_weights, plot_embeddings_pca, plot_attention_heads

__version__ = "0.2.0"

__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "MultiHeadAttention",
    "RelativeMultiHeadAttention",
    "RoPEMultiHeadAttention",
    "FeedForwardBlock",
    "PositionalEncoding",
    "RelativePositionalEncoding",
    "RotaryPositionalEncoding",
    "TransformerConfig",
    "TransformerTrainer",
    "LabelSmoothingLoss",
    "get_transformer_scheduler",
    "TransformerInference",
    "TransformerDataset",
    "TransformerCollator",
    "create_dataloaders",
    "debug_print",
    "plot_attention_weights",
    "plot_embeddings_pca",
    "plot_attention_heads",
]
