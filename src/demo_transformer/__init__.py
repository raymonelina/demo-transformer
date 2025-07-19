"""
Transformer package for sequence-to-sequence tasks.
"""

from .transformer import Transformer
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .attention import MultiHeadAttention
from .feed_forward import FeedForwardBlock
from .positional_encoding import PositionalEncoding
from .relative_positional_encoding import RelativePositionalEncoding, RelativeMultiHeadAttention
from .config import TransformerConfig
from .training import TransformerTrainer, LabelSmoothingLoss, get_transformer_scheduler
from .inference_utils import TransformerInference
from .data import TransformerDataset, TransformerCollator, create_dataloaders
from .debug_utils import debug_print

__version__ = "0.2.0"

__all__ = [
    "Transformer",
    "TransformerEncoder",
    "TransformerDecoder",
    "MultiHeadAttention",
    "RelativeMultiHeadAttention",
    "FeedForwardBlock",
    "PositionalEncoding",
    "RelativePositionalEncoding",
    "TransformerConfig",
    "TransformerTrainer",
    "LabelSmoothingLoss",
    "get_transformer_scheduler",
    "TransformerInference",
    "TransformerDataset",
    "TransformerCollator",
    "create_dataloaders",
    "debug_print",
]