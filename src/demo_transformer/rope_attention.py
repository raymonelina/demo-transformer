"""
Rotary Position Embedding (RoPE) based attention implementation.

This module implements the Rotary Position Embedding (RoPE) attention mechanism
as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
(Su et al., 2021, https://arxiv.org/abs/2104.09864).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .debug_utils import debug_print
from .rotary_positional_encoding import RotaryPositionalEncoding


class RoPEMultiHeadAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embeddings (RoPE).
    
    This implementation is based on the paper "RoFormer: Enhanced Transformer with Rotary 
    Position Embedding" (Su et al., 2021, https://arxiv.org/abs/2104.09864). The key innovation 
    is encoding relative position information through a rotation matrix that is multiplied with 
    query and key representations in the complex space.
    
    Key advantages over standard and relative positional encoding:
    1. Incorporates explicit relative position information without introducing additional parameters
    2. Provides superior performance on tasks requiring precise understanding of token positions
    3. Generalizes extremely well to sequences of arbitrary lengths not seen during training
    4. Preserves the inner product of token representations at different positions
    
    The approach works by:
    - Projecting queries and keys to multi-head representation
    - Applying a rotation in the complex space to each head dimension pair
    - The rotation angle is determined by the absolute position and dimension
    - Computing attention scores with these rotated representations
    
    Mathematical formulation:
    For a query q and key k at positions m and n, RoPE applies a rotation matrix R_θ:
    <R_θ(q_m), R_θ+mΔθ(k_n)> = <q_m, R_(n-m)Δθ(k_n)>
    where θ is the rotation angle determined by the dimension and Δθ is the position-dependent angle.
    
    When store_attention=True, the attention weights (after softmax) are saved in
    last_attn_weights for later visualization or analysis. This is useful for:
    - Visualizing attention patterns to understand model behavior
    - Interpreting which input tokens the model focuses on
    - Debugging attention mechanisms
    - Creating attention heatmaps for explainability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout_rate: float = 0.1,
        debug_mode: bool = False,
        store_attention: bool = False,
    ):
        """
        Initialize RoPE multi-head attention.

        Args:
            embed_dim: Dimension of embeddings. Must be divisible by num_heads.
            num_heads: Number of attention heads. Each head will have dimension embed_dim/num_heads.
            max_seq_len: Maximum sequence length for position embeddings. Determines the range of
                positions that can be represented. RoPE can generalize beyond this length, but
                performance may degrade for much longer sequences.
            dropout_rate: Dropout probability applied to attention weights. Higher values
                provide stronger regularization but may impede learning.
            debug_mode: Whether to print debug information about tensor shapes and values during
                execution. Useful for understanding the attention mechanism's behavior.
            store_attention: Whether to store attention weights for later visualization and analysis.
                When True, attention weights are stored in last_attn_weights after each forward pass.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be divisible by 2 for RoPE"
            )
        
        # Linear projections for query, key, and value
        # These projections transform the input embeddings into query, key, and value representations
        # Each projection preserves the embedding dimension but internally separates it into heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection to combine and transform the attended values back to embedding space
        # This is crucial for integrating information from all attention heads into a unified representation
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Rotary position encoding module
        # Unlike standard positional encoding which is added to embeddings,
        # RoPE applies a rotation to query and key vectors in the complex space
        # This encodes relative position information directly in the attention computation
        # The rotation preserves vector norms while encoding positional information
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        
        # Dropout applied to attention weights
        # This helps prevent overfitting and encourages the model to use diverse attention patterns
        # rather than focusing too strongly on specific token relationships
        self.dropout = nn.Dropout(dropout_rate)
        
        # Debug mode and attention storage
        self.debug_mode = debug_mode
        self.store_attention = store_attention
        self.last_attn_weights = None
        
        # Initialize weights
        self._init_weights()
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of RoPE multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, embed_dim]
            key: Key tensor [batch_size, seq_len_k, embed_dim]
            value: Value tensor [batch_size, seq_len_k, embed_dim]
            mask: Attention mask [batch_size, 1, 1, seq_len_k]
            
        Returns:
            Output tensor [batch_size, seq_len_q, embed_dim]
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)
        
        if self.debug_mode:
            debug_print(query, "query_input", "Query input", "RoPEMultiHeadAttention: ")
            debug_print(key, "key_input", "Key input", "RoPEMultiHeadAttention: ")
            debug_print(value, "value_input", "Value input", "RoPEMultiHeadAttention: ")
            if mask is not None:
                debug_print(mask, "mask", "Attention mask", "RoPEMultiHeadAttention: ")
        
        # Project inputs to query, key, and value representations and reshape for multi-head processing
        # The linear projections transform the input embeddings into specialized representations
        # for querying, storing, and retrieving information in the attention mechanism
        # The view operation reshapes the tensors to separate the embedding dimension into heads
        # This allows each head to focus on different aspects of the input representation
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        if self.debug_mode:
            debug_print(q, "q_projected", "Query after projection", "RoPEMultiHeadAttention: ")
            debug_print(k, "k_projected", "Key after projection", "RoPEMultiHeadAttention: ")
            debug_print(v, "v_projected", "Value after projection", "RoPEMultiHeadAttention: ")
        
        # Apply rotary position embeddings to query and key representations
        # This is the core of RoPE: applying a position-dependent rotation to each dimension pair
        # The rotation encodes relative position information directly in the query and key vectors
        # Unlike standard positional encoding, RoPE doesn't add position information but rotates vectors
        # This preserves the inner product structure while encoding position information
        # We use the maximum sequence length between query and key to ensure proper rotation
        q, k = self.rope.apply_rotary_pos_emb(q, k, seq_len_q, seq_len_k)
        
        if self.debug_mode:
            debug_print(q, "q_rotary", "Query after rotary encoding", "RoPEMultiHeadAttention: ")
            debug_print(k, "k_rotary", "Key after rotary encoding", "RoPEMultiHeadAttention: ")
        
        # Transpose tensors to prepare for batch matrix multiplication in attention computation
        # This rearrangement brings the num_heads dimension before the seq_len dimension
        # which allows for efficient parallel computation across all attention heads
        # The resulting shape facilitates the subsequent matrix multiplications for attention scores
        # and ensures that each head attends independently to the entire sequence
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores with numerical stability
        # Replace any NaN/Inf values to prevent propagation
        q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
        k = torch.where(torch.isfinite(k), k, torch.zeros_like(k))
        
        # Compute and clamp attention scores
        raw_scores = torch.matmul(q, k.transpose(-2, -1))
        raw_scores = torch.where(torch.isfinite(raw_scores), raw_scores, torch.zeros_like(raw_scores))
        attn_scores = torch.clamp(raw_scores / (self.head_dim ** 0.5), min=-10.0, max=10.0)
        
        if self.debug_mode:
            debug_print(
                attn_scores, "attn_scores", "Attention scores before masking", "RoPEMultiHeadAttention: "
            )
        
        # Apply attention mask (if provided)
        # MASK SIZE REQUIREMENTS:
        # - mask must have shape [batch_size, num_heads, seq_len_q, seq_len_k]
        # - This matches attention_scores shape for element-wise masking
        # - batch_size: number of sequences in batch
        # - num_heads: number of attention heads (can broadcast from 1)
        # - seq_len_q: query sequence length (rows in attention matrix)
        # - seq_len_k: key/value sequence length (columns in attention matrix)
        #
        # MASK SEMANTICS:
        # - True values: positions to MASK OUT (set to -∞, become 0 after softmax)
        # - False values: positions to ATTEND TO (keep original scores)
        #
        # COMMON MASK TYPES BY TRANSFORMER COMPONENT:
        # - Padding mask: mask out padding tokens [batch, 1, 1, seq_len]
        #   * Used in: BOTH encoder and decoder (all attention types)
        #   * Purpose: ignore padding tokens in variable-length sequences
        # - Causal mask: prevent attention to future tokens [1, 1, seq_len, seq_len]
        #   * Used in: DECODER ONLY (self-attention)
        #   * Purpose: maintain autoregressive property during training
        # - Combined mask: padding + causal masks element-wise OR'd together
        #   * Used in: DECODER cross-attention (query padding + key padding)
        #   * Purpose: handle both sequence lengths in encoder-decoder attention
        #
        # Mask prevents attention to certain positions by setting scores to -∞
        # After softmax, these become 0, effectively removing their contribution
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
            
            if self.debug_mode:
                debug_print(
                    attn_scores, "masked_attn_scores", "Attention scores after masking", "RoPEMultiHeadAttention: "
                )
        
        # Apply softmax with fallback mechanism
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Replace any remaining NaN/Inf with uniform distribution
        if not torch.isfinite(attn_weights).all():
            uniform_prob = 1.0 / attn_weights.size(-1)
            attn_weights = torch.where(torch.isfinite(attn_weights), attn_weights, uniform_prob)
        
        # Apply dropout to attention weights for regularization
        attn_weights = self.dropout(attn_weights)
        
        if self.debug_mode:
            debug_print(
                attn_weights, "attn_weights", "Attention weights after softmax", "RoPEMultiHeadAttention: "
            )
        
        # Store attention weights if visualization or analysis is requested
        # The detach() call prevents these stored weights from participating in backpropagation
        # This is important for memory efficiency and to avoid modifying the computational graph
        if self.store_attention:
            self.last_attn_weights = attn_weights.detach()  # Store a copy without gradient tracking
        
        # Apply attention weights to values with safety checks
        v = torch.where(torch.isfinite(v), v, torch.zeros_like(v))
        context = torch.matmul(attn_weights, v)
        context = torch.where(torch.isfinite(context), context, torch.zeros_like(context))
        
        if self.debug_mode:
            debug_print(context, "context", "Context after attention", "RoPEMultiHeadAttention: ")
        
        # Transpose and reshape the context vectors back to the original dimensions
        # This step combines information from all attention heads into a single representation
        # The contiguous() call ensures the tensor is stored in a contiguous block of memory
        # which is important for efficient computation in the subsequent operations
        # [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        
        # Final linear projection to transform the context vectors
        # This projection allows the model to combine information from different heads
        # and transform it into a representation suitable for the next layer
        # It's a crucial part of the multi-head attention mechanism that allows
        # different heads to specialize in different aspects of the input
        output = self.out_proj(context)
        
        if self.debug_mode:
            debug_print(output, "output", "Final output", "RoPEMultiHeadAttention: ")
        
        return output
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization with proper scaling."""
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=gain)
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)