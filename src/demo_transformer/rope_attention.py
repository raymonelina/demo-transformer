"""
Rotary Position Embedding (RoPE) based attention implementation.

This module implements the Rotary Position Embedding (RoPE) attention mechanism
as described in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
(Su et al., 2021, https://arxiv.org/abs/2104.09864).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        q, k = self.rope.apply_rotary_pos_emb(q, k, max(seq_len_q, seq_len_k))
        
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
        
        # Compute attention scores using scaled dot-product attention
        # The dot product between rotated query and key vectors now inherently captures relative position
        # The scaling factor (1/sqrt(head_dim)) prevents exploding values in softmax for numerical stability
        # This is particularly important with RoPE since the rotation preserves vector norms
        # [batch_size, num_heads, seq_len_q, head_dim] @ [batch_size, num_heads, head_dim, seq_len_k]
        # -> [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if self.debug_mode:
            debug_print(
                attn_scores, "attn_scores", "Attention scores before masking", "RoPEMultiHeadAttention: "
            )
        
        # Apply mask if provided to prevent attending to certain positions
        # The mask is typically used for two purposes:
        # 1. Padding mask: To ignore padding tokens in variable-length sequences
        # 2. Causal/autoregressive mask: To prevent attending to future positions in sequence generation
        # By setting masked positions to negative infinity, they become zero after softmax
        # This effectively removes their influence on the attention weights
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            
            if self.debug_mode:
                debug_print(
                    attn_scores, "masked_attn_scores", "Attention scores after masking", "RoPEMultiHeadAttention: "
                )
        
        # Apply softmax to normalize attention scores into a probability distribution
        # The softmax operation ensures that for each query position, the attention weights
        # across all key positions sum to 1, creating a proper probability distribution
        # This allows the model to focus on the most relevant tokens while still considering others
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply dropout to attention weights for regularization
        # This prevents the model from becoming too dependent on specific attention patterns
        # and encourages it to use diverse information sources, improving generalization
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
        
        # Apply attention weights to values through matrix multiplication
        # This is the core of the attention mechanism: a weighted sum of value vectors
        # Each query position attends to all key positions with different weights
        # The result is a context vector that aggregates information from the entire sequence
        # with emphasis on the most relevant parts according to the attention distribution
        # [batch_size, num_heads, seq_len_q, seq_len_k] @ [batch_size, num_heads, seq_len_k, head_dim]
        # -> [batch_size, num_heads, seq_len_q, head_dim]
        context = torch.matmul(attn_weights, v)
        
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