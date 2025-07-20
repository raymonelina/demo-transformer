"""
Rotary Position Embedding (RoPE) based attention implementation.
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
    
    This implementation applies rotary position embeddings directly to the
    query and key projections before computing attention scores.
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
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout probability
            debug_mode: Whether to print debug information
            store_attention: Whether to store attention weights for visualization
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
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Rotary position encoding
        self.rope = RotaryPositionalEncoding(self.head_dim, max_seq_len)
        
        # Dropout
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
        
        # Linear projections and reshape for multi-head attention
        # [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        
        if self.debug_mode:
            debug_print(q, "q_projected", "Query after projection", "RoPEMultiHeadAttention: ")
            debug_print(k, "k_projected", "Key after projection", "RoPEMultiHeadAttention: ")
            debug_print(v, "v_projected", "Value after projection", "RoPEMultiHeadAttention: ")
        
        # Apply rotary position embeddings
        q, k = self.rope.apply_rotary_pos_emb(q, k, max(seq_len_q, seq_len_k))
        
        if self.debug_mode:
            debug_print(q, "q_rotary", "Query after rotary encoding", "RoPEMultiHeadAttention: ")
            debug_print(k, "k_rotary", "Key after rotary encoding", "RoPEMultiHeadAttention: ")
        
        # Transpose for attention computation
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len_q, head_dim] @ [batch_size, num_heads, head_dim, seq_len_k]
        # -> [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if self.debug_mode:
            debug_print(
                attn_scores, "attn_scores", "Attention scores before masking", "RoPEMultiHeadAttention: "
            )
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
            
            if self.debug_mode:
                debug_print(
                    attn_scores, "masked_attn_scores", "Attention scores after masking", "RoPEMultiHeadAttention: "
                )
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        if self.debug_mode:
            debug_print(
                attn_weights, "attn_weights", "Attention weights after softmax", "RoPEMultiHeadAttention: "
            )
        
        # Store attention weights if required
        if self.store_attention:
            self.last_attn_weights = attn_weights.detach()
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_len_q, seq_len_k] @ [batch_size, num_heads, seq_len_k, head_dim]
        # -> [batch_size, num_heads, seq_len_q, head_dim]
        context = torch.matmul(attn_weights, v)
        
        if self.debug_mode:
            debug_print(context, "context", "Context after attention", "RoPEMultiHeadAttention: ")
        
        # Transpose and reshape back
        # [batch_size, num_heads, seq_len_q, head_dim] -> [batch_size, seq_len_q, embed_dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(context)
        
        if self.debug_mode:
            debug_print(output, "output", "Final output", "RoPEMultiHeadAttention: ")
        
        return output