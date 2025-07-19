"""
Relative positional encoding implementation for transformer models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .debug_utils import debug_print


class RelativePositionalEncoding(nn.Module):
    """
    Implements relative positional encoding for transformer models.
    Based on "Self-Attention with Relative Position Representations" (Shaw et al., 2018)
    """
    
    def __init__(self, embed_dim: int, max_seq_len: int = 512):
        """
        Initialize relative positional encoding.
        
        Args:
            embed_dim: Dimension of embeddings
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Create relative position embeddings
        # We need 2*max_seq_len-1 positions (-max_seq_len+1 to max_seq_len-1)
        self.relative_positions_embeddings = nn.Parameter(
            torch.zeros(2 * max_seq_len - 1, embed_dim)
        )
        
        # Initialize with sinusoidal pattern
        self._init_embeddings()
        
    def _init_embeddings(self):
        """Initialize embeddings with sinusoidal pattern."""
        positions = torch.arange(-(self.max_seq_len - 1), self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * 
            (-math.log(10000.0) / self.embed_dim)
        )
        
        # Apply sine to even indices
        self.relative_positions_embeddings[:, 0::2] = torch.sin(positions * div_term)
        # Apply cosine to odd indices
        self.relative_positions_embeddings[:, 1::2] = torch.cos(positions * div_term)
        
    def forward(self, length: int) -> torch.Tensor:
        """
        Get relative positional embeddings for a given sequence length.
        
        Args:
            length: Sequence length
            
        Returns:
            Relative positional embeddings [2*length-1, embed_dim]
        """
        if length > self.max_seq_len:
            raise ValueError(
                f"Input sequence length ({length}) exceeds max_seq_len ({self.max_seq_len})"
            )
            
        # Return embeddings for positions from -(length-1) to (length-1)
        start_pos = self.max_seq_len - length
        end_pos = start_pos + 2 * length - 1
        return self.relative_positions_embeddings[start_pos:end_pos]


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with relative positional encoding.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int = 512, debug_mode: bool = False, store_attention: bool = False):
        """
        Initialize relative multi-head attention.
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
            )
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projections for Q, K, V
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Relative position encoding
        self.rel_pos_encoding = RelativePositionalEncoding(self.head_dim, max_seq_len)
        
        # Additional projection for relative positions
        self.pos_key_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)
        
        # Debug mode and attention storage
        self.debug_mode = debug_mode
        self.store_attention = store_attention
        self.last_attention_weights = None
        
    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shift the relative logits to align them properly.
        
        Args:
            x: Input tensor [batch_size, num_heads, seq_len, 2*seq_len-1]
            
        Returns:
            Shifted tensor [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, num_heads, seq_len, _ = x.size()
        
        # Pad to shift from the right to left
        zero_pad = torch.zeros(
            (batch_size, num_heads, seq_len, 1), 
            device=x.device, 
            dtype=x.dtype
        )
        x_padded = torch.cat([zero_pad, x], dim=-1)
        
        # Reshape and slice
        x_padded = x_padded.view(batch_size, num_heads, seq_len + 1, seq_len * 2 - 1)
        x_shifted = x_padded[:, :, 1:].view_as(x)
        
        # Take the appropriate parts
        return x_shifted[:, :, :, :seq_len]
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of relative multi-head attention.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, embed_dim]
            key: Key tensor [batch_size, seq_len_k, embed_dim]
            value: Value tensor [batch_size, seq_len_k, embed_dim]
            mask: Attention mask [batch_size, 1, seq_len_q, seq_len_k]
            
        Returns:
            Output tensor [batch_size, seq_len_q, embed_dim]
        """
        if self.debug_mode:
            debug_print(query, "rel_query_input", "Query input tensor", "RelativeAttention: ")
            debug_print(key, "rel_key_input", "Key input tensor", "RelativeAttention: ")
            debug_print(value, "rel_value_input", "Value input tensor", "RelativeAttention: ")
            if mask is not None:
                debug_print(mask, "rel_attention_mask", "Attention mask tensor", "RelativeAttention: ")
        
        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()
        
        # Project inputs
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        if self.debug_mode:
            debug_print(Q, "rel_Q_projected", "Query after projection", "RelativeAttention: ")
            debug_print(K, "rel_K_projected", "Key after projection", "RelativeAttention: ")
            debug_print(V, "rel_V_projected", "Value after projection", "RelativeAttention: ")
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.debug_mode:
            debug_print(Q, "rel_Q_reshaped", "Query after reshaping for multi-head", "RelativeAttention: ")
            debug_print(K, "rel_K_reshaped", "Key after reshaping for multi-head", "RelativeAttention: ")
            debug_print(V, "rel_V_reshaped", "Value after reshaping for multi-head", "RelativeAttention: ")
        
        # Get relative position embeddings [2*seq_len_k-1, head_dim]
        rel_pos_emb = self.rel_pos_encoding(seq_len_k)
        if self.debug_mode:
            debug_print(rel_pos_emb, "rel_pos_emb", "Relative position embeddings", "RelativeAttention: ")
        
        # Project relative position embeddings
        rel_pos_key = self.pos_key_proj(rel_pos_emb)
        if self.debug_mode:
            debug_print(rel_pos_key, "rel_pos_key", "Projected relative position keys", "RelativeAttention: ")
        
        # Content-content attention
        content_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.debug_mode:
            debug_print(content_scores, "content_scores", "Content-content attention scores", "RelativeAttention: ")
        
        # Content-position attention
        # [batch_size, num_heads, seq_len_q, 2*seq_len_k-1]
        rel_pos_key = rel_pos_key.unsqueeze(0).unsqueeze(0)
        position_scores = torch.matmul(Q, rel_pos_key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.debug_mode:
            debug_print(position_scores, "position_scores", "Content-position attention scores", "RelativeAttention: ")
        
        # Shift and slice position scores to align them
        rel_position_scores = self._rel_shift(position_scores)
        if self.debug_mode:
            debug_print(rel_position_scores, "rel_position_scores", "Shifted position scores", "RelativeAttention: ")
        
        # Combine content and position scores
        attention_scores = content_scores + rel_position_scores
        if self.debug_mode:
            debug_print(attention_scores, "combined_attention_scores", "Combined attention scores", "RelativeAttention: ")
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))
            if self.debug_mode:
                debug_print(attention_scores, "masked_attention_scores", "Attention scores after masking", "RelativeAttention: ")
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        if self.debug_mode:
            debug_print(attention_probs, "attention_probs", "Attention probabilities after softmax", "RelativeAttention: ")
        
        # Store attention weights if requested
        if self.store_attention:
            self.last_attention_weights = attention_probs.detach()
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        if self.debug_mode:
            debug_print(context, "context", "Context vectors after attention", "RelativeAttention: ")
        
        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        if self.debug_mode:
            debug_print(context, "context_reshaped", "Context after reshaping back", "RelativeAttention: ")
        
        # Final projection
        output = self.out_proj(context)
        if self.debug_mode:
            debug_print(output, "rel_attention_output", "Final relative attention output", "RelativeAttention: ")
        
        return output