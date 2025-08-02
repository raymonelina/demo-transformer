"""Multi-Head Attention implementation with detailed mathematical documentation."""

import torch
import torch.nn as nn
import math
from typing import Optional

from .debug_utils import debug_print


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention Is All You Need" (Vaswani et al., 2017).
    
    The multi-head attention mechanism allows the model to jointly attend to information
    from different representation subspaces at different positions. Mathematically:
    
    MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O
    where headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
    
    And the scaled dot-product attention is defined as:
    Attention(Q, K, V) = softmax(QK^T / √dₖ)V
    
    Can be used for:
    - Self-attention (query=key=value) in encoder layers
    - Cross-attention in decoder layers (query from decoder, key/value from encoder)
    - Masked self-attention in decoder layers (with causal mask)

    Args:
        embed_dim: Model dimensionality (d_model in the paper)
        num_heads: Number of parallel attention heads (h in the paper)
        debug_mode: Whether to print debug information about tensors
        store_attention: Whether to store attention weights for visualization
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        debug_mode: bool = False,
        store_attention: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})"
            )

        self.embed_dim = embed_dim  # d_model
        self.num_heads = num_heads  # h
        self.head_dim = embed_dim // num_heads  # d_k = d_v = d_model / h
        self.debug_mode = debug_mode
        self.store_attention = store_attention
        self.last_attention_weights = None

        # Linear projections for queries, keys, and values
        # These implement the learned parameter matrices W^Q, W^K, W^V
        self.query_proj = nn.Linear(embed_dim, embed_dim)  # W^Q ∈ ℝ^(d_model × d_model)
        self.key_proj = nn.Linear(embed_dim, embed_dim)    # W^K ∈ ℝ^(d_model × d_model)
        self.value_proj = nn.Linear(embed_dim, embed_dim)  # W^V ∈ ℝ^(d_model × d_model)

        # Output projection matrix W^O ∈ ℝ^(d_model × d_model)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass implementing multi-head attention.
        
        Mathematical formulation:
        1. Linear projections: Q = XW^Q, K = XW^K, V = XW^V
        2. Reshape for multi-head: Split d_model into h heads of dimension d_k
        3. Scaled dot-product attention per head: Attention(Q, K, V) = softmax(QK^T/√d_k)V
        4. Concatenate heads and apply output projection
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, kv_seq_len, d_model]  
            value: Value tensor [batch_size, kv_seq_len, d_model]
            mask: Optional attention mask [batch_size, num_heads, seq_len, kv_seq_len]
                 True values indicate positions to mask (set to -∞)
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        if self.debug_mode:
            debug_print(query, "query_input", "Query input tensor", "Attention: ")
            debug_print(key, "key_input", "Key input tensor", "Attention: ")
            debug_print(value, "value_input", "Value input tensor", "Attention: ")
            if mask is not None:
                debug_print(mask, "attention_mask", "Attention mask tensor", "Attention: ")

        batch_size, seq_len, _ = query.size()
        # Note: Key and value must always have the same sequence length (they represent the same input)
        # Query can have a different length (e.g., in cross-attention: target vs source lengths)
        _, kv_seq_len, _ = key.size()

        # Step 1: Linear projections
        # Apply learned linear transformations to input embeddings
        # Q = XW^Q, K = XW^K, V = XW^V where X ∈ ℝ^(batch × seq_len × d_model)
        Q = self.query_proj(query)  # [batch_size, seq_len, d_model]
        K = self.key_proj(key)      # [batch_size, kv_seq_len, d_model]
        V = self.value_proj(value)  # [batch_size, kv_seq_len, d_model]

        if self.debug_mode:
            debug_print(Q, "Q_projected", "Query after linear projection Q = XW^Q", "Attention: ")
            debug_print(K, "K_projected", "Key after linear projection K = XW^K", "Attention: ")
            debug_print(V, "V_projected", "Value after linear projection V = XW^V", "Attention: ")

        # Step 2: Reshape for multi-head attention
        # Split d_model dimension into h heads, each of dimension d_k = d_model/h
        # Reshape from [batch, seq_len, d_model] to [batch, h, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.debug_mode:
            debug_print(Q, "Q_reshaped", "Query reshaped to [batch, h, seq_len, d_k]", "Attention: ")
            debug_print(K, "K_reshaped", "Key reshaped to [batch, h, kv_seq_len, d_k]", "Attention: ")
            debug_print(V, "V_reshaped", "Value reshaped to [batch, h, kv_seq_len, d_k]", "Attention: ")

        # Step 3: Scaled dot-product attention
        # Compute attention scores: QK^T / √d_k
        # The scaling factor 1/√d_k prevents the dot products from growing too large,
        # which would push the softmax function into regions with extremely small gradients
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # Shape: [batch_size, num_heads, seq_len, kv_seq_len]

        if self.debug_mode:
            debug_print(
                attention_scores,
                "attention_scores",
                f"Attention scores QK^T/√d_k with scaling factor 1/√{self.head_dim}",
                "Attention: ",
            )

        # Step 4: Apply attention mask (if provided)
        # Mask prevents attention to certain positions by setting scores to -∞
        # After softmax, these become 0, effectively removing their contribution
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))
            if self.debug_mode:
                debug_print(
                    attention_scores,
                    "masked_attention_scores",
                    "Attention scores after masking (masked positions set to -∞)",
                    "Attention: ",
                )

        # Step 5: Apply softmax to get attention probabilities
        # softmax(QK^T/√d_k) normalizes scores to probabilities that sum to 1
        # This creates a probability distribution over the key positions
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # Shape: [batch_size, num_heads, seq_len, kv_seq_len]
        
        if self.debug_mode:
            debug_print(
                attention_probs,
                "attention_probs",
                "Attention probabilities after softmax normalization",
                "Attention: ",
            )

        # Store attention weights for visualization if requested
        # Detach from computation graph to avoid affecting gradients
        if self.store_attention:
            self.last_attention_weights = attention_probs.detach()

        # Step 6: Apply attention to values
        # Weighted sum of values: softmax(QK^T/√d_k)V
        # Each output position is a weighted combination of all value vectors
        context = torch.matmul(attention_probs, V)
        # Shape: [batch_size, num_heads, seq_len, d_k]
        
        if self.debug_mode:
            debug_print(context, "context", "Context vectors from attention-weighted values", "Attention: ")

        # Step 7: Concatenate heads and reshape
        # Transpose back to [batch, seq_len, num_heads, d_k] then reshape to [batch, seq_len, d_model]
        # This implements the Concat operation: Concat(head₁, ..., headₕ)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        if self.debug_mode:
            debug_print(context, "context_reshaped", "Concatenated heads reshaped to [batch, seq_len, d_model]", "Attention: ")

        # Step 8: Final linear projection
        # Apply output projection matrix W^O to get final output
        # MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
        output = self.out_proj(context)
        
        if self.debug_mode:
            debug_print(output, "attention_output", "Final output after projection W^O", "Attention: ")

        return output