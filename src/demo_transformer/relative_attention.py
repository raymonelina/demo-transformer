"""
Relative Multi-Head Attention implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .debug_utils import debug_print
from .relative_positional_encoding import RelativePositionalEncoding


class RelativeMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with relative positional encoding.

    This implementation is based on the paper "Self-Attention with Relative Position Representations"
    (Shaw et al., 2018, https://arxiv.org/abs/1803.02155). The key innovation is incorporating
    explicit relative position information directly into the self-attention mechanism, rather than
    adding absolute position encodings to input embeddings.

    Key advantages over standard positional encoding:
    1. Better generalization to sequence lengths not seen during training
    2. More effective modeling of fine-grained relative position relationships
    3. Improved performance on tasks requiring precise understanding of token relationships

    The approach works by:
    - Computing standard content-content attention (query-key interactions)
    - Adding content-position attention (query interaction with relative position keys)
    - Using a specialized shifting mechanism to align relative positions correctly

    Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T/√d + QR^T/√d)V
    where R represents the relative position embeddings.

    When store_attention=True, the attention weights (after softmax) are saved in
    last_attention_weights for later visualization or analysis. This is useful for:
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
        debug_mode: bool = False,
        store_attention: bool = False,
    ):
        """
        Initialize relative multi-head attention.

        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length for relative positions. This defines the range of
                relative positions that can be represented (-max_seq_len+1 to max_seq_len-1).
                As per Shaw et al. (2018), this clipping of relative positions helps control
                model complexity while still capturing important local context.
            debug_mode: Whether to print debug information about tensors
            store_attention: Whether to store attention weights for visualization
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

        # Relative position encoding - generates embeddings for relative positions
        # This implements the core idea from Shaw et al. (2018) where relative positions from
        # -(max_seq_len-1) to (max_seq_len-1) are encoded as learnable embeddings
        # These embeddings capture the relative distance between tokens in the sequence
        self.rel_pos_encoding = RelativePositionalEncoding(self.head_dim, max_seq_len)

        # Additional projection for relative positions - transforms position embeddings into key space
        # This is analogous to the key projection in standard attention but specifically for positions
        # The bias=False setting follows the original implementation and helps prevent overfitting
        # This projection is critical for the model to learn how to effectively use position information
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
        total_len = 2 * seq_len - 1

        # Create a more robust implementation that doesn't rely on reshaping
        # This avoids potential issues with tensor shapes
        # We want to shift the scores so that scores corresponding to position 0 are at the center
        
        # First create the final output tensor directly
        result = torch.zeros((batch_size, num_heads, seq_len, seq_len), device=x.device, dtype=x.dtype)
        
        # Fill it with the appropriate values from the original tensor
        # The center of the original tensor (position seq_len-1) corresponds to relative position 0
        center_pos = seq_len - 1
        
        for i in range(seq_len):
            # For each position in the target sequence, copy the appropriate slice
            # from the original tensor, shifted according to relative position
            start_pos = center_pos - i
            result[:, :, i, :] = x[:, :, i, start_pos:start_pos + seq_len]
            
        return result

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
                debug_print(
                    mask, "rel_attention_mask", "Attention mask tensor", "RelativeAttention: "
                )

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
            debug_print(
                Q, "rel_Q_reshaped", "Query after reshaping for multi-head", "RelativeAttention: "
            )
            debug_print(
                K, "rel_K_reshaped", "Key after reshaping for multi-head", "RelativeAttention: "
            )
            debug_print(
                V, "rel_V_reshaped", "Value after reshaping for multi-head", "RelativeAttention: "
            )

        # Get relative position embeddings [2*seq_len_k-1, head_dim]
        rel_pos_emb = self.rel_pos_encoding(seq_len_k)
        if self.debug_mode:
            debug_print(
                rel_pos_emb, "rel_pos_emb", "Relative position embeddings", "RelativeAttention: "
            )

        # Project relative position embeddings
        rel_pos_key = self.pos_key_proj(rel_pos_emb)
        if self.debug_mode:
            debug_print(
                rel_pos_key,
                "rel_pos_key",
                "Projected relative position keys",
                "RelativeAttention: ",
            )

        # Content-content attention
        content_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.debug_mode:
            debug_print(
                content_scores,
                "content_scores",
                "Content-content attention scores",
                "RelativeAttention: ",
            )

        # Content-position attention
        # [batch_size, num_heads, seq_len_q, 2*seq_len_k-1]
        rel_pos_key = rel_pos_key.unsqueeze(0).unsqueeze(0)
        position_scores = torch.matmul(Q, rel_pos_key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.debug_mode:
            debug_print(
                position_scores,
                "position_scores",
                "Content-position attention scores",
                "RelativeAttention: ",
            )

        # Shift and slice position scores to align them
        rel_position_scores = self._rel_shift(position_scores)
        if self.debug_mode:
            debug_print(
                rel_position_scores,
                "rel_position_scores",
                "Shifted position scores",
                "RelativeAttention: ",
            )

        # Combine content and position scores
        attention_scores = content_scores + rel_position_scores
        if self.debug_mode:
            debug_print(
                attention_scores,
                "combined_attention_scores",
                "Combined attention scores",
                "RelativeAttention: ",
            )

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))
            if self.debug_mode:
                debug_print(
                    attention_scores,
                    "masked_attention_scores",
                    "Attention scores after masking",
                    "RelativeAttention: ",
                )

        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        if self.debug_mode:
            debug_print(
                attention_probs,
                "attention_probs",
                "Attention probabilities after softmax",
                "RelativeAttention: ",
            )

        # Store attention weights if requested
        # This allows for later visualization of attention patterns and model interpretability
        # The detach() call prevents these stored weights from participating in backpropagation
        if self.store_attention:
            self.last_attention_weights = (
                attention_probs.detach()
            )  # Store a copy without gradient tracking

        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        if self.debug_mode:
            debug_print(
                context, "context", "Context vectors after attention", "RelativeAttention: "
            )

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        if self.debug_mode:
            debug_print(
                context, "context_reshaped", "Context after reshaping back", "RelativeAttention: "
            )

        # Final projection
        output = self.out_proj(context)
        if self.debug_mode:
            debug_print(
                output,
                "rel_attention_output",
                "Final relative attention output",
                "RelativeAttention: ",
            )

        return output