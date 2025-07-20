"""Multi-Head Attention implementation."""

import torch
import torch.nn as nn
import math
from typing import Optional

from .debug_utils import debug_print


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    Can be used for self-attention (query=key=value) or cross-attention.

    Supports masking through the mask parameter, which can be:
    - A causal/autoregressive mask (created in the decoder)
    - A padding mask (created in both encoder and decoder)
    - A combined mask (e.g., both causal and padding in decoder self-attention)

    The mask should be a boolean tensor where True values are positions to be masked out
    (set to -inf before softmax). The attention module applies the mask as provided
    without distinguishing its type - the mask creation happens externally.

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
        debug_mode: bool = False,
        store_attention: bool = False,
    ):
        """Initialize the multi-head attention module.

        Args:
            embed_dim: Dimension of the embeddings
            num_heads: Number of attention heads
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
        self.debug_mode = debug_mode
        self.store_attention = store_attention
        self.last_attention_weights = None

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        if self.debug_mode:
            debug_print(query, "query_input", "Query input tensor", "Attention: ")
            debug_print(key, "key_input", "Key input tensor", "Attention: ")
            debug_print(value, "value_input", "Value input tensor", "Attention: ")
            if mask is not None:
                debug_print(mask, "attention_mask", "Attention mask tensor", "Attention: ")

        batch_size, seq_len_q, _ = query.size()
        _, seq_len_k, _ = key.size()

        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        if self.debug_mode:
            debug_print(Q, "Q_projected", "Query after projection", "Attention: ")
            debug_print(K, "K_projected", "Key after projection", "Attention: ")
            debug_print(V, "V_projected", "Value after projection", "Attention: ")

        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        if self.debug_mode:
            debug_print(Q, "Q_reshaped", "Query after reshaping for multi-head", "Attention: ")
            debug_print(K, "K_reshaped", "Key after reshaping for multi-head", "Attention: ")
            debug_print(V, "V_reshaped", "Value after reshaping for multi-head", "Attention: ")

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if self.debug_mode:
            debug_print(
                attention_scores,
                "attention_scores",
                "Raw attention scores before masking",
                "Attention: ",
            )

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, float("-inf"))
            if self.debug_mode:
                debug_print(
                    attention_scores,
                    "masked_attention_scores",
                    "Attention scores after masking",
                    "Attention: ",
                )

        attention_probs = torch.softmax(attention_scores, dim=-1)
        if self.debug_mode:
            debug_print(
                attention_probs,
                "attention_probs",
                "Attention probabilities after softmax",
                "Attention: ",
            )

        # Store attention weights if requested
        # This allows for later visualization of attention patterns and model interpretability
        # The detach() call prevents these stored weights from participating in backpropagation
        if self.store_attention:
            self.last_attention_weights = (
                attention_probs.detach()
            )  # Store a copy without gradient tracking

        context = torch.matmul(attention_probs, V)
        if self.debug_mode:
            debug_print(context, "context", "Context vectors after attention", "Attention: ")

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        if self.debug_mode:
            debug_print(context, "context_reshaped", "Context after reshaping back", "Attention: ")

        output = self.out_proj(context)
        if self.debug_mode:
            debug_print(output, "attention_output", "Final attention output", "Attention: ")

        return output



