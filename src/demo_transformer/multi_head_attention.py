"""Multi-Head Attention implementation with detailed mathematical documentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from .debug_utils import debug_print


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism as described in "Attention Is All You Need" (Vaswani et al., 2017).

    This implementation uses torch.nn.functional.scaled_dot_product_attention for performance
    when `store_attention` is False, and falls back to a manual implementation when it is True
    to allow for introspection of attention weights.

    Args:
        embed_dim: Model dimensionality (d_model in the paper)
        num_heads: Number of parallel attention heads (h in the paper)
        dropout_prob: Dropout probability for attention weights and output projection.
        debug_mode: Whether to print debug information about tensors.
        store_attention: If True, uses a manual implementation to store attention weights.
                         If False, uses a faster, fused implementation.

    Note:
        For maximum performance, this module and any model containing it can be
        wrapped with `torch.compile(model)`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout_prob: float = 0.1,
        debug_mode: bool = False,
        store_attention: bool = False,
    ):
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

        # Linear projections for queries, keys, and values
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection matrix
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.out_dropout = nn.Dropout(dropout_prob)

        self._init_weights()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass implementing multi-head attention.
        Conditionally uses a fused kernel for performance if not storing attention weights.
        """
        # --- Input shape validation ---
        assert (
            query.dim() == 3
        ), f"Query must be a 3D tensor (batch_size, sequence_length, embed_dim), but got {query.dim()}D"
        assert (
            key.dim() == 3
        ), f"Key must be a 3D tensor (batch_size, sequence_length, embed_dim), but got {key.dim()}D"
        assert (
            value.dim() == 3
        ), f"Value must be a 3D tensor (batch_size, sequence_length, embed_dim), but got {value.dim()}D"

        batch_size, seq_len, _ = query.size()

        assert (
            key.size(0) == batch_size and value.size(0) == batch_size
        ), "Batch sizes of query, key, and value must match"

        assert key.size(1) == value.size(1), "Sequence lengths of key and value must match"

        assert (
            query.size(-1) == self.embed_dim
        ), f"Query embed_dim mismatch: expected {self.embed_dim}, got {query.size(-1)}"
        assert (
            key.size(-1) == self.embed_dim
        ), f"Key embed_dim mismatch: expected {self.embed_dim}, got {key.size(-1)}"
        assert (
            value.size(-1) == self.embed_dim
        ), f"Value embed_dim mismatch: expected {self.embed_dim}, got {value.size(-1)}"

        if self.debug_mode:
            debug_print(query, "query_input", "Query input tensor", "Attention: ")
            debug_print(key, "key_input", "Key input tensor", "Attention: ")
            debug_print(value, "value_input", "Value input tensor", "Attention: ")
            if mask is not None:
                debug_print(mask, "attention_mask", "Attention mask tensor", "Attention: ")

        # Step 1: Linear projections
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        if self.debug_mode:
            debug_print(
                Q, "Q_projected", "Query after linear projection Q = query W^Q", "Attention: "
            )
            debug_print(K, "K_projected", "Key after linear projection K = key W^K", "Attention: ")
            debug_print(
                V, "V_projected", "Value after linear projection V = value W^V", "Attention: "
            )

        # Step 2: Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, key.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, value.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        if self.debug_mode:
            debug_print(
                Q, "Q_reshaped", "Query reshaped to [batch, h, seq_len, head_dim]", "Attention: "
            )
            debug_print(
                K, "K_reshaped", "Key reshaped to [batch, h, kv_seq_len, head_dim]", "Attention: "
            )
            debug_print(
                V, "V_reshaped", "Value reshaped to [batch, h, kv_seq_len, head_dim]", "Attention: "
            )

        # Step 3: Prepare attention mask
        # MASKING IN THE TRANSFORMER ARCHITECTURE:
        # A standard Encoder-Decoder Transformer uses three distinct masking scenarios that work together:
        #
        # 1. Encoder Self-Attention (Source Padding Mask):
        #    - When processing the source sequence, a padding mask is used to prevent the encoder from
        #      attending to meaningless padding tokens. The output of this stage (K, V) is passed to the decoder.
        #
        # 2. Decoder Self-Attention (Target Combined Mask):
        #    - When processing the target sequence generated so far, the decoder uses a combined causal
        #      and padding mask. This prevents it from "cheating" by looking at future tokens while also
        #      ignoring padding. The output of this stage is the query (Q) for the next step.
        #
        # 3. Decoder Cross-Attention (Source Padding Mask again):
        #    - The decoder combines its query (Q) from self-attention with the keys (K) and values (V)
        #      from the encoder. It uses the same source padding mask from step 1. This allows the
        #      decoder to look at the entire source sentence to make its prediction, while still
        #      ignoring the source padding. A causal mask is NOT used here.
        #
        # --- MASK DETAILS ---
        #
        # MASK SHAPE & BROADCASTING:
        # The mask is broadcastable to the shape of the attention scores: [batch_size, num_heads, seq_len, kv_seq_len].
        # Common input shapes that are broadcast automatically:
        # - 2D Padding Mask: [batch_size, kv_seq_len]
        # - 3D Combined Mask: [batch_size, seq_len, kv_seq_len]
        #
        # MASK SEMANTICS:
        # - A `True` value or a large negative number (like -inf) indicates a position should be MASKED OUT.
        # - Masked positions have their attention scores set to -∞, making their contribution 0 after softmax.
        final_mask = None
        if mask is not None:
            # Canonicalize mask to a 4D tensor for broadcasting over heads.
            # Accepted input shapes:
            # - 2D: [batch_size, kv_seq_len] -> [batch_size, 1, 1, kv_seq_len]
            # - 3D: [batch_size, seq_len, kv_seq_len] -> [batch_size, 1, seq_len, kv_seq_len]
            # - 4D: [batch_size, num_heads, seq_len, kv_seq_len] (no-op)
            if mask.dim() == 2:
                final_mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                final_mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                final_mask = mask
            else:
                raise ValueError(f"Attention mask must be 2D, 3D, or 4D, but got {mask.dim()}D")

            # By this point, mask should be 4D and ready for broadcasting or direct use.
            assert (
                final_mask.dim() == 4
            ), f"Attention mask must be 2D, 3D, or 4D, but got {final_mask.dim()}D"

            final_mask = final_mask.to(Q.device)

        if self.store_attention:
            # --- Manual Attention Path (for introspection) ---
            # This path is slower but allows storing attention weights for debugging.

            # --- Note on Implementation ---
            # The following block implements scaled dot-product attention manually.
            # While PyTorch 2.0+ offers a fused and highly optimized function
            # (`torch.nn.functional.scaled_dot_product_attention`), this manual
            # implementation is preserved intentionally.
            # The primary reason is to allow for the `store_attention` feature,
            # which provides introspection into the attention weights. The optimized
            # function does not return attention weights, as it often uses algorithms
            # like FlashAttention that do not explicitly materialize the attention matrix.

            # Step 4: Scaled dot-product attention
            # Compute attention scores
            raw_scores = torch.matmul(Q, K.transpose(-2, -1))

            # Scale the scores to prevent softmax overflow
            attention_scores = raw_scores / math.sqrt(self.head_dim)
            # Shape: [batch_size, num_heads, seq_len, kv_seq_len]

            if self.debug_mode:
                debug_print(
                    attention_scores,
                    "attention_scores",
                    f"Attention scores QK^T/√d_k with scaling factor 1/√{self.head_dim}",
                    "Attention: ",
                )

            # Step 5: Apply attention mask (if provided)
            if final_mask is not None:
                # Apply the mask. It can be a boolean mask (True for masked positions)
                # or a float additive mask (0.0 for keep, -inf for mask).
                if final_mask.dtype == torch.bool:
                    attention_scores = attention_scores.masked_fill(final_mask, float("-inf"))
                else:
                    attention_scores = attention_scores + final_mask.to(attention_scores.dtype)

                if self.debug_mode:
                    debug_print(
                        attention_scores,
                        "masked_attention_scores",
                        "Attention scores after masking",
                        "Attention: ",
                    )

            # Step 6: Robust softmax
            attention_probs = torch.softmax(attention_scores, dim=-1)

            # Check for rows that are not finite, which can happen if all scores
            # in a row are -inf after masking. Replace these with zeros.
            is_bad_row = ~torch.isfinite(attention_probs).all(dim=-1, keepdim=True)
            attention_probs = torch.where(
                is_bad_row, torch.zeros_like(attention_probs), attention_probs
            )
            # Shape: [batch_size, num_heads, seq_len, kv_seq_len]

            if self.debug_mode:
                debug_print(
                    attention_probs,
                    "attention_probs",
                    "Attention probabilities after softmax normalization",
                    "Attention: ",
                )

            # Store attention weights for visualization (path is active because self.store_attention is True)
            # Detach from computation graph to avoid affecting gradients
            self.last_attention_weights = attention_probs.detach()

            # Apply attention dropout for regularization
            attention_probs = self.attn_dropout(attention_probs)

            # Step 7: Apply attention to values
            context = torch.matmul(attention_probs, V)
            # Shape: [batch_size, num_heads, seq_len, d_k]

            if self.debug_mode:
                debug_print(
                    context,
                    "context",
                    "Context vectors from attention-weighted values",
                    "Attention: ",
                )

        else:
            # --- Fused Attention Path (for performance) ---
            # This path is faster and more memory-efficient.
            self.last_attention_weights = None
            if final_mask is not None and final_mask.dtype != torch.bool:
                final_mask = final_mask.to(Q.dtype)
                # Note: When using float16/bfloat16, large negative values in the additive mask
                # will saturate to the minimum representable value, effectively behaving like -inf.
                # This can change the semantics compared to float32 where magnitudes are preserved.

            context = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=final_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )

        # Step 7: Concatenate heads and reshape
        # Transpose back to [batch, seq_len, num_heads, d_k] then reshape to [batch, seq_len, d_model]
        # This implements the Concat operation: Concat(head₁, ..., headₕ)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        if self.debug_mode:
            debug_print(
                context,
                "context_reshaped",
                "Concatenated heads reshaped to [batch, seq_len, d_model]",
                "Attention: ",
            )

        # Step 8: Final linear projection and dropout
        # Apply output projection matrix W^O to get final output
        # MultiHead(Q,K,V) = Concat(head₁,...,headₕ)W^O
        output = self.out_proj(context)
        output = self.out_dropout(output)

        if self.debug_mode:
            debug_print(
                output, "attention_output", "Final output after projection W^O", "Attention: "
            )

        return output

    def _init_weights(self):
        """Initialize weights using the standard Xavier/Glorot initialization."""
        # Use the default gain of 1.0 for xavier_uniform_, which is standard for linear layers.
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Initialize biases to zero
        nn.init.zeros_(self.query_proj.bias)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
