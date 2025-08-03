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

        # Additional projection for relative positions - this generates the R in QR^T
        # In the formula Attention(Q, K, V) = softmax(QK^T/√d + QR^T/√d)V:
        # - rel_pos_encoding creates raw position embeddings
        # - pos_key_proj transforms them into R (the projected relative position keys)
        # This is analogous to how key_proj transforms content into K
        # The bias=False setting follows the original implementation and helps prevent overfitting
        self.pos_key_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # Debug mode and attention storage
        self.debug_mode = debug_mode
        self.store_attention = store_attention
        self.last_attention_weights = None

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Shift the relative logits to align them properly.
        
        This function converts relative position scores from a "full relative" representation
        to a "query-key aligned" representation needed for attention computation.
        
        Example with seq_len=3:
        Input x has shape [batch, heads, 3, 5] representing relative positions [-2, -1, 0, +1, +2]
        
        Concrete example - if input tensor x[0,0,:,:] contains:
        [[a1, a2, a3, a4, a5],   # Query pos 0: scores for rel pos [-2,-1,0,+1,+2]
         [b1, b2, b3, b4, b5],   # Query pos 1: scores for rel pos [-2,-1,0,+1,+2]
         [c1, c2, c3, c4, c5]]   # Query pos 2: scores for rel pos [-2,-1,0,+1,+2]
        
        After shifting, output result[0,0,:,:] will be:
        [[a3, a4, a5],   # Query pos 0 to key pos [0,1,2]: rel pos [0-0,1-0,2-0] = [0,+1,+2] → a3,a4,a5
         [b2, b3, b4],   # Query pos 1 to key pos [0,1,2]: rel pos [0-1,1-1,2-1] = [-1,0,+1] → b2,b3,b4
         [c1, c2, c3]]   # Query pos 2 to key pos [0,1,2]: rel pos [0-2,1-2,2-2] = [-2,-1,0] → c1,c2,c3
        
        This transforms from "all possible relative positions" to "actual query-key pairs"

        Args:
            x: Input tensor [batch_size, num_heads, seq_len, 2*seq_len-1]

        Returns:
            Shifted tensor [batch_size, num_heads, seq_len, seq_len]
        """
        # Input tensor dimensions: [batch_size, num_heads, seq_len, 2*seq_len-1]
        # Example: for seq_len=3, last dim has 5 positions for rel_pos [-2,-1,0,+1,+2]
        batch_size, num_heads, seq_len, _ = x.size()
        total_len = 2 * seq_len - 1

        # Create the final output tensor: [batch_size, num_heads, seq_len, seq_len]
        # This will contain only the valid query-key attention scores
        result = torch.zeros((batch_size, num_heads, seq_len, seq_len), device=x.device, dtype=x.dtype)
        
        # The center of the input tensor (position seq_len-1) corresponds to relative position 0
        # For seq_len=3: center_pos=2, so input[..., 2] = rel_pos 0
        center_pos = seq_len - 1
        
        for i in range(seq_len):
            # For query position i, we need relative positions [i-0, i-1, i-2, ...] = [i, i-1, i-2, ...]
            # These correspond to input positions [center_pos-i, center_pos-i+1, center_pos-i+2, ...]
            # Example: query_pos=0 needs rel_pos [0,-1,-2] → input positions [2,1,0]
            #          query_pos=1 needs rel_pos [1,0,-1] → input positions [1,2,3]
            start_pos = center_pos - i
            # Slice exactly seq_len positions: [start_pos : start_pos + seq_len]
            # This gives us [batch, heads, query_i, key_0:seq_len] attention scores
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
            mask: Optional attention mask [batch_size, num_heads, seq_len_q, seq_len_k]
                 - Must match attention_scores shape for element-wise masking
                 - True values indicate positions to MASK OUT (set to -∞)
                 - False values indicate positions to ATTEND TO (keep scores)
                 - Can broadcast: [batch, 1, seq_len_q, seq_len_k] or [1, 1, seq_len_q, seq_len_k]
                 - Mask types by transformer component:
                   * Padding mask: encoder + decoder (all attention types)
                   * Causal mask: decoder self-attention only
                   * Combined mask: decoder cross-attention (both sequence paddings)

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

        # Step 1: Get relative position embeddings for all possible relative distances
        # For seq_len_k=3, this creates embeddings for relative positions [-2, -1, 0, +1, +2]
        # Shape: [2*seq_len_k-1, head_dim] = [5, head_dim]
        # These are learnable embeddings that encode the "meaning" of each relative distance
        rel_pos_emb = self.rel_pos_encoding(seq_len_k)
        if self.debug_mode:
            debug_print(
                rel_pos_emb, "rel_pos_emb", "Relative position embeddings", "RelativeAttention: "
            )

        # Step 2: Project relative position embeddings into "key space"
        # This transforms position embeddings so they can be compared with queries
        # Just like content keys are projected, position embeddings need projection too
        # Shape remains: [2*seq_len_k-1, head_dim]
        rel_pos_key = self.pos_key_proj(rel_pos_emb)
        if self.debug_mode:
            debug_print(
                rel_pos_key,
                "rel_pos_key",
                "Projected relative position keys",
                "RelativeAttention: ",
            )

        # Step 3: Compute standard content-content attention (like regular attention)
        # This captures semantic similarity: "cat" attending to "sat" based on meaning
        # Q: [batch, heads, seq_len_q, head_dim], K: [batch, heads, seq_len_k, head_dim]
        # Result: [batch, heads, seq_len_q, seq_len_k]
        content_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.debug_mode:
            debug_print(
                content_scores,
                "content_scores",
                "Content-content attention scores",
                "RelativeAttention: ",
            )

        # Step 4: Compute content-position attention (the key innovation)
        # This captures positional preferences: "verbs often attend to subjects 2 positions back"
        # Add batch and head dimensions to rel_pos_key for broadcasting
        # rel_pos_key: [2*seq_len_k-1, head_dim] → [1, 1, 2*seq_len_k-1, head_dim]
        rel_pos_key = rel_pos_key.unsqueeze(0).unsqueeze(0)
        
        # Q: [batch, heads, seq_len_q, head_dim] × rel_pos_key^T: [1, 1, head_dim, 2*seq_len_k-1]
        # Result: [batch, heads, seq_len_q, 2*seq_len_k-1]
        # Each query gets scores for ALL possible relative positions [-2,-1,0,+1,+2]
        position_scores = torch.matmul(Q, rel_pos_key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.debug_mode:
            debug_print(
                position_scores,
                "position_scores",
                "Content-position attention scores",
                "RelativeAttention: ",
            )

        # Shift and slice position scores to align them
        # Input: [batch, heads, seq_len_q, 2*seq_len_k-1] → Output: [batch, heads, seq_len_q, seq_len_k]
        # This transforms from "all possible relative positions" to "actual query-key pairs"
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