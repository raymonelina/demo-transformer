"""Transformer model implementation."""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, List, Tuple
import matplotlib.pyplot as plt

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .config import TransformerConfig
from .debug_utils import debug_print
from .visualization import plot_attention_weights, plot_embeddings_pca, plot_attention_heads


class Transformer(nn.Module):
    """
    A complete Encoder-Decoder Transformer model for sequence-to-sequence tasks.
    """

    def __init__(
        self,
        config: Union[TransformerConfig, Dict[str, Any]],
    ):
        """
        Initialize a Transformer model.

        Args:
            config: A TransformerConfig object or a dictionary with configuration parameters
        """
        super().__init__()

        # Convert dict to config if needed
        if isinstance(config, dict):
            config = TransformerConfig(**config)

        self.config = config

        # Store attention weights for visualization
        self.store_attention = getattr(config, "store_attention", False)

        # Create encoder and decoder
        self.encoder = TransformerEncoder(
            vocab_size=config.src_vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.num_encoder_layers,
            max_seq_len=config.max_seq_len,
            dropout_rate=config.dropout_rate,
            pre_norm=config.pre_norm,
            use_relative_pos=config.use_relative_pos,
            use_rope=config.use_rope,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            debug_mode=config.debug_mode,
            store_attention=self.store_attention,
        )

        self.decoder = TransformerDecoder(
            vocab_size=config.tgt_vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_layers=config.num_decoder_layers,
            max_seq_len=config.max_seq_len,
            dropout_rate=config.dropout_rate,
            pre_norm=config.pre_norm,
            use_relative_pos=config.use_relative_pos,
            use_rope=config.use_rope,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
            debug_mode=config.debug_mode,
            store_attention=self.store_attention,
        )

        # ═══════════════════════════════════════════════════════════════════════════════
        # 🔗 WEIGHT TYING: Share Input/Output Embeddings (Parameter Efficiency)
        # ═══════════════════════════════════════════════════════════════════════════════
        # 
        # WHAT: Input embedding and output projection layers share the same weight matrix
        # WHY: Reduces parameters by ~50% and often improves performance
        # 
        # ACADEMIC PAPERS:
        # • "Using the Output Embedding to Improve Language Models" (Press & Wolf, 2017)
        # • "Attention Is All You Need" (Vaswani et al., 2017) - mentions weight tying
        # • "Tying Word Vectors and Word Classifiers" (Inan et al., 2017)
        # 
        # HOW IT WORKS:
        # Normal case (separate matrices):
        #   input_embedding:    [vocab_size, embed_dim]  - maps token_id → vector
        #   output_projection:  [embed_dim, vocab_size]  - maps vector → logits
        #   Total params: vocab_size × embed_dim × 2
        # 
        # Weight tying (shared matrix):
        #   Both layers use the SAME matrix (transposed for output)
        #   Total params: vocab_size × embed_dim × 1  (50% reduction!)
        # 
        # INTUITION:
        # If word "cat" has embedding [0.1, 0.5, -0.2], then when predicting "cat",
        # the model should output high logit for "cat". Weight tying enforces this
        # symmetry: similar words have similar embeddings AND similar output weights.
        # 
        # REQUIREMENTS:
        # • Source and target vocabularies must be the same size
        # • Only works for encoder-decoder with shared vocabulary (e.g., same language)
        # 
        if config.weight_tying and config.src_vocab_size == config.tgt_vocab_size:
            # Share the weight matrix between input embedding and output projection
            # decoder.token_embedding.weight: [tgt_vocab_size, embed_dim]
            # decoder.output_projection.weight: [tgt_vocab_size, embed_dim] (will be transposed in forward)
            self.decoder.output_projection.weight = self.decoder.token_embedding.weight
            
            # Note: PyTorch Linear layer automatically transposes weight matrix during forward pass:
            # output = input @ weight.T + bias
            # So embedding weight [vocab_size, embed_dim] becomes [embed_dim, vocab_size] for projection

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer model - USED FOR TRAINING with teacher forcing.
        
        ═══════════════════════════════════════════════════════════════════════════════
        🎯 MASKS vs SPECIAL TOKENS: Why We Need Both
        ═══════════════════════════════════════════════════════════════════════════════
        
        MASKS handle STRUCTURAL constraints:
        • Padding masks: "Ignore these positions (they're padding tokens)"
        • Causal masks: "Don't look at future positions (prevent cheating)"
        
        SPECIAL TOKENS provide SEMANTIC meaning:
        • SOS: "Start generating here" (provides initial context)
        • EOS: "Stop generating here" (signals end of sequence)
        • PAD: "This position is meaningless" (just filler for batching)
        
        🔑 KEY INSIGHT: SOS and EOS are REAL tokens that must be visible!
        Masks only hide PAD tokens and future positions, never SOS/EOS.
        
        Example sequence: [SOS, "J'aime", "les", "chats", EOS, PAD, PAD]
        Token IDs:        [  1,      4,     5,      6,    2,   0,   0]
        
        PADDING MASK (what to ignore):
        [False, False, False, False, False, True, True]
         SOS✓   word✓  word✓  word✓  EOS✓   PAD✗  PAD✗
        
        CAUSAL MASK (prevent looking ahead - lower triangular):
        Position 0: Can see [SOS] only
        Position 1: Can see [SOS, "J'aime"] only
        Position 2: Can see [SOS, "J'aime", "les"] only
        Position 3: Can see [SOS, "J'aime", "les", "chats"] only
        Position 4: Can see [SOS, "J'aime", "les", "chats", EOS] only
        
        ═══════════════════════════════════════════════════════════════════════════════
        🎓 TEACHER FORCING Explained
        ═══════════════════════════════════════════════════════════════════════════════
        
        During training, we feed the correct target sequence as input, rather than using
        the model's own predictions. This allows parallel processing of all positions.
        
        Example - Training to translate "I love cats" → "J'aime les chats":
        
        ENCODER (Understanding the source):
        - Input (src_ids):  ["I", "love", "cats"]     # English source (NO SOS/EOS needed!)
        - Output: encoder_output                      # Understanding of English meaning
        - Why no SOS/EOS: Encoder just needs to understand meaning, not generate
        
        DECODER (Generating the target):
        Complete sequence: [SOS, "J'aime", "les", "chats", EOS]  # Full target sequence
        - Input (tgt_ids):  [SOS, "J'aime", "les", "chats"]     # Decoder input (all except last)
        - Target labels:    ["J'aime", "les", "chats", EOS]     # What we want to predict (all except first)
        - Uses: encoder_output + tgt_ids                        # English meaning + French context
        - Why SOS/EOS: Decoder needs to know when to start (SOS) and stop (EOS) generating
        
        🔑 KEY: Input is target "shifted right" - we predict the NEXT token at each position
        
        📝 NOTE: Some tasks DO add special tokens to encoder input:
        - Classification: [CLS] + tokens + [SEP] (BERT-style)
        - But for translation: just raw source tokens work fine
        
        ═══════════════════════════════════════════════════════════════════════════════
        📋 SUMMARY: Each Training Sample Contains 3 Components
        ═══════════════════════════════════════════════════════════════════════════════
        
        For EVERY training example, we need these 3 pieces:
        
        1️⃣ ENCODER INPUT:  ["I", "love", "cats"]                    # Source sentence
        2️⃣ DECODER INPUT:  [SOS, "J'aime", "les", "chats"]          # Target shifted right
        3️⃣ DECODER TARGET: ["J'aime", "les", "chats", EOS]          # What to predict
        
        The decoder input and target are the SAME sequence, just shifted by one position!
        This is the essence of teacher forcing - we give the model the correct previous
        tokens and ask it to predict the next token.
        
        ═══════════════════════════════════════════════════════════════════════════════
        🚀 INFERENCE: How the Model Actually Generates Text
        ═══════════════════════════════════════════════════════════════════════════════
        
        🔄 AUTOREGRESSIVE LOOP: Model generates one token at a time, using its own predictions
        
        🏗️ SETUP PHASE (Done Once):
        - Encoder input: ["I", "love", "cats"] → encoder_output [batch, 3, embed_dim]
        - This encoder_output contains the "meaning" of the English sentence
        - It's computed ONCE and reused for ALL decoding steps (efficiency!)
        
        🔄 GENERATION LOOP (Repeated Until EOS):
        
        Step 1: Generate first word
        - Decoder input: [SOS]
        - Cross-attention: Decoder queries encoder_output to understand English meaning
        - Model thinks: "Given English 'I love cats', what's the first French word?"
        - Model predicts: "J'aime" (logits → argmax/sampling)
        - Current sequence: [SOS, "J'aime"]
        
        Step 2: Generate second word  
        - Decoder input: [SOS, "J'aime"] (using model's own "J'aime"!)
        - Self-attention: Decoder looks at its own [SOS, "J'aime"] for French context
        - Cross-attention: Decoder queries encoder_output to understand English meaning
        - Model thinks: "Given English 'I love cats' + French 'J'aime', what's next?"
        - Model predicts: "les" (based on English meaning + French context)
        - Current sequence: [SOS, "J'aime", "les"]
        
        Step 3: Generate third word
        - Decoder input: [SOS, "J'aime", "les"] (using model's own words!)
        - Self-attention: Decoder looks at its own [SOS, "J'aime", "les"] for French context
        - Cross-attention: Decoder queries encoder_output to understand English meaning
        - Model thinks: "Given English 'I love cats' + French 'J'aime les', what's next?"
        - Model predicts: "chats" (based on English meaning + French context)
        - Current sequence: [SOS, "J'aime", "les", "chats"]
        
        Step 4: Generate stop signal
        - Decoder input: [SOS, "J'aime", "les", "chats"] (using model's own words!)
        - Self-attention: Decoder looks at its own complete French sequence
        - Cross-attention: Decoder queries encoder_output to check if translation is complete
        - Model thinks: "Given English 'I love cats' + French 'J'aime les chats', am I done?"
        - Model predicts: EOS (decides translation is complete)
        - Final output: "J'aime les chats" (remove SOS/EOS for user)
        
        🔑 KEY INSIGHT: Encoder Output is the "Memory" of Source Meaning
        - encoder_output = ALL source tokens encoded as vectors [batch, src_len, embed_dim]
        - At EVERY step, decoder sees ALL encoder outputs via cross-attention
        - Cross-attention computes weighted sum of ALL encoder outputs
        - Attention weights determine how much each source position contributes
        
        🧠 HOW CROSS-ATTENTION WORKS:
        - Decoder position creates Query vector from current French context
        - ALL encoder positions provide Key and Value vectors
        - Attention scores = Query · Keys (how relevant each English position is)
        - Final output = weighted sum of ALL Values (blended English information)
        - Result: Each French token gets information from ALL English tokens
        
        🚨 KEY DIFFERENCES from Training:
        - SEQUENTIAL: One token at a time (slow)
        - AUTOREGRESSIVE: Uses its own predictions (can compound errors)
        - NO TEACHER: No correct answers provided
        - EXPOSURE BIAS: Trained on correct tokens, but uses its own at inference
        - ENCODER REUSE: Same encoder_output used for all steps (efficient!)
        
        Training: Uses CORRECT French words (fast, parallel, stable)
        Inference: Uses MODEL'S OWN French words (slow, sequential, error-prone)
        
        ═══════════════════════════════════════════════════════════════════════════════
        🛡️ MASKING'S ROLE in Teacher Forcing
        ═══════════════════════════════════════════════════════════════════════════════
        
        Even though we give the model the decoder input [SOS, "J'aime", "les", "chats"],
        masking prevents cheating by blocking future tokens:
        
        Position 0 (predicting "J'aime"):
        - Can see: [SOS] ✓ (SOS provides "start French" context)
        - MASKED: ["J'aime", "les", "chats"] ✗ (blocked by causal mask)
        
        Position 1 (predicting "les"):
        - Can see: [SOS, "J'aime"] ✓ (SOS + previous word context)
        - MASKED: ["les", "chats"] ✗ (blocked by causal mask)
        
        Position 2 (predicting "chats"):
        - Can see: [SOS, "J'aime", "les"] ✓ (SOS + previous words context)
        - MASKED: ["chats"] ✗ (blocked by causal mask)
        
        Position 3 (predicting EOS):
        - Can see: [SOS, "J'aime", "les", "chats"] ✓ (full context to decide "stop here")
        - Target: EOS (not in input, but what we want to predict)
        
        🚨 CRITICAL: SOS and EOS are NEVER masked by padding masks!
        - SOS provides essential starting context ("begin French generation")
        - EOS must be predicted to learn when to stop
        - Only PAD tokens get masked by padding masks
        - Only future positions get masked by causal masks
        
        Without masking, the model would cheat by looking at future answers!
        Teacher forcing makes training faster (parallel) and more stable (uses correct tokens).

        Args:
            src_ids: Source token IDs [batch_size, src_seq_len]
            tgt_ids: Complete target token IDs [batch_size, tgt_seq_len]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            tgt_padding_mask: Target padding mask [batch_size, 1, tgt_seq_len, tgt_seq_len]

        Returns:
            Decoder logits [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # 📊 TENSOR DIMENSIONS THROUGHOUT FORWARD PASS
        # ============================================
        # Input tensors:
        #   src_ids: [batch_size, src_seq_len] - Source token IDs (e.g., [2, 5] for batch=2, src_len=5)
        #   tgt_ids: [batch_size, tgt_seq_len] - Target token IDs (e.g., [2, 7] for batch=2, tgt_len=7)
        #   src_padding_mask: [batch_size, 1, 1, src_seq_len] - Source mask (e.g., [2, 1, 1, 5])
        #   tgt_padding_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] - Target mask (e.g., [2, 1, 7, 7])
        
        if hasattr(self.config, "debug_mode") and self.config.debug_mode:
            debug_print(src_ids, "src_ids", "Source token IDs", "Transformer: ")
            debug_print(tgt_ids, "tgt_ids", "Target token IDs", "Transformer: ")
            if src_padding_mask is not None:
                debug_print(
                    src_padding_mask, "src_padding_mask", "Source padding mask", "Transformer: "
                )
            if tgt_padding_mask is not None:
                debug_print(
                    tgt_padding_mask, "tgt_padding_mask", "Target padding mask", "Transformer: "
                )

        # 🔄 ENCODER PASS: src_ids → encoder_output
        # Input:  src_ids [batch_size, src_seq_len]
        # Output: encoder_output [batch_size, src_seq_len, embed_dim]
        # Example: [2, 5] → [2, 5, 512] (each source token becomes a 512-dim vector)
        encoder_output = self.encoder(src_ids, src_padding_mask)

        if hasattr(self.config, "debug_mode") and self.config.debug_mode:
            debug_print(encoder_output, "encoder_output", "Encoder output tensor", "Transformer: ")

        # 🔄 DECODER PASS: tgt_ids + encoder_output → decoder_logits
        # Input:  tgt_ids [batch_size, tgt_seq_len], encoder_output [batch_size, src_seq_len, embed_dim]
        # Output: decoder_logits [batch_size, tgt_seq_len, tgt_vocab_size]
        # Example: [2, 7] + [2, 5, 512] → [2, 7, 32000] (each target position gets vocab-sized logits)
        decoder_logits = self.decoder(tgt_ids, encoder_output, src_padding_mask, tgt_padding_mask)

        if hasattr(self.config, "debug_mode") and self.config.debug_mode:
            debug_print(decoder_logits, "decoder_logits", "Decoder output logits", "Transformer: ")

        # 📤 FINAL OUTPUT: decoder_logits [batch_size, tgt_seq_len, tgt_vocab_size]
        # Each position in target sequence gets a probability distribution over vocabulary
        # Example: [2, 7, 32000] means 2 sequences, 7 positions each, 32000 possible tokens per position
        return decoder_logits

    def encode(
        self, src_ids: torch.Tensor, src_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode the source sequence - USED FOR INFERENCE (Step 1 of 2).
        
        This method encodes the source sequence once and returns the encoder output,
        which can be reused for multiple decoding steps during autoregressive generation.
        
        Inference usage:
        encoder_output = model.encode(src_ids, src_padding_mask)  # Call once
        # Then use encoder_output multiple times in decode() for token generation
        
        Args:
            src_ids: Source token IDs [batch_size, src_seq_len]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            
        Returns:
            Encoder output [batch_size, src_seq_len, embed_dim]
        """
        return self.encoder(src_ids, src_padding_mask)

    def decode(
        self,
        tgt_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode with target sequence and encoder output - USED FOR INFERENCE (Step 2 of 2).
        
        This method takes a partial target sequence and generates predictions for the next token.
        Called iteratively during autoregressive generation, where the sequence grows one token
        at a time: [SOS] → [SOS, token1] → [SOS, token1, token2] → ...
        
        Inference usage:
        for step in range(max_len):
            current_seq = [SOS, generated_token1, generated_token2, ...]  # Growing sequence
            logits = model.decode(current_seq, encoder_output, src_padding_mask)
            next_token = argmax(logits[:, -1, :])  # Only use prediction for last position
            
        Args:
            tgt_ids: Partial target sequence [batch_size, current_seq_len]
            encoder_output: Pre-computed encoder output [batch_size, src_seq_len, embed_dim]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            
        Returns:
            Decoder logits [batch_size, current_seq_len, tgt_vocab_size]
        """
        return self.decoder(tgt_ids, encoder_output, src_padding_mask, tgt_padding_mask)

    @classmethod
    def from_pretrained(cls, model_path: str) -> "Transformer":
        """Load a pretrained model from a checkpoint."""
        checkpoint = torch.load(model_path, map_location="cpu")
        config = TransformerConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_pretrained(self, model_path: str) -> None:
        """Save the model to a checkpoint."""
        config_dict = {k: v for k, v in self.config.__dict__.items()}
        checkpoint = {"config": config_dict, "model_state_dict": self.state_dict()}
        torch.save(checkpoint, model_path)
        
        # Print file size
        file_size = os.path.getsize(model_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"Model saved to {model_path} (Size: {file_size_mb:.2f} MB)")

    def get_encoder_attention_weights(self) -> List[torch.Tensor]:
        """Get attention weights from all encoder layers.

        Returns:
            List of attention weight tensors, one per layer
        """
        if not self.store_attention:
            raise ValueError("Attention weights not stored. Initialize with store_attention=True")

        attention_weights = []
        for i, layer in enumerate(self.encoder.encoder_layers):
            attention_weights.append(layer.self_attn.last_attention_weights)

        return attention_weights

    def get_decoder_self_attention_weights(self) -> List[torch.Tensor]:
        """Get self-attention weights from all decoder layers.

        Returns:
            List of attention weight tensors, one per layer
        """
        if not self.store_attention:
            raise ValueError("Attention weights not stored. Initialize with store_attention=True")

        attention_weights = []
        for i, layer in enumerate(self.decoder.decoder_layers):
            attention_weights.append(layer.self_attn.last_attention_weights)

        return attention_weights

    def get_decoder_cross_attention_weights(self) -> List[torch.Tensor]:
        """Get cross-attention weights from all decoder layers.

        Returns:
            List of attention weight tensors, one per layer
        """
        if not self.store_attention:
            raise ValueError("Attention weights not stored. Initialize with store_attention=True")

        attention_weights = []
        for i, layer in enumerate(self.decoder.decoder_layers):
            attention_weights.append(layer.cross_attn.last_attention_weights)

        return attention_weights

    def visualize_encoder_attention(
        self,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize encoder self-attention weights.

        Args:
            layer_idx: Index of the encoder layer to visualize
            head_idx: Index of the attention head to visualize (None for average)
            tokens: Optional list of token strings for axis labels
            **kwargs: Additional arguments to pass to plot_attention_weights

        Returns:
            Matplotlib figure object
        """
        attention_weights = self.get_encoder_attention_weights()
        return plot_attention_weights(
            attention_weights[layer_idx],
            tokens=tokens,
            title="Encoder Self-Attention",
            layer_idx=layer_idx,
            head_idx=head_idx,
            **kwargs,
        )

    def visualize_decoder_self_attention(
        self,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize decoder self-attention weights.

        Args:
            layer_idx: Index of the decoder layer to visualize
            head_idx: Index of the attention head to visualize (None for average)
            tokens: Optional list of token strings for axis labels
            **kwargs: Additional arguments to pass to plot_attention_weights

        Returns:
            Matplotlib figure object
        """
        attention_weights = self.get_decoder_self_attention_weights()
        return plot_attention_weights(
            attention_weights[layer_idx],
            tokens=tokens,
            title="Decoder Self-Attention",
            layer_idx=layer_idx,
            head_idx=head_idx,
            **kwargs,
        )

    def visualize_decoder_cross_attention(
        self,
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        src_tokens: Optional[List[str]] = None,
        tgt_tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize decoder cross-attention weights.

        Args:
            layer_idx: Index of the decoder layer to visualize
            head_idx: Index of the attention head to visualize (None for average)
            src_tokens: Optional list of source token strings for x-axis labels
            tgt_tokens: Optional list of target token strings for y-axis labels
            **kwargs: Additional arguments to pass to plot_attention_weights

        Returns:
            Matplotlib figure object
        """
        attention_weights = self.get_decoder_cross_attention_weights()
        # For cross-attention, we need both source and target tokens for proper visualization
        # Source tokens should be on x-axis (keys) and target tokens on y-axis (queries)
        return plot_attention_weights(
            attention_weights[layer_idx],
            tokens=src_tokens,  # Source tokens for cross-attention keys
            title="Decoder Cross-Attention",
            layer_idx=layer_idx,
            head_idx=head_idx,
            **kwargs,
        )

    def visualize_encoder_embeddings(
        self, input_ids: torch.Tensor, tokens: Optional[List[str]] = None, **kwargs
    ) -> plt.Figure:
        """Visualize encoder token embeddings using PCA.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            tokens: Optional list of token strings for labels
            **kwargs: Additional arguments to pass to plot_embeddings_pca

        Returns:
            Matplotlib figure object
        """
        with torch.no_grad():
            embeddings = self.encoder.token_embedding(input_ids[0])  # Take first batch

        return plot_embeddings_pca(
            embeddings, tokens=tokens, title="Encoder Token Embeddings", **kwargs
        )

    def visualize_attention_heads(
        self,
        attention_type: str = "encoder",
        layer_idx: int = 0,
        tokens: Optional[List[str]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Visualize multiple attention heads in a grid.

        Args:
            attention_type: Type of attention to visualize ("encoder", "decoder_self", or "decoder_cross")
            layer_idx: Index of the layer to visualize
            tokens: Optional list of token strings for axis labels
            **kwargs: Additional arguments to pass to plot_attention_heads

        Returns:
            Matplotlib figure object
        """
        if attention_type == "encoder":
            attention_weights = self.get_encoder_attention_weights()[layer_idx]
        elif attention_type == "decoder_self":
            attention_weights = self.get_decoder_self_attention_weights()[layer_idx]
        elif attention_type == "decoder_cross":
            attention_weights = self.get_decoder_cross_attention_weights()[layer_idx]
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        return plot_attention_heads(attention_weights, tokens=tokens, layer_idx=layer_idx, **kwargs)
