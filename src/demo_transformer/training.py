"""
Training utilities for the transformer model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from typing import Dict, List, Optional, Union, Callable, Tuple, Any

from .transformer import Transformer


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for sequence generation tasks.
    
    Label smoothing is a regularization technique that prevents the model from becoming
    overconfident by assigning some probability mass to incorrect labels. Instead of
    using hard targets (one-hot vectors), it uses soft targets where the correct class
    gets probability (1-ε) and the remaining ε is distributed uniformly among other classes.
    
    Academic References:
    - "Rethinking the Inception Architecture for Computer Vision" (Szegedy et al., 2016)
      https://arxiv.org/abs/1512.00567
    - "Attention Is All You Need" (Vaswani et al., 2017) - used label smoothing ε=0.1
      https://arxiv.org/abs/1706.03762
    
    Benefits:
    1. Prevents overconfident predictions and overfitting
    2. Improves model calibration (predicted probabilities better reflect actual accuracy)
    3. Acts as regularization, often improving generalization
    4. Reduces the gap between training and validation performance
    
    Example:
    Without smoothing: [0, 0, 1, 0, 0] (one-hot)
    With smoothing ε=0.1: [0.025, 0.025, 0.9, 0.025, 0.025] (soft targets)
    """
    
    def __init__(self, smoothing: float = 0.1, vocab_size: int = 0, ignore_index: int = -100):
        """
        Initialize label smoothing loss.
        
        Args:
            smoothing: Smoothing factor (0.0 means no smoothing)
            vocab_size: Size of the vocabulary
            ignore_index: Index to ignore in the loss calculation (e.g., padding)
        """
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the label smoothing loss.
        
        The loss is computed as KL divergence between the smoothed target distribution
        and the predicted distribution: KL(q_smooth || p_pred) = -∑ q_smooth * log(p_pred)
        
        Args:
            pred: Prediction logits [batch_size * seq_len, vocab_size] (flattened)
            target: Target indices [batch_size * seq_len] (flattened)
            
        Returns:
            Smoothed loss value
        """
        # Convert logits to log probabilities
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # Create smoothed target distribution
            # All incorrect classes get uniform probability: ε/(V-1)
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            
            # Correct class gets probability: (1-ε)
            # scatter_ fills the correct positions with confidence value
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            
            # Ignore padding tokens by setting their distribution to zero
            mask = (target == self.ignore_index).unsqueeze(-1)
            true_dist.masked_fill_(mask, 0.0)
            
        # Compute KL divergence: -∑ q_smooth * log(p_pred)
        return torch.sum(-true_dist * pred, dim=-1).mean()


class TransformerTrainer:
    """
    Trainer class for the Transformer model.
    """
    
    def __init__(
        self,
        model: Transformer,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr: float = 1e-4,
        label_smoothing: float = 0.1,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Transformer model
            optimizer: Optimizer (if None, will create Adam optimizer)
            lr: Learning rate (used if optimizer is None)
            label_smoothing: Label smoothing factor
            pad_token_id: Padding token ID to ignore in loss calculation
            device: Device to use for training
        """
        self.model = model
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=lr)
        
        # Set up loss function
        # Label smoothing is particularly effective for sequence generation tasks
        # as it prevents the model from being overconfident about token predictions
        if label_smoothing > 0.0:
            self.criterion = LabelSmoothingLoss(
                smoothing=label_smoothing,
                vocab_size=model.config.tgt_vocab_size,
                ignore_index=pad_token_id
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            
        # Note: For proper Transformer training, consider using the original paper's
        # learning rate schedule with get_transformer_scheduler()
            
        self.pad_token_id = pad_token_id
        
        # Set device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else 
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = device
            
        self.model.to(self.device)
        
    def train_step(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            src_ids: Source token IDs [batch_size, src_seq_len]
                Raw source tokens without SOS/EOS (encoder input)
            tgt_ids: Complete target sequence [batch_size, tgt_seq_len] 
                Full target sequence with SOS at start and EOS at end
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
                Boolean mask where True indicates padding positions to ignore
            tgt_padding_mask: Target padding mask [batch_size, 1, tgt_seq_len, tgt_seq_len]
                2D boolean mask for target self-attention (masks padding in both dimensions)
        
        Example:
            # English-French translation batch
            src_ids = [
                [4, 15, 23],      # "I love cats" (no SOS/EOS)
                [8, 12, 0]        # "Hello world" + padding
            ]
            tgt_ids = [
                [1, 7, 19, 25, 2],  # [SOS, "J'aime", "les", "chats", EOS]
                [1, 11, 16, 2, 0]   # [SOS, "Bonjour", "monde", EOS] + padding
            ]
            src_padding_mask = [
                [[[False, False, False]]],  # No padding in first sequence
                [[[False, False, True]]]    # Third position is padding
            ]
            tgt_padding_mask = [
                # 5x5 matrix - no padding positions
                [[[False, False, False, False, False],
                  [False, False, False, False, False],
                  [False, False, False, False, False],
                  [False, False, False, False, False],
                  [False, False, False, False, False]]],
                # 5x5 matrix - last position is padding
                [[[False, False, False, False, True],
                  [False, False, False, False, True],
                  [False, False, False, False, True],
                  [False, False, False, False, True],
                  [True,  True,  True,  True,  True]]]
            ]
            
        Returns:
            Dictionary with loss and other metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move inputs to device
        src_ids = src_ids.to(self.device)
        tgt_ids = tgt_ids.to(self.device)
        if src_padding_mask is not None:
            src_padding_mask = src_padding_mask.to(self.device)
        if tgt_padding_mask is not None:
            tgt_padding_mask = tgt_padding_mask.to(self.device)
        
        # Teacher forcing: split complete target sequence into input/target
        input_tgt_ids = tgt_ids[:, :-1]  # Decoder input: [SOS, "J'aime", "les", "chats"]
        target_tgt_ids = tgt_ids[:, 1:]  # Decoder target: ["J'aime", "les", "chats", EOS]
        
        # Adjust target padding mask for input sequence (remove last position)
        input_tgt_padding_mask = None
        if tgt_padding_mask is not None:
            input_tgt_padding_mask = tgt_padding_mask[:, :, :-1, :-1]
        
        # Forward pass with both source and target masks
        logits = self.model(
            src_ids, 
            input_tgt_ids, 
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=input_tgt_padding_mask
        )
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_tgt_ids.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Robust Gradient Handling for Training Stability
        # =====================================================
        # This section implements a two-tier approach to handle gradient instability,
        # which is particularly important for deep transformer models.
        
        # 1. Aggressive Gradient Clipping (max_norm=0.1)
        # -----------------------------------------------
        # Standard practice uses max_norm=1.0, but transformers with random initialization
        # can experience severe gradient explosion. The smaller clipping threshold prevents
        # this while still allowing meaningful parameter updates.
        # 
        # Academic Context:
        # - "On the difficulty of training recurrent neural networks" (Pascanu et al., 2013)
        # - Gradient clipping is essential for RNNs and attention-based models
        # - Transformers, despite not being recurrent, can suffer similar gradient issues
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        
        # 2. Gradient Validity Check and Conditional Update
        # -------------------------------------------------
        # Even after clipping, gradients can become NaN/inf due to:
        # - Numerical instability in attention computations (softmax overflow)
        # - Division by zero in layer normalization
        # - Accumulation of small numerical errors across deep layers
        # 
        # This check prevents model corruption by skipping invalid updates.
        # This is NOT just a demo hack - it's a robust training practice used in:
        # - Large language model training (GPT, BERT, etc.)
        # - Mixed precision training where overflow is common
        # - Distributed training where gradient synchronization can fail
        # 
        # General Applicability:
        # ✅ Recommended for production transformer training
        # ✅ Essential when using mixed precision (FP16/BF16)
        # ✅ Critical for large models prone to instability
        # ✅ Useful during hyperparameter search when stability varies
        if torch.isfinite(grad_norm):
            self.optimizer.step()
        else:
            # Log the occurrence for monitoring training health
            print("Warning: Skipping optimizer step due to invalid gradients")
            # Clear invalid gradients to prevent accumulation in subsequent steps
            # This is crucial because some optimizers (like Adam) maintain momentum
            # that could be corrupted by NaN/inf values
            self.optimizer.zero_grad()
        
        # Return training metrics including whether optimizer step was taken
        # Note: loss.item() is always finite here because we computed it before backward()
        # The gradient issues occur during backpropagation, not forward pass
        return {
            "loss": loss.item(),
            "optimizer_step_taken": torch.isfinite(grad_norm).item()
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 10,
        callback: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader for training data
            log_interval: How often to log progress
            callback: Optional callback function called after each batch
            
        Returns:
            Dictionary with average loss and other metrics
        """
        total_loss = 0.0
        num_batches = len(dataloader)
        
        for i, batch in enumerate(dataloader):
            src_ids = batch["src_ids"]
            tgt_ids = batch["tgt_ids"]
            src_padding_mask = batch.get("src_padding_mask")
            tgt_padding_mask = batch.get("tgt_padding_mask")
            
            metrics = self.train_step(src_ids, tgt_ids, src_padding_mask, tgt_padding_mask)
            total_loss += metrics["loss"]
            
            if (i + 1) % log_interval == 0:
                print(f"Batch {i+1}/{num_batches}, Loss: {metrics['loss']:.4f}")
                
            if callback is not None:
                callback(metrics)
                
        return {"loss": total_loss / num_batches}
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            
        Returns:
            Dictionary with average loss and other metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        with torch.no_grad():
            for batch in dataloader:
                src_ids = batch["src_ids"].to(self.device)
                tgt_ids = batch["tgt_ids"].to(self.device)
                src_padding_mask = batch.get("src_padding_mask")
                tgt_padding_mask = batch.get("tgt_padding_mask")
                
                if src_padding_mask is not None:
                    src_padding_mask = src_padding_mask.to(self.device)
                if tgt_padding_mask is not None:
                    tgt_padding_mask = tgt_padding_mask.to(self.device)
                
                # Teacher forcing: use target tokens as input, but predict next token
                input_tgt_ids = tgt_ids[:, :-1]  # Remove last token
                target_tgt_ids = tgt_ids[:, 1:]  # Remove first token (usually SOS)
                
                # Adjust target padding mask for input sequence (remove last position)
                input_tgt_padding_mask = None
                if tgt_padding_mask is not None:
                    input_tgt_padding_mask = tgt_padding_mask[:, :, :-1, :-1]
                
                # Forward pass with both source and target masks
                logits = self.model(
                    src_ids, 
                    input_tgt_ids, 
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=input_tgt_padding_mask
                )
                
                # Calculate loss
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_tgt_ids.reshape(-1))
                total_loss += loss.item()
                
        return {"loss": total_loss / num_batches}


def get_transformer_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup_steps: int = 4000,
    factor: float = 1.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Get the learning rate scheduler for Transformer as described in "Attention Is All You Need".
    
    Academic Reference:
    - "Attention Is All You Need" (Vaswani et al., 2017), Section 5.3
      https://arxiv.org/abs/1706.03762
    
    The original paper uses this learning rate schedule:
    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    
    This corresponds to:
    1. Linear warmup phase (steps 1 to warmup_steps):
       - LR increases linearly from 0 to peak_lr
       - peak_lr = d_model^(-0.5) * warmup_steps^(-0.5)
       - Formula: d_model^(-0.5) * step_num * warmup_steps^(-1.5)
    
    2. Decay phase (steps > warmup_steps):
       - LR decreases proportionally to inverse square root of step number
       - Formula: d_model^(-0.5) * step_num^(-0.5)
    
    Why this works:
    - Warmup prevents instability in early training when gradients are large
    - The 1/√step decay is gentler than exponential decay, allowing continued learning
    - d_model^(-0.5) scaling ensures larger models use smaller learning rates
    
    Original paper settings:
    - d_model = 512, warmup_steps = 4000
    - Peak LR ≈ 0.0007 (reached at step 4000)
    - Base optimizer: Adam with β1=0.9, β2=0.98, ε=10^(-9)
    
    PyTorch Implementation:
    Uses LambdaLR scheduler which multiplies the base LR by the lambda function result.
    Set base_lr=1.0 in optimizer, as the lambda function computes the actual LR.
    
    Args:
        optimizer: Optimizer to schedule (should have base_lr=1.0)
        d_model: Model dimension (embed_dim)
        warmup_steps: Number of warmup steps (original paper used 4000)
        factor: Additional scaling factor (default 1.0)
        
    Returns:
        PyTorch LambdaLR scheduler
        
    Example:
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = get_transformer_scheduler(optimizer, d_model=512, warmup_steps=4000)
        
        # In training loop:
        optimizer.step()
        scheduler.step()
    """
    def lr_lambda(step):
        """
        Compute learning rate multiplier for given step.
        
        Args:
            step: Current training step (1-indexed)
            
        Returns:
            Learning rate multiplier
        """
        # Avoid division by zero at step 0
        if step == 0:
            return 0
        
        # Original Transformer LR schedule formula
        # lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        warmup_factor = step * (warmup_steps ** -1.5)  # Linear warmup
        decay_factor = step ** -0.5  # Inverse square root decay
        
        return factor * (d_model ** -0.5) * min(decay_factor, warmup_factor)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)