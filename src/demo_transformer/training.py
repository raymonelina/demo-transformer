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
        
        Args:
            pred: Prediction logits [batch_size, seq_len, vocab_size]
            target: Target indices [batch_size, seq_len]
            
        Returns:
            Smoothed loss value
        """
        pred = pred.log_softmax(dim=-1)
        with torch.no_grad():
            # Create a tensor with smoothing prob for all tokens
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            
            # Fill in the confidence for the true tokens
            true_dist.scatter_(2, target.unsqueeze(2), self.confidence)
            
            # Create mask for padding tokens to ignore
            mask = (target == self.ignore_index).unsqueeze(-1)
            true_dist.masked_fill_(mask, 0.0)
            
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
        if label_smoothing > 0.0:
            self.criterion = LabelSmoothingLoss(
                smoothing=label_smoothing,
                vocab_size=model.config.tgt_vocab_size,
                ignore_index=pad_token_id
            )
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
            
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
            tgt_ids: Target token IDs [batch_size, tgt_seq_len]
            src_padding_mask: Source padding mask [batch_size, 1, 1, src_seq_len]
            tgt_padding_mask: Target padding mask [batch_size, 1, tgt_seq_len, tgt_seq_len]
            
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
        
        # Teacher forcing: use target tokens as input, but predict next token
        input_tgt_ids = tgt_ids[:, :-1]  # Remove last token
        target_tgt_ids = tgt_ids[:, 1:]  # Remove first token (usually SOS)
        
        # Forward pass
        logits = self.model(
            src_ids, 
            input_tgt_ids, 
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask
        )
        
        # Calculate loss
        loss = self.criterion(logits.view(-1, logits.size(-1)), target_tgt_ids.reshape(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
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
                
                # Forward pass
                logits = self.model(
                    src_ids, 
                    input_tgt_ids, 
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask
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
    Get a learning rate scheduler for the Transformer as described in the original paper.
    
    Args:
        optimizer: Optimizer to schedule
        d_model: Model dimension
        warmup_steps: Number of warmup steps
        factor: Scaling factor
        
    Returns:
        Learning rate scheduler
    """
    def lr_lambda(step):
        # Linear warmup followed by rsqrt decay
        if step == 0:
            return 0
        return factor * (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)