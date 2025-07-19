"""
Data utilities for the transformer model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any


class TransformerDataset(Dataset):
    """
    Dataset for transformer sequence-to-sequence tasks.
    """
    
    def __init__(
        self,
        src_ids: List[List[int]],
        tgt_ids: List[List[int]],
        pad_token_id: int = 0,
    ):
        """
        Initialize the dataset.
        
        Args:
            src_ids: List of source token ID sequences
            tgt_ids: List of target token ID sequences
            pad_token_id: Padding token ID
        """
        assert len(src_ids) == len(tgt_ids), "Source and target must have the same length"
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids
        self.pad_token_id = pad_token_id
        
    def __len__(self) -> int:
        return len(self.src_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {
            "src_ids": self.src_ids[idx],
            "tgt_ids": self.tgt_ids[idx],
        }


class TransformerCollator:
    """
    Collator for transformer batches.
    """
    
    def __init__(
        self,
        pad_token_id: int = 0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the collator.
        
        Args:
            pad_token_id: Padding token ID
            device: Device to put tensors on
        """
        self.pad_token_id = pad_token_id
        self.device = device
        
    def __call__(self, batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Batch dictionary with padded tensors
        """
        # Get max lengths
        src_max_len = max(len(sample["src_ids"]) for sample in batch)
        tgt_max_len = max(len(sample["tgt_ids"]) for sample in batch)
        
        # Prepare tensors
        batch_size = len(batch)
        src_ids = torch.full((batch_size, src_max_len), self.pad_token_id, dtype=torch.long)
        tgt_ids = torch.full((batch_size, tgt_max_len), self.pad_token_id, dtype=torch.long)
        
        # Create padding masks
        src_padding_mask = torch.zeros((batch_size, 1, 1, src_max_len), dtype=torch.bool)
        
        # Fill tensors and masks
        for i, sample in enumerate(batch):
            src = sample["src_ids"]
            tgt = sample["tgt_ids"]
            
            src_len = len(src)
            tgt_len = len(tgt)
            
            # Fill source and target tensors
            src_ids[i, :src_len] = torch.tensor(src, dtype=torch.long)
            tgt_ids[i, :tgt_len] = torch.tensor(tgt, dtype=torch.long)
            
            # Fill padding masks (True for padding positions)
            src_padding_mask[i, :, :, src_len:] = True
        
        # Move to device if specified
        if self.device is not None:
            src_ids = src_ids.to(self.device)
            tgt_ids = tgt_ids.to(self.device)
            src_padding_mask = src_padding_mask.to(self.device)
        
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_padding_mask": src_padding_mask,
        }


def create_dataloaders(
    src_train: List[List[int]],
    tgt_train: List[List[int]],
    src_val: Optional[List[List[int]]] = None,
    tgt_val: Optional[List[List[int]]] = None,
    batch_size: int = 32,
    pad_token_id: int = 0,
    device: Optional[torch.device] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation dataloaders.
    
    Args:
        src_train: Source training sequences
        tgt_train: Target training sequences
        src_val: Source validation sequences
        tgt_val: Target validation sequences
        batch_size: Batch size
        pad_token_id: Padding token ID
        device: Device to put tensors on
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = TransformerDataset(src_train, tgt_train, pad_token_id)
    
    # Create collator
    collator = TransformerCollator(pad_token_id, device)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )
    
    # Create validation dataloader if validation data is provided
    val_dataloader = None
    if src_val is not None and tgt_val is not None:
        val_dataset = TransformerDataset(src_val, tgt_val, pad_token_id)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        )
    
    return train_dataloader, val_dataloader