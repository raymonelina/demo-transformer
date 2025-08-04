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
    
    This dataset handles paired source-target sequences for training and evaluation.
    All sequences should be pre-tokenized and converted to integer token IDs.
    
    Data Format Requirements:
    - src_ids: List of source sequences, each sequence is a list of token IDs
    - tgt_ids: List of target sequences, each sequence is a list of token IDs
    - Both lists must have the same length (paired sequences)
    - Token IDs should be integers in range [0, vocab_size-1]
    
    Special Tokens (typical setup):
    - pad_token_id: 0 (padding)
    - sos_token_id: 1 (start-of-sequence)
    - eos_token_id: 2 (end-of-sequence)
    - unk_token_id: 3 (unknown token)
    
    Training Data Example (English-French Translation):
    ```python
    # Raw text pairs
    en_sentences = ["Hello world", "How are you?"]
    fr_sentences = ["Bonjour monde", "Comment allez-vous?"]
    
    # After tokenization and numericalization
    src_ids = [
        [245, 678],      # ["Hello", "world"] - NO SOS/EOS for encoder!
        [123, 456, 789]  # ["How", "are", "you?"] - NO SOS/EOS for encoder!
    ]
    # Complete target sequences (will be split into input/target during training)
    tgt_ids = [
        [1, 891, 234, 2],         # [SOS, "Bonjour", "monde", EOS]
        [1, 567, 890, 123, 2]     # [SOS, "Comment", "allez-vous?", EOS]
    ]
    # During training, this gets split into:
    # Decoder input = tgt_ids[:, :-1]  → [1, 891, 234] = [SOS, "Bonjour", "monde"]
    # Decoder target = tgt_ids[:, 1:]  → [891, 234, 2] = ["Bonjour", "monde", EOS]
    
    dataset = TransformerDataset(src_ids, tgt_ids, pad_token_id=0)
    ```
    
    Inference Data Example:
    ```python
    # For inference, you only need source sequences
    src_ids = [
        [245, 678],      # ["Hello", "world"] - NO SOS/EOS for encoder!
        [999, 888, 777]  # ["Good", "morning", "sir"] - NO SOS/EOS for encoder!
    ]
    # Targets not needed for inference, but dataset requires them
    # Use dummy targets or just SOS tokens
    tgt_ids = [
        [1],  # Just SOS token
        [1]   # Just SOS token
    ]
    
    inference_dataset = TransformerDataset(src_ids, tgt_ids, pad_token_id=0)
    ```
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
    Collator for transformer batches with dynamic padding and mask creation.
    
    This collator handles batching of variable-length sequences by:
    1. Finding the maximum length in each batch
    2. Padding all sequences to that length
    3. Creating attention masks to ignore padding tokens
    
    Three Types of Masks in Transformer:
    1. Source Padding Mask: Prevents attention to source padding tokens
    2. Target Padding Mask: Prevents attention to target padding tokens  
    3. Causal Mask: Prevents attention to future tokens (generated in decoder)
    
    Batch Output Format:
    - src_ids: [batch_size, max_src_len] - Source token IDs
    - tgt_ids: [batch_size, max_tgt_len] - Target token IDs
    - src_padding_mask: [batch_size, 1, 1, max_src_len] - Source padding mask
    - tgt_padding_mask: [batch_size, 1, max_tgt_len, max_tgt_len] - Target padding mask
    
    Padding Mask Convention:
    - False (0): Real tokens (attend to these)
    - True (1): Padding tokens (ignore these)
    
    Example Batch Processing:
    ```python
    # Input batch (variable lengths)
    batch = [
        {"src_ids": [245, 678], "tgt_ids": [1, 891, 234, 2]},
        {"src_ids": [123, 456, 789, 999], "tgt_ids": [1, 567, 890, 2]}
    ]
    
    # After collation (padded to max lengths)
    output = {
        "src_ids": tensor([
            [245, 678, 0, 0],      # Padded with 2 zeros
            [123, 456, 789, 999]   # No padding needed
        ]),
        "tgt_ids": tensor([
            [1, 891, 234, 2],  # No padding needed
            [1, 567, 890, 0]   # Padded with 1 zero
        ]),
        "src_padding_mask": tensor([...]),  # Source padding positions
        "tgt_padding_mask": tensor([...])   # Target padding positions
    }
    ```
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
        Collate a batch of samples with dynamic padding and mask creation.
        
        Creates both source and target padding masks to handle variable-length sequences.
        The target padding mask will be combined with the causal mask in the decoder.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Batch dictionary with padded tensors and attention masks
        """
        # Find maximum sequence lengths in this batch
        src_max_len = max(len(sample["src_ids"]) for sample in batch)
        tgt_max_len = max(len(sample["tgt_ids"]) for sample in batch)
        
        # Initialize padded tensors filled with padding token
        batch_size = len(batch)
        src_ids = torch.full((batch_size, src_max_len), self.pad_token_id, dtype=torch.long)
        tgt_ids = torch.full((batch_size, tgt_max_len), self.pad_token_id, dtype=torch.long)
        
        # Create padding masks (False=real tokens, True=padding tokens)
        # Source mask: [batch_size, 1, 1, src_max_len] for cross-attention
        src_padding_mask = torch.zeros((batch_size, 1, 1, src_max_len), dtype=torch.bool)
        # Target mask: [batch_size, 1, tgt_max_len, tgt_max_len] for self-attention
        tgt_padding_mask = torch.zeros((batch_size, 1, tgt_max_len, tgt_max_len), dtype=torch.bool)
        
        # Process each sample in the batch
        for i, sample in enumerate(batch):
            src = sample["src_ids"]
            tgt = sample["tgt_ids"]
            
            src_len = len(src)
            tgt_len = len(tgt)
            
            # Copy actual token IDs into padded tensors
            src_ids[i, :src_len] = torch.tensor(src, dtype=torch.long)
            tgt_ids[i, :tgt_len] = torch.tensor(tgt, dtype=torch.long)
            
            # Mark padding positions as True (will be masked out in attention)
            # Source padding mask: mask positions beyond actual sequence length
            src_padding_mask[i, :, :, src_len:] = True
            
            # Target padding mask: mask padding positions in both query and key dimensions
            # This creates a mask where padded positions cannot attend or be attended to
            if tgt_len < tgt_max_len:
                # Mask rows (queries) corresponding to padding positions
                tgt_padding_mask[i, :, tgt_len:, :] = True
                # Mask columns (keys) corresponding to padding positions
                tgt_padding_mask[i, :, :, tgt_len:] = True
        
        # Move tensors to specified device if provided
        if self.device is not None:
            src_ids = src_ids.to(self.device)
            tgt_ids = tgt_ids.to(self.device)
            src_padding_mask = src_padding_mask.to(self.device)
            tgt_padding_mask = tgt_padding_mask.to(self.device)
        
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_padding_mask": src_padding_mask,
            "tgt_padding_mask": tgt_padding_mask,
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
    Create training and validation dataloaders for transformer training.
    
    This function creates PyTorch DataLoaders with proper batching, padding,
    and shuffling for transformer sequence-to-sequence training.
    
    Complete Data Pipeline Example:
    ```python
    # 1. Prepare tokenized data
    src_train = [
        [245, 678],      # "Hello world" - NO SOS/EOS for encoder!
        [123, 456, 789], # "How are you?" - NO SOS/EOS for encoder!
        [999, 888]       # "Good morning" - NO SOS/EOS for encoder!
    ]
    tgt_train = [
        [1, 891, 234, 2],      # "Bonjour monde"
        [1, 567, 890, 123, 2], # "Comment allez-vous?"
        [1, 777, 666, 2]       # "Bonjour"
    ]
    
    # 2. Create dataloaders
    train_loader, val_loader = create_dataloaders(
        src_train=src_train,
        tgt_train=tgt_train,
        batch_size=2,
        pad_token_id=0
    )
    
    # 3. Use in training loop
    for batch in train_loader:
        src_ids = batch["src_ids"]              # [batch_size, max_src_len]
        tgt_ids = batch["tgt_ids"]              # [batch_size, max_tgt_len]
        src_mask = batch["src_padding_mask"]     # [batch_size, 1, 1, max_src_len]
        tgt_mask = batch["tgt_padding_mask"]     # [batch_size, 1, max_tgt_len, max_tgt_len]
        
        # Split tgt_ids into decoder input and target
        decoder_input = tgt_ids[:, :-1]    # [SOS, "J'aime", "les", "chats"]
        decoder_target = tgt_ids[:, 1:]    # ["J'aime", "les", "chats", EOS]
        
        # Forward pass with decoder input (not full tgt_ids!)
        # Note: tgt_mask needs to match decoder_input size, not decoder_target
        decoder_mask = tgt_mask[:, :, :-1, :-1]  # Remove last row/col to match decoder_input
        logits = model(src_ids, decoder_input, src_mask, decoder_mask)
        loss = criterion(logits.view(-1, vocab_size), decoder_target.reshape(-1))
    ```
    
    Data Preparation Guidelines:
    1. Tokenize text: "Hello world" → ["Hello", "world"]
    2. Convert to IDs: ["Hello", "world"] → [245, 678]
    3. Add special tokens:
       - Source (encoder): [245, 678] → [245, 678] (NO SOS/EOS!)
       - Target (decoder): [891, 234] → [1, 891, 234, 2] (SOS + tokens + EOS)
    4. For demo purposes, random token IDs can be used: torch.randint(1, vocab_size, (seq_len,))
    5. Ensure consistent vocabulary between source and target (or separate vocabs)
    
    Args:
        src_train: Source training sequences (list of token ID lists)
        tgt_train: Target training sequences (list of token ID lists)
        src_val: Source validation sequences (optional)
        tgt_val: Target validation sequences (optional)
        batch_size: Number of sequences per batch
        pad_token_id: Token ID used for padding shorter sequences
        device: Device to place tensors on (CPU/GPU)
        num_workers: Number of subprocesses for data loading
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
        val_dataloader is None if validation data not provided
    """
    # Create datasets with paired source-target sequences
    train_dataset = TransformerDataset(src_train, tgt_train, pad_token_id)
    
    # Create collator for dynamic padding and mask generation
    collator = TransformerCollator(pad_token_id, device)
    
    # Create training dataloader with shuffling
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle for better training
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
            shuffle=False,  # No shuffling for validation
            collate_fn=collator,
            num_workers=num_workers,
        )
    
    return train_dataloader, val_dataloader