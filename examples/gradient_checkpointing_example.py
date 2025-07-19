"""
Example demonstrating the use of gradient checkpointing in the transformer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import gc
from demo_transformer import Transformer, TransformerConfig, TransformerTrainer


def measure_memory_usage(model, src_ids, tgt_ids, src_padding_mask):
    """Measure peak memory usage during a forward and backward pass."""
    # Clear cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Record initial memory
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Forward pass
    model.train()
    logits = model(src_ids, tgt_ids, src_padding_mask)
    
    # Compute loss
    loss = criterion(logits.view(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))
    
    # Backward pass
    loss.backward()
    
    # Get peak memory usage
    peak_memory = torch.cuda.max_memory_allocated()
    
    # Clean up
    optimizer.zero_grad()
    del logits, loss
    torch.cuda.empty_cache()
    gc.collect()
    
    return peak_memory - initial_memory


def measure_training_time(model, src_ids, tgt_ids, src_padding_mask, num_iterations=10):
    """Measure average training time per iteration."""
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Warmup
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_ids, src_padding_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))
        loss.backward()
    
    # Measure time
    start_time = time.time()
    model.train()
    for _ in range(num_iterations):
        optimizer.zero_grad()
        logits = model(src_ids, tgt_ids, src_padding_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_ids[:, 1:].reshape(-1))
        loss.backward()
    end_time = time.time()
    
    return (end_time - start_time) / num_iterations


def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU to demonstrate memory savings.")
        return
    
    device = torch.device("cuda")
    print(f"Using device: {device}")
    
    # Create a larger model to better demonstrate the memory savings
    base_config = TransformerConfig(
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_seq_len=512,
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        dropout_rate=0.1,
        use_gradient_checkpointing=False,  # No checkpointing
    )
    
    checkpointing_config = TransformerConfig(
        embed_dim=512,
        num_heads=8,
        ff_dim=2048,
        num_encoder_layers=6,
        num_decoder_layers=6,
        max_seq_len=512,
        src_vocab_size=32000,
        tgt_vocab_size=32000,
        dropout_rate=0.1,
        use_gradient_checkpointing=True,  # With checkpointing
    )
    
    print("Initializing models...")
    base_model = Transformer(base_config).to(device)
    checkpointing_model = Transformer(checkpointing_config).to(device)
    
    # Create dummy inputs (larger size to demonstrate memory savings)
    batch_size = 8
    seq_len = 256
    
    src_ids = torch.randint(0, base_config.src_vocab_size, (batch_size, seq_len)).to(device)
    tgt_ids = torch.randint(0, base_config.tgt_vocab_size, (batch_size, seq_len)).to(device)
    src_padding_mask = torch.zeros(batch_size, 1, 1, seq_len, dtype=torch.bool).to(device)
    
    print(f"Input shapes - Source: {src_ids.shape}, Target: {tgt_ids.shape}")
    
    # Measure memory usage
    print("\nMeasuring memory usage...")
    base_memory = measure_memory_usage(base_model, src_ids, tgt_ids, src_padding_mask)
    checkpointing_memory = measure_memory_usage(checkpointing_model, src_ids, tgt_ids, src_padding_mask)
    
    print(f"Memory usage without checkpointing: {base_memory / (1024**2):.2f} MB")
    print(f"Memory usage with checkpointing: {checkpointing_memory / (1024**2):.2f} MB")
    print(f"Memory savings: {(1 - checkpointing_memory / base_memory) * 100:.2f}%")
    
    # Measure training time
    print("\nMeasuring training time...")
    base_time = measure_training_time(base_model, src_ids, tgt_ids, src_padding_mask)
    checkpointing_time = measure_training_time(checkpointing_model, src_ids, tgt_ids, src_padding_mask)
    
    print(f"Average iteration time without checkpointing: {base_time * 1000:.2f} ms")
    print(f"Average iteration time with checkpointing: {checkpointing_time * 1000:.2f} ms")
    print(f"Time overhead: {(checkpointing_time / base_time - 1) * 100:.2f}%")
    
    print("\nExample completed.")


if __name__ == "__main__":
    main()