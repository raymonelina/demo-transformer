"""
Example demonstrating the use of data utilities for transformer training.
"""

import torch
from demo_transformer import (
    Transformer,
    TransformerConfig,
    TransformerTrainer,
    TransformerDataset,
    create_dataloaders,
    get_transformer_scheduler,
)
import torch.optim as optim


def main():
    # Configuration
    config = TransformerConfig(
        embed_dim=256,
        num_heads=8,
        ff_dim=512,
        num_encoder_layers=2,
        num_decoder_layers=2,
        max_seq_len=32,
        src_vocab_size=1000,
        tgt_vocab_size=1000,
        dropout_rate=0.1,
    )
    
    print("Creating synthetic dataset...")
    
    # Create synthetic data
    num_samples = 100
    src_data = []
    tgt_data = []
    
    for _ in range(num_samples):
        # Random source sequence (no SOS/EOS)
        src_len = torch.randint(5, 15, (1,)).item()
        src_seq = torch.randint(1, config.src_vocab_size, (src_len,)).tolist()
        
        # Random target sequence (with SOS/EOS)
        tgt_len = torch.randint(5, 15, (1,)).item()
        tgt_seq = [config.sos_token_id] + torch.randint(1, config.tgt_vocab_size, (tgt_len-2,)).tolist() + [config.eos_token_id]
        
        src_data.append(src_seq)
        tgt_data.append(tgt_seq)
    
    print(f"Created {num_samples} samples")
    print(f"Sample source: {src_data[0]}")
    print(f"Sample target: {tgt_data[0]}")
    
    # Create dataloaders using the utility function
    train_loader, val_loader = create_dataloaders(
        src_train=src_data[:80],
        tgt_train=tgt_data[:80],
        src_val=src_data[80:],
        tgt_val=tgt_data[80:],
        batch_size=8,
        pad_token_id=config.pad_token_id,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Initialize model and trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = get_transformer_scheduler(optimizer, config.embed_dim, warmup_steps=100)
    
    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        label_smoothing=0.1,
        device=device,
    )
    
    print("\nTraining for 2 epochs...")
    
    # Training loop
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}/2")
        
        # Train
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            metrics = trainer.train_step(
                batch["src_ids"],
                batch["tgt_ids"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            total_loss += metrics["loss"]
            scheduler.step()
            
            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {metrics['loss']:.4f}")
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"  Average train loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["src_ids"],
                    batch["tgt_ids"][:, :-1],  # Remove last token for input
                    batch["src_padding_mask"],
                    batch["tgt_padding_mask"][:, :, :-1, :-1] if batch["tgt_padding_mask"] is not None else None,
                )
                # Simple loss calculation for validation
                targets = batch["tgt_ids"][:, 1:].reshape(-1)  # Remove first token (SOS)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets,
                    ignore_index=config.pad_token_id,
                )
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"  Average val loss: {avg_val_loss:.4f}")
    
    print("\nTraining completed!")
    
    # Show a sample batch structure
    print("\nSample batch structure:")
    sample_batch = next(iter(train_loader))
    for key, value in sample_batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()