"""
Training script for English-Chinese translation using Hugging Face datasets.
Uses the demo_transformer data utilities for proper data preparation.
"""

import torch
import torch.optim as optim
import os
import json
import pandas as pd
from collections import Counter
from datasets import load_dataset
from demo_transformer import (
    Transformer,
    TransformerConfig,
    TransformerTrainer,
    TransformerDataset,
    create_dataloaders,
    get_transformer_scheduler,
)


def load_full_huggingface_dataset():
    """Load the complete English-Chinese translation dataset from Hugging Face."""
    print("Loading FULL English-Chinese translation dataset from Hugging Face...")
    
    try:
        # Load the dataset
        print("Loading dataset: swaption2009/20k-en-zh-translation-pinyin-hsk")
        huggingface_dataset = load_dataset("swaption2009/20k-en-zh-translation-pinyin-hsk", data_dir="")
        huggingface_dataset = huggingface_dataset["train"]
        
        # Process the FULL dataset
        num_of_rows = huggingface_dataset.num_rows // 5
        pairs = []
        
        print(f"Processing ALL {num_of_rows} translation pairs...")
        print("This may take a few minutes...")
        
        for i in range(num_of_rows):
            try:
                english_idx = (i * 5) + 0
                chinese_idx = (i * 5) + 2
                
                if english_idx < len(huggingface_dataset) and chinese_idx < len(huggingface_dataset):
                    english = huggingface_dataset[english_idx]["text"].strip("english: ").strip()
                    chinese = huggingface_dataset[chinese_idx]["text"].strip("mandarin: ").strip()
                    
                    # More lenient filtering for full dataset
                    if len(english) > 1 and len(chinese) > 0 and len(english) < 100 and len(chinese) < 100:
                        pairs.append((english, chinese))
                        
            except (IndexError, KeyError) as e:
                continue
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{num_of_rows} pairs...")
        
        print(f"Successfully loaded {len(pairs)} translation pairs from full dataset")
        return pairs
        
    except Exception as e:
        print(f"Failed to load Hugging Face dataset: {e}")
        return []


def build_vocab(sentences, max_vocab_size=5000):
    """Build vocabulary from sentences."""
    counter = Counter()
    
    for sentence in sentences:
        # Chinese: character-level tokenization
        if any('\u4e00' <= char <= '\u9fff' for char in sentence):
            tokens = list(sentence.replace(" ", ""))
        else:  # English: word-level tokenization
            tokens = sentence.lower().split()
        counter.update(tokens)
    
    # Special tokens
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
    
    # Add most common tokens
    for token, _ in counter.most_common(max_vocab_size - 4):
        vocab[token] = len(vocab)
    
    return vocab


def tokenize_sentence(sentence, vocab, is_chinese=False):
    """Tokenize sentence using vocabulary."""
    if is_chinese:
        tokens = list(sentence.replace(" ", ""))
    else:
        tokens = sentence.lower().split()
    
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


def prepare_training_data(pairs, max_src_len=50, max_tgt_len=50):
    """Prepare training data with tokenization."""
    print("\nPreparing training data...")
    
    # Filter pairs by length
    filtered_pairs = []
    for en, zh in pairs:
        if len(en.split()) <= max_src_len and len(zh) <= max_tgt_len:
            filtered_pairs.append((en, zh))
    
    print(f"Filtered to {len(filtered_pairs)} pairs within length limits")
    
    # Build vocabularies
    en_sentences = [pair[0] for pair in filtered_pairs]
    zh_sentences = [pair[1] for pair in filtered_pairs]
    
    en_vocab = build_vocab(en_sentences, max_vocab_size=8000)  # Larger vocab for full dataset
    zh_vocab = build_vocab(zh_sentences, max_vocab_size=8000)
    
    print(f"English vocab size: {len(en_vocab)}")
    print(f"Chinese vocab size: {len(zh_vocab)}")
    
    # Tokenize sentences
    src_data = []
    tgt_data = []
    
    for en, zh in filtered_pairs:
        en_tokens = tokenize_sentence(en, en_vocab, is_chinese=False)
        zh_tokens = tokenize_sentence(zh, zh_vocab, is_chinese=True)
        
        # Add SOS and EOS to target (required by transformer)
        zh_tokens = [zh_vocab["<sos>"]] + zh_tokens + [zh_vocab["<eos>"]]
        
        src_data.append(en_tokens)
        tgt_data.append(zh_tokens)
    
    # Show tokenization examples
    print("\nTokenization examples:")
    print("-" * 40)
    for i in range(min(5, len(filtered_pairs))):
        en, zh = filtered_pairs[i]
        en_tokens = src_data[i]
        zh_tokens = tgt_data[i]
        print(f"{i+1}. EN: {en}")
        print(f"   Tokens: {en_tokens}")
        print(f"   ZH: {zh}")
        print(f"   Tokens: {zh_tokens}")
        print()
    
    return src_data, tgt_data, en_vocab, zh_vocab, filtered_pairs


def load_model(save_dir):
    """Load existing model if available."""
    model_path = os.path.join(save_dir, 'model.pt')
    
    if os.path.exists(model_path):
        print(f"Found existing model: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            return checkpoint
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None


def save_model(model, optimizer, scheduler, epoch, loss, save_dir, config, en_vocab, zh_vocab):
    """Save complete model with all training state."""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'config': config,
        'en_vocab': en_vocab,
        'zh_vocab': zh_vocab
    }
    
    model_path = os.path.join(save_dir, 'model.pt')
    torch.save(checkpoint, model_path)
    
    # Calculate and display model size
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    print(f"  Model saved: {model_size:.1f} MB")


def train_model(src_data, tgt_data, en_vocab, zh_vocab, save_dir=None):
    """Train the transformer model with resumable training and detailed stats."""
    if save_dir is None:
        save_dir = os.path.expanduser("~/models/translation_model")
        print(f"Using default save directory: {save_dir}")
    """Train the transformer model with resumable training and detailed stats."""
    print("\n" + "="*60)
    print("TRAINING TRANSFORMER MODEL")
    print("="*60)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Device selection with MPS support for macOS
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"\nDevice Information:")
    print(f"  - Using device: {device}")
    if device.type == 'cuda':
        print(f"  - GPU name: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    elif device.type == 'mps':
        print(f"  - Apple Silicon GPU acceleration enabled")
    else:
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        print(f"  - MPS available: {torch.backends.mps.is_available()}")
        print(f"  - Running on CPU (training will be slower)")
    
    # Try to load existing model first
    checkpoint = load_model(save_dir)
    
    if checkpoint:
        print("\nFound existing model:")
        config = checkpoint['config']
        print(f"  - Embedding dimension: {config.embed_dim}")
        print(f"  - Encoder/Decoder layers: {config.num_encoder_layers}/{config.num_decoder_layers}")
        print(f"  - Vocab sizes: {config.src_vocab_size}/{config.tgt_vocab_size}")
        print(f"  - Last epoch: {checkpoint['epoch']}")
        print(f"  - Last training loss: {checkpoint['loss']:.4f}")
                
        # Test model on dataset
        model = Transformer(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Quick dataset compatibility check
        try:
            train_loader, _ = create_dataloaders(
                src_train=src_data[:100],
                tgt_train=tgt_data[:100],
                batch_size=2,
                pad_token_id=config.pad_token_id,
            )
            sample_batch = next(iter(train_loader))
            sample_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in sample_batch.items()}
            with torch.no_grad():
                _ = model(sample_batch["src_ids"], sample_batch["tgt_ids"][:, :-1], 
                         sample_batch["src_padding_mask"])
            print("  - Dataset compatibility: OK")
            
        except Exception as e:
            print(f"  - Dataset compatibility: ERROR - {e}")
        
        response = input("\nContinue training existing model? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            print("Continuing with existing model...")
            use_existing = True
        else:
            print("Starting fresh model...")
            use_existing = False
    else:
        print("No existing model found.")
        use_existing = False
    
    if not use_existing:
        # Create new configuration
        config = TransformerConfig(
            embed_dim=512,
            num_heads=4,
            ff_dim=2048,
            num_encoder_layers=4,
            num_decoder_layers=4,
            max_seq_len=128,
            src_vocab_size=len(en_vocab),
            tgt_vocab_size=len(zh_vocab),
            dropout_rate=0.1,
            pad_token_id=0,
            sos_token_id=1,
            eos_token_id=2,
        )
        
        print(f"\nNew Model Configuration:")
        print(f"  - Embedding dimension: {config.embed_dim}")
        print(f"  - Number of heads: {config.num_heads}")
        print(f"  - Feed-forward dimension: {config.ff_dim}")
        print(f"  - Encoder layers: {config.num_encoder_layers}")
        print(f"  - Decoder layers: {config.num_decoder_layers}")
        print(f"  - Max sequence length: {config.max_seq_len}")
        print(f"  - Source vocab size: {config.src_vocab_size}")
        print(f"  - Target vocab size: {config.tgt_vocab_size}")
        
        response = input("\nProceed with this configuration? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Training cancelled.")
            return None, None
        
        model = Transformer(config).to(device)
        checkpoint = None
    
    # Use all data for training
    train_loader, _ = create_dataloaders(
        src_train=src_data,
        tgt_train=tgt_data,
        batch_size=32,
        pad_token_id=config.pad_token_id,
    )
    
    print(f"\nDataset:")
    print(f"  - Training samples: {len(src_data)}")
    print(f"  - Training batches: {len(train_loader)}")
    
    # --- PyTorch 2.0 Compilation (for performance) ---
    # If using PyTorch 2.0+, torch.compile can significantly speed up the model.
    # Note: torch.compile support for Apple's MPS is still experimental and may fail.
    if hasattr(torch, 'compile') and device.type != 'mps':
        print("\nAttempting to compile model with torch.compile... (requires PyTorch 2.0+)")
        try:
            model = torch.compile(model, mode="max-autotune")
            print("  - Model compiled successfully.")
        except Exception as e:
            print(f"  - Model compilation failed: {e}")
    elif device.type == 'mps':
        print("\nSkipping model compilation: torch.compile support for MPS is experimental.")
    else:
        print("\nSkipping model compilation (torch.compile not available).")

    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Training parameters
    learning_rate = 0.0001
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_transformer_scheduler(optimizer, config.embed_dim, warmup_steps=4000)
    
    # Trainer
    trainer = TransformerTrainer(
        model=model,
        optimizer=optimizer,
        label_smoothing=0.1,
        device=device,
    )
    
    # Load checkpoint states if continuing
    start_epoch = 0
    if use_existing and checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"\nResuming from epoch {start_epoch + 1}")
    
    # Training parameters
    num_epochs = 2000  # More epochs for full dataset
    save_every = 1   # Save every 1 epochs
    
    print(f"\nTraining Parameters:")
    print(f"  - Total epochs: {num_epochs}")
    print(f"  - Starting from epoch: {start_epoch + 1}")
    print(f"  - Save every: {save_every} epochs")
    print(f"  - Batch size: 32")
    print(f"  - Learning rate: {learning_rate}")
    
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Training loop with detailed stats
    best_val_loss = float('inf')
    avg_train_loss = 0.0  # Initialize to avoid UnboundLocalError
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            metrics = trainer.train_step(
                batch["src_ids"],
                batch["tgt_ids"],
                batch["src_padding_mask"],
                batch["tgt_padding_mask"],
            )
            total_train_loss += metrics["loss"]
            num_train_batches += 1
            scheduler.step()
            
            # Progress indicator
            if (batch_idx + 1) % 100 == 0:
                # Get learning rate from optimizer (more reliable)
                current_lr = optimizer.param_groups[0]['lr']
                # Calculate global step for scheduler debugging
                global_step = epoch * len(train_loader) + batch_idx + 1
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {metrics['loss']:.4f}, LR: {current_lr:.24f}, Step: {global_step}")
        
        avg_train_loss = total_train_loss / num_train_batches
        
        # Display epoch stats
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.8f}")
        
        # Save best model based on training loss
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            best_model_dir = os.path.join(save_dir, "best_model")
            save_model(model, optimizer, scheduler, epoch + 1, avg_train_loss, best_model_dir, config, en_vocab, zh_vocab)
            print(f"  *** New best model saved! (Train Loss: {best_val_loss:.4f}) ***")
            print(f"  *** Location: {best_model_dir} ***")
        
        # Periodic saving
        if (epoch + 1) % save_every == 0:
            save_model(model, optimizer, scheduler, epoch + 1, avg_train_loss, save_dir, config, en_vocab, zh_vocab)
            print(f"  Model saved at epoch {epoch+1}")
            print(f"  Location: {save_dir}")
    
    # Final save
    save_model(model, optimizer, scheduler, num_epochs, avg_train_loss, save_dir, config, en_vocab, zh_vocab)
    print(f"Final model saved to: {save_dir}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    return model, config


def main():
    """Main training function."""
    print("English-Chinese Translation Model Training")
    print("=" * 50)
    
    # Load FULL dataset
    pairs = load_full_huggingface_dataset()
    
    if not pairs:
        print("No data loaded. Exiting.")
        return
    
    print(f"\n" + "="*60)
    print(f"DATASET LOADED SUCCESSFULLY")
    print(f"="*60)
    print(f"Total translation pairs: {len(pairs):,}")
    
    # Show examples
    print("\nSample translations:")
    print("-" * 40)
    for i, (en, zh) in enumerate(pairs[:10]):
        print(f"{i+1:2d}. EN: {en}")
        print(f"    ZH: {zh}")
        print()
    
    # Ask for confirmation
    print(f"\nDataset Statistics:")
    print(f"  - Total pairs: {len(pairs):,}")
    print(f"  - Average English length: {sum(len(en.split()) for en, zh in pairs) / len(pairs):.1f} words")
    print(f"  - Average Chinese length: {sum(len(zh) for en, zh in pairs) / len(pairs):.1f} characters")
    
    print("\n" + "="*60)
    response = input("Do you want to proceed with training on this full dataset? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        return
    

    # Prepare data using TransformerDataset
    src_data, tgt_data, en_vocab, zh_vocab, filtered_pairs = prepare_training_data(pairs)
    
    print(f"\nFinal dataset after filtering:")
    print(f"  - Training pairs: {len(src_data):,}")
    print(f"  - English vocabulary: {len(en_vocab):,} tokens")
    print(f"  - Chinese vocabulary: {len(zh_vocab):,} tokens")

    
    # Train model - save to user's home directory
    save_dir = os.path.expanduser("~/models/translation_model")
    model, config = train_model(src_data, tgt_data, en_vocab, zh_vocab, save_dir)
    
    print("\nTraining completed successfully!")
    print("Run 'uv run python examples/inference_translation_model.py' to test the trained model.")


if __name__ == "__main__":
    main()