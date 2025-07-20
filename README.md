# Transformer Package

A comprehensive implementation of the Transformer Encoder-Decoder architecture for sequence-to-sequence tasks.

## Features

- Complete Transformer implementation with encoder and decoder
- Configuration system for easy model customization
- Multiple decoding strategies:
  - Greedy decoding
  - Beam search decoding
  - Sampling with temperature, top-k, and top-p (nucleus) sampling
- Training utilities with label smoothing and learning rate scheduling
- Data handling utilities for sequence-to-sequence tasks
- Model saving and loading functionality
- Pre-layer normalization option for more stable training
- Weight tying option for parameter efficiency
- Relative positional encoding for better handling of variable-length sequences
- Gradient checkpointing for memory-efficient training of large models
- Debug mode for printing tensor shapes and values during execution
- Visualization tools for attention weights and embeddings

## 🛠️ Installation

Set up the virtual environment and install the package using [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -e .
```

Or install all dependencies in one step:

```bash
uv sync             # installs default dependencies
uv sync --extra dev # installs [dev] group as well
```

For development setup:

```bash
uv pip install -e ".[dev]"
```

---

## ✅ Running Tests

This project uses [`pytest`](https://docs.pytest.org/) for testing. To run tests:

```bash
# Recommended: via uv
uv run pytest
```

Other options:

```bash
# Run tests as a Python module
python -m pytest

# Run with verbose output
python -m pytest -v

# Show test print/log output
python -m pytest -s

# Run a specific test file
python -m pytest tests/test_attention.py

# Generate coverage report
python -m pytest --cov=demo_transformer
```

## Usage

### Basic Usage

Here's a simple example of how to use the transformer package:

```python
from demo_transformer import Transformer, TransformerConfig

# Create a configuration
config = TransformerConfig(
    embed_dim=512,
    num_heads=8,
    ff_dim=2048,
    num_encoder_layers=6,
    num_decoder_layers=6,
    max_seq_len=512,
    src_vocab_size=32000,
    tgt_vocab_size=32000,
    use_relative_pos=False,  # Whether to use relative positional encoding
    use_gradient_checkpointing=False,  # Whether to use gradient checkpointing
    debug_mode=False,  # Whether to print debug information about tensors
    store_attention=False,  # Whether to store attention weights for visualization
)

# Initialize the model
model = Transformer(config)
```

### Training

```python
from demo_transformer import TransformerTrainer, get_transformer_scheduler
import torch.optim as optim

# Create optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = get_transformer_scheduler(optimizer, config.embed_dim, warmup_steps=4000)

# Create trainer
trainer = TransformerTrainer(
    model=model,
    optimizer=optimizer,
    label_smoothing=0.1,
)

# Train step
metrics = trainer.train_step(src_ids, tgt_ids, src_padding_mask)
print(f"Loss: {metrics['loss']}")
```

### Inference

```python
from demo_transformer import TransformerInference

# Create inference module
inference = TransformerInference(model)

# Greedy decoding
greedy_output = inference.greedy_decode(src_ids, src_padding_mask, max_output_len=100)

# Beam search
beam_outputs = inference.beam_search_decode(
    src_ids, src_padding_mask, max_output_len=100, beam_size=5
)

# Sampling
sampled_output = inference.sample_decode(
    src_ids, src_padding_mask, max_output_len=100, temperature=0.8, top_k=50, top_p=0.9
)
```

### Running the Example

You can run the example script to see a complete demonstration:

```bash
python examples/basic_usage.py
```

### Debug Mode

You can enable debug mode to print tensor shapes and values during execution:

```python
config = TransformerConfig(
    # ... other parameters ...
    debug_mode=True,  # Enable debug printing
)
```

Run the debug printing example:

```bash
python examples/debug_printing_example.py
```

### Visualization

You can visualize attention weights and embeddings to better understand the model:

```python
config = TransformerConfig(
    # ... other parameters ...
    store_attention=True,  # Enable attention weight storage for visualization
)

model = Transformer(config)

# Run a forward pass to generate attention weights
output = model(src_ids, tgt_ids, src_padding_mask)

# Visualize encoder self-attention
fig = model.visualize_encoder_attention(
    layer_idx=0,  # First encoder layer
    head_idx=None,  # Average across all heads
    tokens=src_tokens  # Optional token labels
)

# Visualize decoder cross-attention
fig = model.visualize_decoder_cross_attention(
    layer_idx=0,  # First decoder layer
    head_idx=2,  # Third attention head
    src_tokens=src_tokens,  # Source token labels
    tgt_tokens=tgt_tokens  # Target token labels
)

# Visualize token embeddings using PCA
fig = model.visualize_encoder_embeddings(
    input_ids=src_ids,
    tokens=src_tokens
)

# Visualize all attention heads in a grid
fig = model.visualize_attention_heads(
    attention_type="encoder",  # "encoder", "decoder_self", or "decoder_cross"
    layer_idx=0
)
```

Run the visualization example:

```bash
python examples/visualization_example.py
```

## Components

- **Transformer**: The main model combining encoder and decoder
- **TransformerEncoder**: The encoder part of the transformer
- **TransformerDecoder**: The decoder part of the transformer
- **MultiHeadAttention**: Implementation of multi-head attention mechanism
- **TransformerConfig**: Configuration class for the transformer model
- **TransformerTrainer**: Utilities for training the model
- **TransformerInference**: Utilities for inference with different decoding strategies
- **TransformerDataset**: Dataset class for transformer sequence-to-sequence tasks
