# Transformer Minimal

A minimal, interview-friendly implementation of the Transformer architecture from scratch.

## Architecture

Standard Encoder-Decoder Transformer with:

- **Encoder**: Multi-head self-attention + feed-forward layers
- **Decoder**: Masked self-attention + cross-attention + feed-forward layers
- **Positional Encoding**: Learnable embeddings for position information

## Module Structure

- `self_attention.py` — Multi-head self-attention mechanism
- `cross_attention.py` — Multi-head cross-attention for decoder
- `encoder.py` — Transformer encoder stack
- `decoder.py` — Transformer decoder stack
- `transformer.py` — Full encoder-decoder model

## Quick Start

Run with `uv run` (dependencies are installed automatically):

```sh
uv run python -c "from transformer import Transformer; print(Transformer)"
```

## Example Usage

```python
import torch
from transformer import Transformer

# Initialize model
model = Transformer(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_enc_layers=6,
    num_dec_layers=6,
    vocab_size=10000,
    max_seq_len=512,
)

# Create dummy input (batch_size=2, seq_len=10)
src = torch.randint(0, 10000, (2, 10))
tgt = torch.randint(0, 10000, (2, 8))

# Forward pass
logits = model(src, tgt)  # Shape: (2, 8, 10000)
```

## Interview Focus

Key concepts to understand:

1. **Self-Attention**: How queries, keys, and values enable parallel token interaction
2. **Multi-Head Attention**: Why splitting into multiple heads improves representation
3. **Positional Encoding**: Why position information matters and how to inject it
4. **Cross-Attention**: How decoder accesses encoder information
5. **Masking**: Why causal masking is needed for autoregressive generation
6. **Layer Normalization & Residuals**: Pre-norm vs post-norm design choices
