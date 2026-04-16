# Implementing GPT-2 from Scratch — Complete Guide

A full walkthrough of building a 124M parameter GPT-2 model in PyTorch, covering every component from tokenization to text generation.

---

## Table of Contents

1. [GPT Configuration](#1-gpt-configuration)
2. [Architecture Overview](#2-architecture-overview)
3. [Tokenization](#3-tokenization)
4. [Embeddings](#4-embeddings)
5. [Layer Normalization](#5-layer-normalization)
6. [GELU Activation Function](#6-gelu-activation-function)
7. [FeedForward Network](#7-feedforward-network)
8. [Shortcut (Residual) Connections](#8-shortcut-residual-connections)
9. [Transformer Block](#9-transformer-block)
10. [Multi-Head Attention](#10-multi-head-attention)
11. [Full GPT Model](#11-full-gpt-model)
12. [Parameter Count & Memory](#12-parameter-count--memory)

---

## 1. GPT Configuration

Everything is driven by a single config dictionary. Pass this into every module.

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # GPT-2 BPE tokenizer vocabulary size
    "context_length": 1024, # Max tokens the model can process at once
    "emb_dim": 768,         # Embedding dimension (width of every layer)
    "n_heads": 12,          # Attention heads (768 / 12 = 64 dims per head)
    "n_layers": 12,         # Number of stacked transformer blocks
    "drop_rate": 0.1,       # Dropout rate for regularization
    "qkv_bias": False       # GPT-2 uses no bias in Q, K, V projections
}
```

`emb_dim=768` is the "spine" of the model — every token is a 768-dimensional vector at every stage inside the network.

---

## 2. Architecture Overview

```
Input text
    ↓
Tokenizer (tiktoken, GPT-2 BPE)       → [batch, seq_len]
    ↓
Token Embedding + Position Embedding  → [batch, seq_len, 768]
    ↓
Dropout
    ↓
┌─────────────────────────────────┐
│      Transformer Block × 12     │
│                                 │
│  LayerNorm → MultiHeadAttention │
│           ⊕ residual            │
│  LayerNorm → FeedForward(GELU)  │
│           ⊕ residual            │
└─────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
Linear Output Head                    → [batch, seq_len, 50257]
    ↓
Logits (one score per vocab token)
```

The shape `[batch, seq_len, 50257]` means: for every token position, the model outputs 50,257 scores — one per vocabulary item. The highest score = the model's next-token prediction.

---

## 3. Tokenization

```python
import tiktoken
import torch

tokenizer = tiktoken.get_encoding("gpt2")

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch = []
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)

print(batch)
# tensor([[6109, 3626, 6100,  345],
#         [6109, 1110, 6622,  257]])

print(batch.shape)  # torch.Size([2, 4])
```

`tiktoken` converts each word (or sub-word) into an integer ID from a vocabulary of 50,257 entries. The output shape `[2, 4]` means 2 sentences × 4 tokens each.

---

## 4. Embeddings

```python
import torch.nn as nn

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])      # 50257 × 768
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])  # 1024  × 768
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape        # [2, 4]

        tok_embeds = self.tok_emb(in_idx)         # [2, 4, 768]  — each token ID → vector
        pos_embeds = self.pos_emb(                # [4,    768]  — each position → vector
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_embeds + pos_embeds               # [2, 4, 768]  — add position info
        x = self.drop_emb(x)
        return x
```

**Why two embeddings?** Transformers have no built-in sense of order. The positional embedding injects position information by adding a learned position vector to each token vector. Position 0 looks different from position 3 even if the token ID is identical.

---

## 5. Layer Normalization

### Manual example first

```python
import torch

torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
output = layer(batch_example)

mean     = output.mean(dim=-1, keepdim=True)
variance = output.var(dim=-1, keepdim=True)

normalised_output = (output - mean) / torch.sqrt(variance)
```

`dim=-1` normalizes across the last dimension (embedding size), not across the batch. `keepdim=True` keeps the shape as `[2, 1]` instead of collapsing to `[2]`.

### Full LayerNorm class

```python
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))   # learnable γ (gamma)
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # learnable β (beta)

    def forward(self, x):
        mean         = x.mean(dim=-1, keepdim=True)
        variance     = x.var(dim=-1, keepdim=True, unbiased=False)
        normalised_x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.scale * normalised_x + self.shift
```

**Key details:**

- `eps = 1e-5` — prevents division by zero when variance is nearly 0.
- `scale` (γ) and `shift` (β) — learnable parameters that allow the model to undo normalization if it helps performance.
- `unbiased=False` — divides by `n` not `n-1`, matching GPT-2's original TensorFlow implementation.

```python
ln = LayerNorm(emb_dim=5)
output_ln = ln(batch_example)
print(output_ln.mean(dim=-1))     # ≈ 0.0 for every row
print(output_ln.var(dim=-1))      # ≈ 1.0 for every row
```

---

## 6. GELU Activation Function

```python
import math

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)
        ))
```

### GELU vs ReLU

| Property | ReLU | GELU |
|---|---|---|
| Formula | `max(0, x)` | `x * Φ(x)` (Gaussian CDF) |
| Negative inputs | Always 0 | Small non-zero output |
| Gradient at 0 | Undefined (hard edge) | Smooth, differentiable |
| Dead neurons | Can occur | Less likely |
| Used in GPT-2 | No | Yes |

```python
gelu, relu = GELU(), nn.ReLU()
x = torch.linspace(-5, 5, steps=100)

# ReLU: 0 for x < 0, linear for x >= 0
# GELU: smooth curve, slightly negative for small negative x,
#       approaches ReLU for large positive x
```

GELU's smooth gradient transition allows for more nuanced parameter updates during training, especially in deep networks like GPT-2.

---

## 7. FeedForward Network

```python
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 768 → 3072  EXPAND
            nn.GELU(),                                        # non-linearity
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 3072 → 768  CONTRACT
        )

    def forward(self, x):
        return self.layers(x)
```

```python
ffn = FeedForward(GPT_CONFIG_124M)
x   = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)  # torch.Size([2, 3, 768]) — shape is preserved
```

**Why 4× expansion?** The hidden layer (3072 dims) gives the model room to learn complex non-linear combinations of features before projecting back to 768. This is where much of the model's "knowledge" gets stored. The input and output shapes match so blocks can be stacked without dimension adjustments.

---

## 8. Shortcut (Residual) Connections

### Demonstrating the vanishing gradient problem

```python
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[i], layer_sizes[i+1]), nn.ReLU())
            for i in range(len(layer_sizes) - 1)
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and layer_output.shape == x.shape:
                x = layer_output + x   # ← residual: add original input back
            else:
                x = layer_output
        return x
```

```python
def print_gradients(model, x):
    output = model(x)
    target = torch.tensor([[0.0]])
    loss   = nn.MSELoss()(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} → gradient mean: {param.grad.abs().mean().item():.6f}")
```

### Results comparison

```
# WITHOUT shortcuts (use_shortcut=False):
layers.4.0.weight → gradient mean: 0.000345
layers.3.0.weight → gradient mean: 0.000060   ← shrinking
layers.2.0.weight → gradient mean: 0.000012   ← vanishing
layers.1.0.weight → gradient mean: 0.000002   ← almost dead
layers.0.0.weight → gradient mean: 0.000000   ← dead!

# WITH shortcuts (use_shortcut=True):
layers.4.0.weight → gradient mean: 0.000345
layers.3.0.weight → gradient mean: 0.000280   ← stable!
layers.2.0.weight → gradient mean: 0.000220   ← stable!
layers.1.0.weight → gradient mean: 0.000200   ← stable!
layers.0.0.weight → gradient mean: 0.000190   ← flows all the way!
```

Shortcut connections create a "gradient highway" — even if the learned path's gradient is tiny, the `+ x` term ensures a direct, unobstructed path for gradients back to early layers.

---

## 9. Transformer Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff           = FeedForward(cfg)
        self.norm1        = LayerNorm(cfg["emb_dim"])
        self.norm2        = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # ── Attention sub-block ─────────────────────────────
        shortcut = x                        # save input for residual
        x = self.norm1(x)                  # Pre-LayerNorm (normalize BEFORE attention)
        x = self.att(x)                    # multi-head self-attention
        x = self.drop_shortcut(x)          # dropout
        x = x + shortcut                   # ⊕ add residual

        # ── FeedForward sub-block ───────────────────────────
        shortcut = x                        # save for residual
        x = self.norm2(x)                  # Pre-LayerNorm (normalize BEFORE FFN)
        x = self.ff(x)                     # 768 → 3072 → 768 with GELU
        x = self.drop_shortcut(x)          # dropout
        x = x + shortcut                   # ⊕ add residual

        return x                            # shape [batch, seq_len, 768] — unchanged
```

```python
torch.manual_seed(123)
x      = torch.rand(2, 4, 768)             # [batch=2, tokens=4, emb=768]
block  = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape: ", x.shape)            # torch.Size([2, 4, 768])
print("Output shape:", output.shape)       # torch.Size([2, 4, 768]) — preserved!
```

**Pre-LayerNorm vs Post-LayerNorm:**

The notebook uses Pre-LayerNorm (normalize before each sub-layer). The original 2017 transformer paper used Post-LayerNorm (normalize after). Pre-LayerNorm leads to more stable training dynamics and is the standard in modern LLMs.

---

## 10. Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out    = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads   # 768 // 12 = 64 dims per head

        self.W_query  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key    = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value  = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)

        # Causal mask: upper triangle = True (masked out = future tokens)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)   # [b, num_tokens, d_out]
        keys    = self.W_key(x)     # [b, num_tokens, d_out]
        values  = self.W_value(x)   # [b, num_tokens, d_out]

        # Split d_out into num_heads × head_dim
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape after: [b, num_heads, num_tokens, head_dim]

        # Scaled dot-product attention
        attn_scores = queries @ keys.transpose(2, 3)   # [b, heads, tokens, tokens]

        # Apply causal mask: set future positions to -inf → softmax → 0
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        context_vec = (attn_weights @ values).transpose(1, 2)             # [b, tokens, heads, head_dim]
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  # merge heads
        context_vec = self.out_proj(context_vec)                          # final projection

        return context_vec
```

**Attention flow in plain English:**

1. Each token produces a Query (what am I looking for?), a Key (what do I offer?), and a Value (what info do I carry?).
2. Scores = `Query @ Key^T` — high score means token A is very relevant to token B.
3. Causal mask fills future-position scores with `-inf` so they become 0 after softmax (the model cannot "see ahead").
4. Weights = `softmax(scores / √64)` — normalizes scores to probabilities.
5. Output = weighted sum of Values — each token's output is a blend of all tokens it attended to.

---

## 11. Full GPT Model

```python
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb   = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb   = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb  = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]   # 12 blocks
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)      # passes through all 12 TransformerBlocks
        x = self.final_norm(x)
        logits = self.out_head(x)   # project to vocabulary space
        return logits
```

```python
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out   = model(batch)

print("Input batch:\n", batch)       # [2, 4] — 2 sentences, 4 tokens each
print("Output shape:", out.shape)    # torch.Size([2, 4, 50257])
```

The output `[2, 4, 50257]` means: for each of the 4 token positions in both sentences, the model produces 50,257 logit scores. Convert to text:

```python
predicted_token_ids = torch.argmax(out, dim=-1)   # [2, 4]
print(tokenizer.decode(predicted_token_ids[0].tolist()))   # decoded sentence 1
```

---

## 12. Parameter Count & Memory

```python
# Total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
# Output: 163,009,536  (≈ 163M)

# Why 163M, not 124M?
print("Token embedding shape: ", model.tok_emb.weight.shape)  # [50257, 768]
print("Output head shape:     ", model.out_head.weight.shape)  # [50257, 768]
# Both have the same shape → GPT-2 originally reuses tok_emb weights (weight tying)
# We count them separately, adding 38.6M extra params

# Subtract output head to get true GPT-2 count
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(f"Parameters (weight-tied): {total_params_gpt2:,}")
# Output: 124,412,160  ✓ matches "124M" in the model name

# Memory footprint (float32 = 4 bytes per parameter)
total_size_bytes = total_params * 4
total_size_mb    = total_size_bytes / (1024 * 1024)
print(f"Memory: {total_size_mb:.2f} MB")
# Output: ≈ 621.83 MB
```

### Parameter breakdown by component

| Component | Parameters | Notes |
|---|---|---|
| Token embedding | 38,597,376 | 50,257 × 768 |
| Position embedding | 786,432 | 1,024 × 768 |
| 12 × Transformer blocks | ~85M | Attention + FFN + norms |
| Final LayerNorm | 1,536 | scale + shift |
| Output head | 38,597,376 | 50,257 × 768 (same as tok_emb) |
| **Total** | **163,009,536** | **≈ 621 MB at float32** |

---

## Quick Reference — Class Summary

| Class | Purpose | Input → Output |
|---|---|---|
| `LayerNorm` | Normalizes activations per token | `[b, t, 768]` → `[b, t, 768]` |
| `GELU` | Smooth non-linear activation | `[any]` → `[any]` |
| `FeedForward` | Per-token MLP (expand + contract) | `[b, t, 768]` → `[b, t, 768]` |
| `MultiHeadAttention` | Cross-token information mixing | `[b, t, 768]` → `[b, t, 768]` |
| `TransformerBlock` | One full attention + FFN unit | `[b, t, 768]` → `[b, t, 768]` |
| `GPTModel` | Full pipeline (12 stacked blocks) | `[b, t]` → `[b, t, 50257]` |

---

## Dependencies

```bash
pip install torch tiktoken
```

```python
import torch
import torch.nn as nn
import tiktoken
import math
import matplotlib.pyplot as plt
```

---

*Based on the notebook `LLM_-_24.ipynb` — Implementing a GPT model from scratch to generate text.*
