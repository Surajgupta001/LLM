# 🧠 Multi-Head Attention (MHA) — Simple Explanation (PyTorch)

This document explains **Multi-Head Attention**, one of the most important concepts in Transformers and Large Language Models (LLMs) like GPT, in a very simple and beginner-friendly way.

---

## 🌟 Big Picture

At a glance:

* Input: token embeddings with shape (batch, tokens, d_in)
* Each head learns different relationships in parallel
* Output: combined context with shape (batch, tokens, d_out)

## 🧾 Notation

* d_in: input embedding size
* d_out: output embedding size
* num_heads: number of attention heads
* d_head: per-head dimension (d_out / num_heads)
* Q, K, V: query, key, and value projections

## ❓ What is Attention?

Attention lets a model decide how much to focus on other tokens while processing a token.

> **Which words are important when understanding a sentence?**

### Example

> "The animal didn't cross the street because **it** was tired."

Attention helps the model understand that **"it" refers to "animal"**.

---

## 🔥 What is Multi‑Head Attention?

Multi‑Head Attention runs several attention heads in parallel. Each head projects Q, K, and V, computes scaled dot‑product attention, and produces its own context. The head outputs are concatenated and mixed with a final projection.

Instead of looking at relationships in only **one way**, Multi‑Head Attention looks at them in **multiple ways at the same time**.

### Single Head Attention

Looks from one perspective.

### Multi‑Head Attention

Looks from many perspectives simultaneously.

### Example Perspectives

* Head 1 → Grammar
* Head 2 → Meaning
* Head 3 → Subject‑object relations
* Head 4 → Word position

Then all results are combined.

---

## 🧩 Why Multiple Heads?

Language is complex. One attention head cannot capture everything.

Multiple heads help capture:

* Syntax (grammar)
* Semantics (meaning)
* Context
* Long‑range dependencies

👉 More heads = richer understanding

---

## 🏗️ Code Structure (PyTorch Implementation)

## Class Definition

```python
class MultiHeadAttention(nn.Module):
```

This creates a custom neural network layer for Multi‑Head Attention.

---

## ⚙️ Parameters Explained

```python
def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
```

| Parameter      | Meaning                        |
| -------------- | ------------------------------ |
| d_in           | Input embedding size           |
| d_out          | Output embedding size          |
| context_length | Maximum sequence length        |
| dropout        | Prevents overfitting           |
| num_heads      | Number of attention heads      |
| qkv_bias       | Whether linear layers use bias |

---

## 🔑 Important Rule

```python
assert d_out % num_heads == 0
```

The output dimension must be divisible by the number of heads.

---

## ✂️ Splitting Dimensions

Each head gets a portion of the embedding.

```plain
head_dim = d_out / num_heads
```

## Example

* d_out = 768
* num_heads = 12
* head_dim = 64

Each head processes a 64‑dimensional vector.

---

## 🔍 Query, Key, Value (QKV)

Attention uses three matrices:

## Query (Q)

What we are searching for

## Key (K)

What we compare against

## Value (V)

Information we retrieve

---

## 🧠 Why QKV?

Think of a search engine:

* Query → Your search question
* Key → Indexed data
* Value → Actual result

---

## 🧮 Linear Layers for QKV

```python
self.W_query = nn.Linear(...)
self.W_key   = nn.Linear(...)
self.W_value = nn.Linear(...)
```

These layers convert input embeddings into Q, K, and V.

---

## 🔄 Reshaping for Multiple Heads

Original shape:

```plain
(batch, tokens, d_out)
```

After splitting into heads:

```plain
(batch, num_heads, tokens, head_dim)
```

This split is purely a reshape; no data is copied. This is done using:

* `.view()`
* `.transpose()`

Each head processes attention independently.

---

## ⚡ Attention Calculation

## Formula

$$
	ext{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d_{head}}}\right) V
$$

Here d_head is the per‑head dimension.

### Steps

1. Compare Query with Keys
2. Compute similarity scores
3. Normalize with Softmax
4. Multiply by Values

---

## 🧠 Causal Mask (Used in GPT)

Prevents the model from seeing future words.

Example:

Input: "I am learning"

When predicting "learning", the model cannot look ahead.

This ensures fair prediction during training.

---

## 🔗 Combining All Heads

After attention is computed for each head:

```plain
(batch, heads, tokens, head_dim)
```

We transpose back, then reshape to merge the head dimension. Combine back into:

```plain
(batch, tokens, d_out)
```

---

## 🧱 Output Projection Layer

```python
self.out_proj = nn.Linear(d_out, d_out)
```

This mixes information from all heads.

---

## 🤯 Why Multi‑Head Attention is Powerful

It allows the model to:

* Understand context
* Capture relationships between words
* Handle long sentences
* Resolve ambiguity

---

## 🧠 Real GPT‑2 Example

Small GPT‑2 model:

* 12 attention heads
* Embedding size: 768

Larger models use more heads and larger embeddings.

---

## 🪄 Simple Analogy

### 🎥 Multiple Cameras Watching the Same Scene

Each camera captures a different angle.

Combining all views gives a full understanding.

👉 That is Multi‑Head Attention.

---

## ❤️ What to Learn Next (LLM Roadmap)

To fully understand Transformers and GPT, learn:

* Self‑Attention
* Multi‑Head Attention
* Transformer Blocks
* Positional Encoding

---

## 🚀 Summary

Multi‑Head Attention is the core mechanism that allows modern AI models to understand language deeply by focusing on different aspects of a sentence at the same time.

---
