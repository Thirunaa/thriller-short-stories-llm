# 🗡️ Thriller Forge — building a GPT language model from scratch

This repository contains a **complete, from-scratch large-language-model (LLM)
pipeline** — we wrote the neural network ourselves (no Hugging Face `transformers`,
no pre-trained weights), trained it on real data, and wrapped it in a web app that
keeps learning from human feedback.

This README is written so that **anyone who is ~18 and knows a little math (vectors,
matrix multiplication, derivatives, probability) and the basics of machine learning
(a neural net is layers of weights trained by gradient descent)** can understand
*exactly* what we built, how, and why — down to the individual matrix multiply.

> **TL;DR of the result:** a 51-million-parameter GPT we trained on an RTX 5090 now
> writes fully coherent, grammatical short stories. Open `http://localhost:5173`,
> type a story opening, and it continues it.

---

## Table of contents

1. [What is a language model, really?](#1-what-is-a-language-model-really)
2. [The big picture (system diagram)](#2-the-big-picture-system-diagram)
3. [Step 1 — Tokenization: turning text into numbers](#3-step-1--tokenization-turning-text-into-numbers)
4. [Step 2 — The model, atom by atom](#4-step-2--the-model-atom-by-atom)
5. [Step 3 — Training: how the model learns](#5-step-3--training-how-the-model-learns)
6. [Step 4 — Generation: how the model writes](#6-step-4--generation-how-the-model-writes)
7. [Step 5 — Serving + the feedback loop](#7-step-5--serving--the-continuous-improvement-loop)
8. [The datasets we trained on](#8-the-datasets-we-trained-on)
9. [The journey: 7M on a laptop → 51M on a GPU](#9-the-journey-7m-on-a-laptop--51m-on-a-gpu)
10. [Every file, explained](#10-every-file-explained)
11. [How to run it](#11-how-to-run-it)
12. [API reference](#12-api-reference)

---

## 1. What is a language model, really?

A language model does **one** thing: given some text, it predicts the **next word**
(more precisely, the next *token* — see §3). That's it. Everything else — writing
stories, answering questions — is that single skill applied over and over.

```
Input:  "It was a dark and stormy"
Model:  P(next token) = { " night": 0.71, " day": 0.04, " evening": 0.03, ... }
Pick:   " night"
Input:  "It was a dark and stormy night"   ← feed it back in, predict again
```

Doing this repeatedly ("**autoregressive** generation") produces whole stories. The
model is just a giant mathematical function with ~51 million tunable numbers
(**parameters / weights**). **Training** is the process of nudging those 51M numbers
until the model's next-token predictions match real text. We do that with
**gradient descent** (the same idea as fitting a line to points, just with millions
of dimensions).

The specific function we use is a **Transformer** (the "T" in GPT = Generative
**Pre-trained Transformer**). Its superpower is **self-attention**, explained in §4.

---

## 2. The big picture (system diagram)

```
                         ┌───────────────────────── TRAINING (offline) ──────────────────────────┐
                         │                                                                        │
  HuggingFace / IMDb /   │   raw text ──►  TOKENIZER  ──►  token ids ──►  train.bin / val.bin      │
  Wikipedia / Gutenberg  │  (stories)      (BPE)          [12, 318, ...]   (one big uint16 array)  │
                         │                                          │                             │
                         │                                          ▼                             │
                         │         ┌─────────────────────  TRAINING LOOP  ──────────────────────┐ │
                         │         │  get_batch → MiniGPT forward → cross-entropy loss →         │ │
                         │         │  backprop (gradients) → AdamW updates 51M weights → repeat  │ │
                         │         └──────────────────────────────┬──────────────────────────────┘ │
                         │                                        ▼                                 │
                         │                                  Orbax checkpoint  (weights on disk)     │
                         └────────────────────────────────────────┬───────────────────────────────┘
                                                                   │ load
                         ┌───────────────────────── SERVING (online) ─┼──────────────────────────┐
                         │                                            ▼                           │
   Browser (React)  ──prompt──►  FastAPI server  ──►  ModelService  ──►  MiniGPT.generate()       │
   localhost:5173    ◄─story───  localhost:8000   ◄──  (holds weights   ◄── autoregressive sampling│
        │  👍 / 👎 / ✎edit            │                 in memory)                                 │
        ▼                            ▼                                                            │
   feedback ───────────►  SQLite queue ──►  Continuous Trainer (background): fine-tune on liked    │
                                            stories → new checkpoint → hot-swap live model         │
                         └────────────────────────────────────────────────────────────────────────┘
```

Two halves: **training** turns text into weights (a checkpoint); **serving** loads
those weights and answers requests, while a background worker keeps improving them
from human feedback.

---

## 3. Step 1 — Tokenization: turning text into numbers

Neural networks only do math on numbers, so first we convert text into a list of
integers. We use **BPE (Byte-Pair Encoding)** via OpenAI's `tiktoken` (the same
"GPT-2" vocabulary). BPE splits text into ~50,000 common chunks called **tokens** —
frequent whole words get one token, rare words get split into pieces.

```
"The detective smiled."
        │ tiktoken (GPT-2 BPE)
        ▼
["The", " detective", " smiled", "."]   →   [464, 16838, 13541, 13]
```

- **Vocabulary size = 50,257** real tokens. In the model we round up to
  **`vocab_size = 50304`** (a multiple of 64) because GPUs multiply matrices faster
  when dimensions are "nice" numbers. The extra fake tokens are never emitted.
- A special token **`<|endoftext|>` (id 50256, called `EOT`)** marks the boundary
  between documents, so the model learns where stories start and end.

Code: [`backend/tokenizer.py`](backend/tokenizer.py) — `encode_ordinary(text)`,
`decode(ids)`, and the `EOT` constant.

> **Why tokens and not characters or words?** Characters make sequences too long;
> whole words make the vocabulary huge and can't handle new words. BPE is the
> sweet spot used by essentially all modern LLMs.

---

## 4. Step 2 — The model, atom by atom

Our network is **`MiniGPT`** in [`backend/model.py`](backend/model.py), written in
**JAX** (a NumPy-like library that runs on GPU and computes gradients automatically)
with **Flax NNX** (a small layer that organizes weights into modules).

### 4.0 The dimensions (current trained model)

| Symbol | Name | Value | Meaning |
|---|---|---|---|
| `V` | `vocab_size` | 50304 | number of possible tokens |
| `T` | `block_size` | 256 | **context length** — how many tokens it sees at once |
| `C` | `n_embd` | 512 | width of every internal vector ("embedding dim") |
| `L` | `n_layer` | 8 | number of Transformer blocks stacked |
| `H` | `n_head` | 8 | attention heads per block (head size = 512/8 = **64**) |
| `B` | `batch_size` | 16 | how many sequences we process in parallel while training |

These live in [`backend/config.py`](backend/config.py) (`ModelConfig`). Total
trainable parameters: **51,061,248 (~51M)**.

### 4.1 Embeddings — giving each token a meaning vector

Two lookup tables turn token ids into vectors:

- **Token embedding `wte`**: a `(50304 × 512)` matrix. Row *i* is the 512-number
  "meaning vector" for token *i*. `wte[idx]` looks up one row per token.
- **Position embedding `wpe`**: a `(256 × 512)` matrix giving each *slot* 0…255 its
  own vector — this is how the model knows token **order** (attention itself is
  order-blind).

```
x = wte[idx] + wpe[0..T-1]        # shape (B, T, 512)
```

Every token is now a 512-dim vector that encodes *what* it is and *where* it is.

### 4.2 Self-attention — the heart of the Transformer

Attention lets each token **look at every earlier token and pull in relevant
information**. Intuition: the word "it" should look back to find which noun it
refers to. Each token produces three vectors:

- **Query (Q)** — "what am I looking for?"
- **Key (K)** — "what do I offer?"
- **Value (V)** — "what information do I carry?"

A token compares its **Query** against every **Key** (a dot product = similarity);
high similarity means "pay attention here," and it then takes a weighted average of
those tokens' **Values**.

The math (one head), exactly as in `CausalSelfAttention.__call__`:

```
Q, K, V = split( x @ W_qkv )            # W_qkv is Linear 512 → 1536 (=3×512)
scores  = (Q @ Kᵀ) / √64                # (B, H, T, T) — every token vs every token
scores  = mask(scores)                  # causal mask: set "future" positions to −∞
weights = softmax(scores)               # each row sums to 1 (a probability dist.)
out     = weights @ V                   # weighted average of the Values
out     = out @ W_proj                  # Linear 512 → 512
```

- **`√64` scaling** keeps the dot products from getting huge (which would make
  softmax too "spiky").
- **Causal mask** (lower-triangular `jnp.tril`) is critical: token *t* may only
  attend to tokens `≤ t`. The model must predict the future, so it isn't allowed to
  *see* it. This is what makes it a **decoder-only / GPT** model.
- **Multi-head (H=8):** we run 8 attentions in parallel, each on a 64-dim slice, so
  different heads can specialize (one tracks subjects, another punctuation, …).

```
        token vectors (B,T,512)
              │
      ┌───────┴────────┐  Linear 512→1536
      Q (B,8,T,64)  K (B,8,T,64)  V (B,8,T,64)
      │              │             │
      └──► Q·Kᵀ/√64 ─┘             │     (B,8,T,T) similarity scores
              │ causal mask + softmax
              ▼
        weights (B,8,T,T) ──── × V ──►  (B,8,T,64) ──merge heads──► (B,T,512) ──Linear──► out
```

### 4.3 The MLP — per-token "thinking"

After mixing information across tokens, each token vector is processed
*independently* by a small 2-layer network (`MLP`):

```
h = GELU( x @ W_fc )      # Linear 512 → 2048  (expand 4×), GELU non-linearity
out = h @ W_proj          # Linear 2048 → 512  (project back)
```

GELU is a smooth activation function (like a soft ReLU). The 4× expansion gives the
model room to compute richer features.

### 4.4 LayerNorm + residuals — keeping training stable

Each block wraps attention and MLP in two standard tricks:

- **LayerNorm** normalizes a vector to mean 0 / variance 1 (then rescales). It keeps
  numbers in a sane range so training doesn't blow up. We use **pre-norm**
  (normalize *before* each sub-layer).
- **Residual connection** (`x = x + sublayer(norm(x))`): the block *adds* its output
  to its input instead of replacing it. This gives gradients a clean "highway" back
  through all 8 layers, which is what makes deep networks trainable.

One **Block** (`Block.__call__`), repeated **L = 8** times:

```
x = x + Attention( LayerNorm(x) )
x = x + MLP(       LayerNorm(x) )
```

### 4.5 The output head — back to vocabulary

After the 8 blocks and a final LayerNorm, we turn each 512-dim vector back into a
score for every possible next token:

```
logits = x @ wteᵀ          # (B, T, 512) @ (512, 50304) = (B, T, 50304)
```

**Weight tying** (`tie_weights = True`): we reuse the **same** token-embedding matrix
`wte` for the output projection (just transposed). This saves ~25M parameters and
usually helps quality — the matrix that maps *token → vector* is the natural inverse
of *vector → token*.

### 4.6 The whole forward pass (atomic diagram)

```
idx  (B, T)  int token ids
  │
  ├─ wte[idx]              → (B,T,512)   token meanings
  ├─ + wpe[0..T-1]         → (B,T,512)   + positions
  ▼
x (B,T,512)
  │
  │   ┌──────────── Block 1 ────────────┐
  ├──►│ x += Attn(LayerNorm(x))         │   ← mixes info across tokens (causal)
  │   │ x += MLP (LayerNorm(x))         │   ← per-token processing
  │   └─────────────────────────────────┘
  │     ⋮  (Blocks 2 … 8, identical shape)
  ▼
ln_f LayerNorm                → (B,T,512)
  │
  ▼
logits = x @ wteᵀ             → (B,T,50304)   score for every next token, at every position
```

Parameter budget (why it's ~51M):

```
token+output embedding (tied)  50304 × 512                 = 25.8M
position embedding             256 × 512                   =  0.1M
per block: attn(512×1536 + 512×512) + mlp(512×2048 ×2)     ≈  3.15M  ×8 = 25.2M
final layernorm                                            ≈  0.001M
                                                           ───────────────
                                                    total  ≈ 51.1M ✓
```

---

## 5. Step 3 — Training: how the model learns

Code: [`backend/train.py`](backend/train.py) (loop) + [`backend/data.py`](backend/data.py)
(batching) + [`backend/checkpointing.py`](backend/checkpointing.py) (saving).

### 5.1 The data on disk

`prepare_*.py` tokenizes all training text into one giant flat array of token ids
and saves it as **`train.bin` / `val.bin`** (`uint16`, memory-mapped so we never
load it all into RAM). Our current shards hold **~105 million tokens**.

### 5.2 A training batch

`get_batch` picks `B` random start positions and slices out windows of length `T`.
The **target `y` is the input `x` shifted by one token** — because the task is "given
tokens `0…t`, predict token `t+1`", *for every position at once*:

```
x = tokens[i        : i+256]      # (B, 256)
y = tokens[i+1      : i+257]      # (B, 256)  ← x shifted left by 1
```

### 5.3 The loss — how wrong were we?

The model outputs `logits (B,T,50304)`. We turn them into probabilities with
**softmax** and measure the **cross-entropy** against the true next token:

```
loss = −mean over all positions of  log P(correct next token)
     = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
```

Loss = 0 would mean perfect prediction. **Perplexity = e^loss** is the intuitive
version: "on average the model was as unsure as if choosing uniformly among this
many tokens." Our final model: loss **1.86 → perplexity ≈ 6.4** (it effectively
narrows ~50,000 options down to ~6 — that's why it's coherent).

### 5.4 Gradient descent with AdamW

`nnx.value_and_grad` computes the **gradient** of the loss w.r.t. all 51M weights
(JAX does this automatically via backprop). The **optimizer** (`optax.adamw`) nudges
each weight a little in the direction that lowers the loss. Key knobs (`TrainConfig`):

- **learning rate** `3e-4` with a **warmup + cosine decay** schedule (start small,
  rise, then smoothly shrink to near zero) — standard for stable Transformer training.
- **AdamW** = Adam (adaptive per-weight step sizes) + **weight decay** `0.1`
  (gently pulls weights toward 0 to reduce overfitting).
- **gradient clipping** at norm 1.0 (caps over-large updates).
- **dropout** `0.1` (randomly zeroes 10% of activations during training so the model
  can't memorize / overfit).

One training step (`train_step`, JIT-compiled by JAX for speed):

```
get_batch → logits = model(x) → loss = cross_entropy(logits, y)
          → grads = ∂loss/∂weights → AdamW: weights -= lr · adam(grads)
```

Repeat ~16,000 times. Every 1,000 steps we measure loss on held-out `val.bin`; if
it improved, we **save a checkpoint** (Orbax writes the weights + a `model_config.json`
so we can rebuild the exact architecture later). We keep the **best-val** checkpoint
(early stopping — guards against overfitting).

---

## 6. Step 4 — Generation: how the model writes

Code: [`backend/generate.py`](backend/generate.py).

To write, we run the model **one token at a time**, feeding each new token back in
(autoregressive). The next-token probabilities are shaped by three controls:

- **Temperature** (e.g. 0.7): divides the logits before softmax. Low = safe/repetitive,
  high = creative/random.
- **Top-k** (e.g. 40): only sample from the 40 most likely tokens (ignore the long
  tail of nonsense).
- **Repetition penalty** (1.3): down-weights tokens used in the last ~48 steps. This
  stops a small model from falling into "the the the…" loops.

```
prompt → tokens → [ forward → logits at last position → temperature → top-k
                    → repetition penalty → sample one token ] → append → repeat
                  → stop at <|endoftext|> or max_new_tokens → decode → text
```

**Two real engineering details** (both fixed bugs from the build):

1. **Fixed-size right-padded buffer + `logits_at(idx, pos)`.** The JIT-compiled
   forward needs a constant input shape, so we keep a `(1, 256)` buffer with the
   prompt at positions `0…L-1` and read logits at the **last real position** `pos`.
   The causal mask makes the model ignore the empty padding to the right —
   automatically correct, and it compiles **once**. (An earlier version left-padded
   with `EOT` tokens, which was out-of-distribution and produced garbage.)
2. **Only the last position's logits** are computed during decoding
   (`model.logits_at`) — the 50k-wide output projection over all 256 positions would
   be ~256× more compute than we need per step.

---

## 7. Step 5 — Serving + the continuous-improvement loop

### 7.1 Serving

- [`backend/inference_service.py`](backend/inference_service.py) — `ModelService`
  holds the live model in memory, picks the **best checkpoint** to serve (newest
  fine-tune, else newest pretrain), and supports an **atomic hot-swap** so we can
  replace the model without dropping requests.
- [`backend/server.py`](backend/server.py) — a **FastAPI** web server. Endpoints are
  `async` and offload the (blocking) generation to a thread pool so the server stays
  responsive. Generation requests are logged to SQLite.
- [`frontend/`](frontend/) — a **React + Vite + Tailwind** "story studio": a prompt
  box + sampling controls, the generated story with 👍/👎/✎-edit buttons, and a live
  **monitoring panel** (model version, params, val loss, feedback counts, trainer
  status) that polls `/api/status`.

### 7.2 The feedback loop (closing the circle)

[`backend/continuous.py`](backend/continuous.py) + [`backend/feedback.py`](backend/feedback.py).

```
1. You generate a story        → logged to SQLite (generations table)
2. You 👍 it (or ✎ edit+approve) → stored as a "preferred story" (feedback table)
3. A background daemon waits until ≥ N liked stories accumulate (or you click
   "⚡ Improve now")
4. It restores the current weights, fine-tunes them on those liked stories
   (gentle learning rate, few steps), saves a NEW checkpoint, and HOT-SWAPS it in
5. The served model version ticks up — future stories come from the improved model
```

This is **preference-driven fine-tuning** with clean, versioned checkpoints — every
improvement is reproducible and roll-back-able, not a fragile live edit.

---

## 8. The datasets we trained on

All data is built by scripts and cached under `backend/data_cache/` (git-ignored).
We use only **freely available / public-domain** sources, and for IMDb we use its
**official downloadable datasets** (we do *not* scrape the website).

| Corpus | Source | Built by | What it teaches |
|---|---|---|---|
| **TinyStories** ⭐ | [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories) (HuggingFace) | `prepare_tinystories.py` | **Coherent grammar.** Simple, clean GPT-4-written kids' stories — the proven signal for teaching a *small* model to write fluently. ~100M tokens. |
| **Thriller plots** | IMDb official datasets → ranked → CMU corpus + Wikipedia plot sections | `datagen/imdb_select.py`, `plots_cmu.py`, `wiki.py`, `build_corpus.py` | Thriller/horror vocabulary & plot structure (1,970 movies + 5,245 series episodes). |
| **Classic prose** | Project Gutenberg (Dracula, Sherlock Holmes, Poe, Jekyll & Hyde, The Moonstone…) | `datagen/gutenberg.py` | Real narrative prose: dialogue, paragraphs, sentence flow. |

The current model is trained mostly on **TinyStories** (for coherence) with a light
~5% thriller/prose mix (for flavor) — **~105M tokens total, 517k documents**. This
choice — *coherence over genre* — is why the latest model finally writes clean
grammar (see §9).

**Cleaning matters a lot for a small model.** [`backend/textclean.py`](backend/textclean.py)
strips leftover wiki markup, citations, HTML; `gutenberg.py` removes license
boilerplate / tables-of-contents and fixes text encoding. Garbage in the data wastes
the model's limited capacity.

---

## 9. The journey: 7M on a laptop → 51M on a GPU

We didn't get here in one shot. Each row was a real iteration in this project, and
the lesson each time was usually **"data and scale, not architecture."**

| # | Params | Training data | Context | Hardware | Val loss | Perplexity | Output quality |
|---|---|---|---|---|---|---|---|
| 1 | 7M  | generic writing chats | 128 | CPU | ~6.2 | ~490 | barely structured |
| 2 | 15M | thriller plots + Gutenberg | 128 | CPU | 6.05 | ~425 | thriller *words*, broken grammar |
| 3 | 34M | + more prose | **512** | **GPU** | 5.06 | ~158 | fluent, dialogue-rich, dark |
| 4 | **51M** | **TinyStories** + light mix | 256 | **GPU** | **1.86** | **~6.4** | **coherent, grammatical stories** |

Key turning points (all real fixes in the commit history):

- **Chat-format → raw story LM.** Wrapping every story in `### User/### Assistant`
  made the tiny model overfit the *template* and emit tag-salad. Training on raw
  prose fixed it.
- **The generation buffer bug.** Left-padding with `EOT` was out-of-distribution and
  produced newline/garbage loops. The right-padded buffer (§6) fixed generation for
  *every* model.
- **CPU → GPU (WSL2).** JAX on native Windows is CPU-only. We installed Ubuntu in
  **WSL2** and `jax[cuda12]` to use the **RTX 5090 (Blackwell)** — **~58× faster**
  training, which made the bigger models and bigger context windows practical.
- **The data was the grammar fix.** Plot summaries (telegraphic) and Gutenberg
  (archaic) couldn't teach clean modern grammar to a small model. **TinyStories**
  did — perplexity dropped from ~158 to ~6.4. Trade-off: the prose is simpler/gentler.

---

## 10. Every file, explained

### Backend — the model & training (`backend/`)

| File | What it does |
|---|---|
| [`config.py`](backend/config.py) | `ModelConfig` (architecture), `TrainConfig` (optimizer schedule), `ContinuousConfig` (feedback loop), and all file paths. |
| [`tokenizer.py`](backend/tokenizer.py) | tiktoken GPT-2 BPE wrapper: `encode_ordinary` / `decode` / `EOT`. |
| [`model.py`](backend/model.py) | **The neural network.** `CausalSelfAttention`, `MLP`, `Block`, `MiniGPT` (forward pass, `logits_at` for fast decoding), `param_count`. |
| [`data.py`](backend/data.py) | Memory-maps `train.bin`/`val.bin`; `get_batch` builds (x, y) training pairs. |
| [`train.py`](backend/train.py) | `make_optimizer` (AdamW + warmup-cosine), `train_step` (one JIT step), `run_training` (the loop + checkpointing + early stop). Also the CLI. |
| [`checkpointing.py`](backend/checkpointing.py) | `Checkpointer`: Orbax save/restore of model weights + `model_config.json` + `meta.json`; keeps the best 3. |
| [`generate.py`](backend/generate.py) | Autoregressive sampling: temperature, top-k, repetition penalty, the right-padded decode buffer. |
| [`inference_service.py`](backend/inference_service.py) | `ModelService`: holds live weights, chooses best checkpoint, atomic hot-swap, thread-safe `generate`. |
| [`feedback.py`](backend/feedback.py) | SQLite store: `generations` + `feedback` tables, queue queries, stats. |
| [`continuous.py`](backend/continuous.py) | Background daemon that fine-tunes on liked stories and hot-swaps the model. |
| [`server.py`](backend/server.py) | FastAPI app: `/api/generate`, `/api/feedback`, `/api/status`, `/api/train/trigger`, `/api/model/reload`. |
| [`prepare_data.py`](backend/prepare_data.py) | Tokenizes the local thriller/prose `*_corpus.jsonl` into shards (story-mode, document-shuffled). |
| [`prepare_tinystories.py`](backend/prepare_tinystories.py) | Streams TinyStories from HF + light thriller mix → shards. |
| [`textclean.py`](backend/textclean.py) | Robust text cleaner (markup/refs/HTML, unicode-normalize, dedup). |

### Backend — the data pipeline (`backend/datagen/`)

| File | What it does |
|---|---|
| [`imdb_select.py`](backend/datagen/imdb_select.py) | Downloads IMDb official datasets; ranks best thriller/horror movies & series by an IMDb-style weighted rating. |
| [`plots_cmu.py`](backend/datagen/plots_cmu.py) | CMU Movie Summary Corpus → fast `(title, year) → plot` lookup. |
| [`wiki.py`](backend/datagen/wiki.py) | Keyless Wikipedia: film "Plot" sections + episode-table/season summaries. |
| [`build_corpus.py`](backend/datagen/build_corpus.py) | Orchestrates plots + episodes → `thriller_corpus.jsonl` (resumable, cached). |
| [`gutenberg.py`](backend/datagen/gutenberg.py) | Public-domain classics → cleaned `prose_corpus.jsonl`. |
| [`common.py`](backend/datagen/common.py) | Shared HTTP, paths, title-normalization, caching helpers. |

### Frontend (`frontend/src/`)

| File | What it does |
|---|---|
| [`App.jsx`](frontend/src/App.jsx) | The studio: prompt box, sampling controls, story list, status polling. |
| [`components/StoryCard.jsx`](frontend/src/components/StoryCard.jsx) | One generated story + 👍/👎/✎-edit feedback controls. |
| [`components/MonitorPanel.jsx`](frontend/src/components/MonitorPanel.jsx) | Live dashboard: model version/params/val-loss, feedback funnel, trainer state, "⚡ Improve now". |
| [`api.js`](frontend/src/api.js) | Thin `fetch` wrapper around the API. |

---

## 11. How to run it

### Backend (CPU — works anywhere)

```bash
cd backend
pip install -r requirements.txt
python prepare_tinystories.py --ts-tokens 20000000   # or: python prepare_data.py
python train.py --max-iters 2000                     # writes a checkpoint
python server.py                                     # http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api → :8000)
```

### GPU training (NVIDIA, via WSL2) — what produced the 51M model

```bash
wsl --install -d Ubuntu-24.04                        # one-time
python3 -m venv ~/llm-venv
~/llm-venv/bin/pip install "jax[cuda12]" flax optax orbax-checkpoint tiktoken \
    numpy pandas fastapi "uvicorn[standard]" python-multipart
cd /mnt/c/.../backend
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
~/llm-venv/bin/python prepare_tinystories.py --ts-tokens 100000000
~/llm-venv/bin/python train.py --n-embd 512 --n-layer 8 --n-head 8 \
    --block-size 256 --dropout 0.1 --batch-size 16 --max-iters 16000 --eval-interval 1000
# serve on the GPU (WSL2 forwards localhost:8000 to Windows, so the Vite proxy works):
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 ~/llm-venv/bin/python server.py
```

`config.py` defaults are sized for CPU; pass `--n-embd/--n-layer/--block-size/...`
to `train.py` to scale up. The architecture is identical at every size.

---

## 12. API reference

| Method | Route | Purpose |
|---|---|---|
| `GET`  | `/api/health` | liveness |
| `GET`  | `/api/status` | model version/params/val-loss, feedback funnel, trainer state, data meta |
| `POST` | `/api/generate` | `{prompt, max_new_tokens, temperature, top_k}` → story + `generation_id` |
| `POST` | `/api/feedback` | `{generation_id, rating: "up"\|"down", edited_text?}` |
| `POST` | `/api/train/trigger` | schedule a feedback fine-tune round now |
| `POST` | `/api/model/reload` | reload the newest checkpoint from disk |

---

## Tech stack

**Model/training:** JAX + Flax NNX (network), Optax (optimizer), Orbax (checkpoints),
tiktoken (tokenizer). **Serving:** FastAPI + Uvicorn, SQLite. **Frontend:** React +
Vite + Tailwind. **Hardware:** CPU for the small models; NVIDIA RTX 5090 via WSL2 +
CUDA JAX for the 51M model.

> **Honest limitations.** This is a ~51M-parameter model (real LLMs are 1,000–10,000×
> bigger) trained on simple stories. It writes *coherent, grammatical* short
> narratives but doesn't plan long plots or hold deep world knowledge. It's built to
> *teach and demonstrate* the full LLM lifecycle — architecture, data, training,
> serving, and a human-feedback loop — end to end, from scratch.
