# 🗡️ Thriller Forge — a from-scratch JAX LLM that improves from your feedback

A complete **build → train → deploy → continuously-improve** workflow for a small
GPT language model, end to end:

- **Model:** a MiniGPT (decoder-only transformer) written from scratch in **JAX + Flax NNX**
- **Training:** **Optax** optimizer, cosine LR schedule, **Orbax** checkpointing
- **Data:** the [`Nopm/Opus_WritingStruct`](https://huggingface.co/datasets/Nopm/Opus_WritingStruct)
  creative-writing dataset (chat-format conversations), tokenized with **tiktoken** (GPT-2 BPE)
- **Serving:** a **FastAPI** backend that loads a checkpoint and generates stories
- **Frontend:** a **React + Vite + Tailwind** story studio with a live monitoring panel
- **Continuous improvement:** human thumbs-up / edits become training data; a background
  worker fine-tunes from the latest checkpoint and **hot-swaps** the live model — no restart

```
 ┌──────────┐   prompt    ┌─────────────┐  generate   ┌──────────────┐
 │  React   │ ──────────► │   FastAPI   │ ──────────► │  MiniGPT     │
 │  Studio  │ ◄────────── │   server    │ ◄────────── │ (JAX/Flax)   │
 └────┬─────┘   story      └──────┬──────┘   text      └──────▲───────┘
      │ 👍 / ✎ edit               │ feedback                  │ hot-swap
      ▼                           ▼                           │
 ┌──────────┐            ┌─────────────┐  fine-tune   ┌───────┴───────┐
 │ feedback │ ─────────► │  SQLite     │ ──────────► │  Continuous   │
 │  + edits │            │  queue      │   samples   │  Trainer      │──► Orbax ckpt
 └──────────┘            └─────────────┘             └───────────────┘
```

---

## The continuous-improvement loop (the interesting part)

1. A user generates a story; it is logged to SQLite (`generations`).
2. The user rates it 👍 / 👎, or **edits & approves** an improved version.
3. A 👍 (using the edited text if provided) becomes a `(prompt → preferred story)`
   training pair (`feedback`, `used_in_training = 0`).
4. A background daemon (`continuous.py`) watches the queue. Once `min_new_samples`
   fresh pairs accumulate — or an admin clicks **⚡ Improve now** — it:
   - restores a fresh copy of the currently-served weights (the base),
   - fine-tunes them on the feedback pairs (gentle LR, few steps),
   - saves a **new versioned Orbax checkpoint**, and
   - **hot-swaps** it into the live `ModelService` (atomic, lock-guarded).
5. The next generation is served by the improved model. The version number ticks up.

This is preference-driven online fine-tuning with clean checkpoint lineage, rather
than unstable per-token online updates — so every improvement is reproducible and
roll-back-able.

---

## Project layout

```
backend/
  config.py            # model / train / continuous configs + paths
  tokenizer.py         # tiktoken GPT-2 BPE + chat formatting
  model.py             # MiniGPT in Flax NNX (attention, blocks, weight tying)
  data.py              # memmap token shards + batching
  prepare_data.py      # download HF dataset -> train.bin / val.bin
  train.py             # Optax training loop + pretrain CLI (shared train_step)
  checkpointing.py     # Orbax save/restore of NNX state + config sidecars
  generate.py          # temperature / top-k autoregressive sampling
  feedback.py          # SQLite store of generations + human feedback
  inference_service.py # in-memory model holder with atomic hot-swap
  continuous.py        # background fine-tune worker (the loop above)
  server.py            # FastAPI app
frontend/
  src/App.jsx          # story studio + monitor layout
  src/components/StoryCard.jsx     # one story + 👍/👎/edit controls
  src/components/MonitorPanel.jsx  # live model + feedback + trainer dashboard
```

---

## Quickstart

### 1. Backend

```bash
cd backend
pip install -r requirements.txt          # JAX (CPU), Flax, Optax, Orbax, FastAPI, ...

python prepare_data.py --max-rows 3000   # download + tokenize the dataset
python train.py --max-iters 2000         # pretrain the tiny model (writes a checkpoint)
python server.py                         # serve on http://localhost:8000
```

Or just run `./run_backend.ps1` (PowerShell) — it auto-prepares data + a checkpoint
if missing, then starts the server.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173  (proxies /api -> :8000)
```

Open the studio, forge a story, and rate it. Watch the **Continuous trainer** panel:
after enough 👍 it fine-tunes and the served **model version ticks up** automatically.

---

## Configuration

All knobs live in [`backend/config.py`](backend/config.py):

| Config | Key fields | Default (tiny / CPU) |
|---|---|---|
| `ModelConfig` | `n_layer, n_head, n_embd, block_size, tie_weights` | 4 / 4 / 128 / 128 / true |
| `TrainConfig` | `batch_size, learning_rate, max_iters, warmup_iters` | 16 / 3e-4 / 2000 / 100 |
| `ContinuousConfig` | `min_new_samples, poll_seconds, finetune_iters` | 8 / 30 / 60 |

The defaults are sized to **train on CPU in minutes**. To scale up (e.g. on a GPU box
via WSL2/Linux with a CUDA build of JAX), bump `n_layer/n_embd/block_size` and
`max_iters`, and pass `--n-embd 384 --n-layer 6 --block-size 256` to `train.py`.

> **Note on quality:** the tiny defaults at a couple thousand CPU iterations produce
> *structurally* GPT-like text but not polished prose. Train longer / larger for
> coherence — the architecture and pipeline are the same.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET`  | `/api/health` | liveness |
| `GET`  | `/api/status` | model version/source, feedback funnel, trainer state, data meta |
| `POST` | `/api/generate` | `{prompt, max_new_tokens, temperature, top_k}` → story + `generation_id` |
| `POST` | `/api/feedback` | `{generation_id, rating: "up"\|"down", edited_text?}` |
| `POST` | `/api/train/trigger` | schedule a fine-tune round now |
| `POST` | `/api/model/reload` | reload the newest checkpoint from disk |

---

## Environment notes

- Built & verified on **Windows 11, Python 3.14, JAX 0.10 (CPU)**.
- JAX on **native Windows is CPU-only**; for GPU use WSL2/Linux with a CUDA jaxlib.
- All endpoints are `async def` and offload blocking generation to an executor — this
  sidesteps a `nest_asyncio`/anyio interaction in the JAX+uvicorn stack on 3.14.
