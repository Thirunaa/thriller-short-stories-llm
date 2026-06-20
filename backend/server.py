"""FastAPI server: story generation, feedback capture, monitoring, and control
of the continuous-improvement worker.

    uvicorn server:app --port 8000      (or: python server.py)
"""
from __future__ import annotations

import asyncio
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import data
import feedback
from config import DATA_DIR
from inference_service import service
from continuous import trainer


# ---------------------------------------------------------------------------
# Lifespan: init DB, load model, start the background trainer.
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    feedback.init_db()
    service.load_initial()
    trainer.start()
    yield
    trainer.stop()


app = FastAPI(title="Thriller Short-Story MiniGPT", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single worker: CPU autoregressive decoding is serialized anyway, and offloading
# to an explicit asyncio executor keeps the event loop responsive. Endpoints are
# all `async def` to avoid FastAPI's anyio threadpool dispatch (which the JAX/orbax
# stack breaks here by applying nest_asyncio to the running loop).
_EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")


async def _offload(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXEC, lambda: fn(*args))


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(200, ge=1, le=1000)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0)            # 0 disables top-k
    seed: Optional[int] = None


class FeedbackRequest(BaseModel):
    generation_id: str
    rating: str                              # "up" | "down"
    edited_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/status")
async def status():
    return {
        "model": service.status(),
        "feedback": feedback.stats(),
        "trainer": trainer.status(),
        "data": _data_meta(),
    }


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if service.model is None:
        raise HTTPException(503, "model not loaded")
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    top_k = req.top_k if req.top_k > 0 else None
    output = await _offload(
        service.generate, req.prompt, req.max_new_tokens, req.temperature, top_k, seed
    )
    params = {"max_new_tokens": req.max_new_tokens, "temperature": req.temperature,
              "top_k": req.top_k, "seed": seed}
    gen_id = feedback.add_generation(req.prompt, output, params, service.version)
    return {
        "generation_id": gen_id,
        "prompt": req.prompt,
        "output": output,
        "model_version": service.version,
        "model_source": service.source,
        "trained": service.source != "untrained",
        "params": params,
    }


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    rating = 1 if req.rating.lower() in ("up", "1", "+1", "good") else -1
    fb_id = feedback.add_feedback(req.generation_id, rating, req.edited_text)
    return {"feedback_id": fb_id, "rating": rating, "stats": feedback.stats()}


@app.post("/api/train/trigger")
async def trigger_training():
    return trainer.trigger_now()


@app.post("/api/model/reload")
async def reload_model():
    changed = await _offload(service.reload_latest)
    return {"reloaded": changed, "model": service.status()}


def _data_meta() -> dict:
    import json
    import os
    path = os.path.join(DATA_DIR, "meta.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"prepared": data.has_data()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
