"""
Birdsong FastAPI backend.

Endpoints
---------
GET  /health             – liveness probe
GET  /api/species        – list all species with metadata
POST /api/process        – upload audio → manifold JSON
POST /api/classify       – manifold JSON → top-K nearest species
POST /api/describe       – classify result → LLM species description
POST /api/search         – natural language query → ranked species
POST /api/chat           – conversational birdsong assistant
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.pipeline import process_audio
from api.store import SpeciesStore
from api import llm

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Birdsong API",
    description="Acoustic manifold extraction, species classification, and LLM intelligence",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DATA_PATH = Path(os.environ.get("BIRDSONG_DATA_PATH", "birdsong_data.json"))
store = SpeciesStore(DATA_PATH)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ManifoldPayload(BaseModel):
    species: str
    sr: int
    hop_length: int
    duration_s: float
    features_used: list[str]
    t: list[float]
    xyz: list[list[float]]
    energy: list[float]
    spectral_centroid: list[float]


class ClassifyRequest(BaseModel):
    manifold: ManifoldPayload
    k: int = 3


class DescribeRequest(BaseModel):
    """Send the classify() output directly to get an LLM description."""
    species: str                          # top-ranked species name
    matches: list[dict[str, Any]]        # full classify() response


class SearchRequest(BaseModel):
    """Natural language bird search."""
    query: str


class ChatMessage(BaseModel):
    role: str   # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    """Multi-turn conversation. history[-1] is the latest user message."""
    history: list[ChatMessage]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _llm_check() -> None:
    """Raise 503 if Gemini API key is not configured."""
    if not llm._available():
        raise HTTPException(
            503,
            "LLM not available — set the GEMINI_API_KEY environment variable on Render",
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "species_loaded": str(len(store)),
        "llm": "ready" if llm._available() else "not configured",
    }


@app.get("/api/species", tags=["data"])
def list_species() -> list[dict[str, Any]]:
    return store.list_metadata()


@app.post("/api/process", tags=["audio"])
async def process(
    file: UploadFile = File(...),
    duration: float = 10.0,
) -> JSONResponse:
    """Upload audio → manifold JSON."""
    duration = min(max(duration, 1.0), 60.0)
    max_bytes = 50 * 1024 * 1024

    content_length = int(file.headers.get("content-length", 0))
    if content_length > max_bytes:
        raise HTTPException(413, "File too large (max 50 MB)")

    suffix = Path(file.filename or "upload").suffix.lower() or ".wav"
    if suffix not in {".mp3", ".wav", ".ogg", ".flac"}:
        raise HTTPException(415, f"Unsupported audio format: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        chunk_total = 0
        async for chunk in file:
            chunk_total += len(chunk)
            if chunk_total > max_bytes:
                raise HTTPException(413, "File too large (max 50 MB)")
            tmp.write(chunk)

    try:
        species_name = Path(file.filename or "upload").stem.lower().replace(" ", "_")
        manifold = process_audio(tmp_path, species=species_name, duration=duration)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, f"Processing failed: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    return JSONResponse(content=manifold)


@app.post("/api/classify", tags=["ml"])
def classify(body: ClassifyRequest) -> list[dict[str, Any]]:
    """Manifold → top-K nearest species."""
    k = max(1, min(body.k, len(store)))
    return store.classify(body.manifold.model_dump(), k=k)


@app.post("/api/describe", tags=["llm"])
async def describe(body: DescribeRequest) -> dict[str, str]:
    """
    Takes the output of /api/classify and returns an LLM-generated
    natural language description of the top species.

    Typical flow:
      1. POST /api/process  →  manifold
      2. POST /api/classify →  matches
      3. POST /api/describe →  { species, description }
    """
    _llm_check()
    try:
        text = await llm.describe(body.species, body.matches)
    except Exception as exc:
        raise HTTPException(502, f"LLM error: {exc}") from exc
    return {"species": body.species, "description": text}


@app.post("/api/search", tags=["llm"])
async def search(body: SearchRequest) -> list[dict[str, Any]]:
    """
    Natural language search over the species database.

    Example queries:
      - "birds that sing at dawn"
      - "melodic woodland birds"
      - "find me something similar to a robin"
    """
    _llm_check()
    species_names = [s["species"] for s in store.list_metadata()]
    try:
        results = await llm.search(body.query, species_names)
    except Exception as exc:
        raise HTTPException(502, f"LLM error: {exc}") from exc
    return results


@app.post("/api/chat", tags=["llm"])
async def chat(body: ChatRequest) -> dict[str, str]:
    """
    Multi-turn conversational birdsong assistant.

    Send the full conversation history each time.
    The assistant has context of all species in the database.

    Example body:
    {
      "history": [
        {"role": "user", "content": "What makes a blackbird's song distinctive?"}
      ]
    }
    """
    _llm_check()
    species_names = [s["species"] for s in store.list_metadata()]
    history = [m.model_dump() for m in body.history]
    try:
        reply = await llm.chat(history, species_names)
    except Exception as exc:
        raise HTTPException(502, f"LLM error: {exc}") from exc
    return {"role": "assistant", "content": reply}
