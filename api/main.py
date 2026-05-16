"""Birdsong FastAPI backend.

Endpoints
---------
GET  /health          – liveness probe (keep-alive cron target)
GET  /api/species     – list all species with metadata
POST /api/process     – upload audio → returns manifold JSON
POST /api/classify    – manifold JSON → top-K nearest species
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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Birdsong API",
    description="Acoustic manifold extraction and species classification",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Render/GitHub Pages domain in prod
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Species store – loaded once at startup from birdsong_data.json
# ---------------------------------------------------------------------------

DATA_PATH = Path(os.environ.get("BIRDSONG_DATA_PATH", "birdsong_data.json"))
store = SpeciesStore(DATA_PATH)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class ManifoldPayload(BaseModel):
    """Manifold JSON as produced by process_audio (or the browser pipeline)."""
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health() -> dict[str, str]:
    """Liveness probe. Point your Render keep-alive cron here."""
    return {"status": "ok", "species_loaded": str(len(store))}


@app.get("/api/species", tags=["data"])
def list_species() -> list[dict[str, Any]]:
    """Return all species with lightweight metadata (no full xyz arrays)."""
    return store.list_metadata()


@app.post("/api/process", tags=["audio"])
async def process(
    file: UploadFile = File(..., description="Audio file (.mp3 / .wav / .ogg / .flac)"),
    duration: float = 10.0,
) -> JSONResponse:
    """Upload an audio file, run the Python pipeline, return a manifold JSON.

    Parameters
    ----------
    file:
        The audio upload. Max 50 MB enforced by checking content-length header
        before reading; full body is read into a temp file for librosa.
    duration:
        Seconds to analyse (default 10). Values above 60 are clamped.
    """
    duration = min(max(duration, 1.0), 60.0)

    # Size guard – reject before reading entire body
    max_bytes = 50 * 1024 * 1024  # 50 MB
    content_length = int(file.headers.get("content-length", 0))
    if content_length > max_bytes:
        raise HTTPException(413, "File too large (max 50 MB)")

    suffix = Path(file.filename or "upload").suffix.lower() or ".wav"
    if suffix not in {".mp3", ".wav", ".ogg", ".flac"}:
        raise HTTPException(415, f"Unsupported audio format: {suffix}")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        chunk_total = 0
        async for chunk in file:  # streaming read
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
    """Compare a manifold against the species index, return top-K matches.

    Returns
    -------
    List of dicts with keys: rank, species, distance, similarity_pct
    """
    k = max(1, min(body.k, len(store)))
    results = store.classify(body.manifold.model_dump(), k=k)
    return results
