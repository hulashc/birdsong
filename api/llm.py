"""
api/llm.py
LLM integration using Google Gemini Flash Lite (free tier).

Three capabilities:
  1. describe(species, matches)        — generate a rich species description
                                          after classification
  2. search(query, species_list)       — natural language → ranked species
  3. chat(history, species_context)    — conversational birdsong assistant

The GEMINI_API_KEY env var must be set (Render env vars / local .env).
If the key is absent the module still imports — endpoints return a
clear 503 rather than crashing at startup.
"""

from __future__ import annotations

import os
import json
from typing import Any

import httpx

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash-lite:generateContent"
)


def _available() -> bool:
    return bool(GEMINI_API_KEY)


async def _call(prompt: str, temperature: float = 0.7) -> str:
    """Single-turn call to Gemini Flash Lite. Returns the text response."""
    if not _available():
        raise RuntimeError("GEMINI_API_KEY not set")

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 512,
        },
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc


async def describe(species: str, matches: list[dict[str, Any]]) -> str:
    match_summary = ", ".join(
        f"{m['species']} ({m['similarity_pct']:.0f}%)"
        for m in matches[:3]
    )
    prompt = f"""You are an expert ornithologist and birdsong analyst.

An audio recording was uploaded and the acoustic classifier returned these matches:
{match_summary}

The top match is: {species}

Write a concise, engaging description (3-4 sentences) covering:
- What this bird looks like and where it lives
- What its song or call sounds like (use evocative language)
- Why it might acoustically resemble the other matched species
- One interesting fact about this bird

Be specific and vivid. Do not use bullet points. Write as flowing prose."""

    return await _call(prompt, temperature=0.75)


async def search(query: str, species_list: list[str]) -> list[dict[str, Any]]:
    species_json = json.dumps(species_list)
    prompt = f"""You are a birdsong expert assistant.

Available species in the database: {species_json}

User query: "{query}"

Return a JSON array of objects ranking the most relevant species for this query.
Each object must have:
  - "species": the species name (must exactly match one from the list above)
  - "relevance_pct": integer 0-100 indicating relevance to the query
  - "reason": one sentence explaining why this species matches

Return ONLY valid JSON, no markdown, no explanation outside the JSON.
Include at most 5 results. If no species match, return an empty array []."""

    raw = await _call(prompt, temperature=0.3)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]

    try:
        results = json.loads(raw)
        validated = [
            {
                "species": r["species"],
                "relevance_pct": int(r["relevance_pct"]),
                "reason": r["reason"],
            }
            for r in results
            if r.get("species") in species_list
        ]
        return sorted(validated, key=lambda x: x["relevance_pct"], reverse=True)
    except (json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"LLM returned invalid JSON: {raw}") from exc


async def chat(
    history: list[dict[str, str]],
    species_context: list[str],
) -> str:
    context_str = ", ".join(species_context)
    system = (
        f"You are Birdsong Assistant, an expert ornithologist AI. "
        f"The species database currently contains: {context_str}. "
        f"Answer questions about birds, their calls, habitats, and behaviours. "
        f"Be concise, accurate, and engaging. If asked about a species not in the "
        f"database, answer generally but note it isn't in the current collection."
    )

    conversation = f"System: {system}\n\n"
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        conversation += f"{role}: {turn['content']}\n"
    conversation += "Assistant:"

    return await _call(conversation, temperature=0.7)
