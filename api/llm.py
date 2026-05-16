"""
api/llm.py
LLM integration using Groq (free tier, 30 RPM).

Three capabilities:
  1. describe(species, matches)        — generate a rich species description
                                          after classification
  2. search(query, species_list)       — natural language → ranked species
  3. chat(history, species_context)    — conversational birdsong assistant

The GROQ_API_KEY env var must be set (Render env vars / local .env).
If the key is absent the module still imports — endpoints return a
clear 503 rather than crashing at startup.
"""

from __future__ import annotations

import asyncio
import os
import json
import re
from typing import Any

import httpx

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

_MAX_RETRIES = 3
_RETRY_DELAY = 8


def _available() -> bool:
    return bool(GROQ_API_KEY)


def _sanitise_error(message: str) -> str:
    if GROQ_API_KEY:
        message = message.replace(GROQ_API_KEY, "[REDACTED]")
    message = re.sub(r"Bearer\s+[^'\s]+", "Bearer [REDACTED]", message)
    return message


async def _call(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Single call to Groq. Retries up to 3x on 429."""
    if not _available():
        raise RuntimeError("GROQ_API_KEY not set")

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(GROQ_URL, headers=headers, json=payload)
                if resp.status_code == 429:
                    await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as exc:
            sanitised = _sanitise_error(str(exc))
            if exc.response.status_code != 429:
                raise RuntimeError(sanitised) from exc
        except Exception as exc:
            raise RuntimeError(_sanitise_error(str(exc))) from exc

    raise RuntimeError("Too many requests — please wait a moment and try again.")


async def _call_chat(messages: list[dict], temperature: float = 0.7) -> str:
    """Multi-turn call to Groq using a full messages array."""
    if not _available():
        raise RuntimeError("GROQ_API_KEY not set")

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 512,
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(GROQ_URL, headers=headers, json=payload)
                if resp.status_code == 429:
                    await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as exc:
            sanitised = _sanitise_error(str(exc))
            if exc.response.status_code != 429:
                raise RuntimeError(sanitised) from exc
        except Exception as exc:
            raise RuntimeError(_sanitise_error(str(exc))) from exc

    raise RuntimeError("Too many requests — please wait a moment and try again.")


async def describe(species: str, matches: list[dict[str, Any]]) -> str:
    match_summary = ", ".join(
        f"{m['species']} ({m['similarity_pct']:.0f}%)"
        for m in matches[:3]
    )
    system = "You are an expert ornithologist and birdsong analyst. Be specific, vivid, and concise. Write as flowing prose, no bullet points."
    user = f"""An audio recording was uploaded and the acoustic classifier returned these matches: {match_summary}

The top match is: {species}

Write a concise engaging description (3-4 sentences) covering: what this bird looks like and where it lives, what its song sounds like, why it might acoustically resemble the other matched species, and one interesting fact."""
    return await _call(system, user, temperature=0.75)


async def search(query: str, species_list: list[str]) -> list[dict[str, Any]]:
    species_json = json.dumps(species_list)
    system = "You are a birdsong expert. Return only valid JSON, no markdown, no explanation."
    user = f"""Available species: {species_json}

User query: "{query}"

Return a JSON array of up to 5 objects, each with:
  - "species": exact name from the list
  - "relevance_pct": integer 0-100
  - "reason": one sentence

If no species match, return []."""

    raw = await _call(system, user, temperature=0.3)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]

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
    system_content = (
        f"You are Birdsong Assistant, an expert ornithologist AI. "
        f"The species database currently contains: {context_str}. "
        f"Answer questions about birds, their calls, habitats, and behaviours. "
        f"Be concise, accurate, and engaging. If asked about a species not in the "
        f"database, answer generally but note it isn't in the current collection."
    )

    messages = [{"role": "system", "content": system_content}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    return await _call_chat(messages, temperature=0.7)
