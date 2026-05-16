"""Unit tests for the audio pipeline.

Run with:  pytest api/tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest


def _make_sine_wav(path: str, duration: float = 2.0, sr: int = 22050) -> None:
    """Write a simple sine-wave WAV file for testing."""
    import wave, struct, math  # noqa: E401

    n_samples = int(sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    # Mix two frequencies so chroma / MFCC are non-trivial
    signal = 0.5 * np.sin(2 * math.pi * 440 * t) + 0.3 * np.sin(2 * math.pi * 880 * t)
    signal = (signal * 32767).astype(np.int16)

    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal.tobytes())


def test_process_audio_returns_valid_schema(tmp_path):
    """process_audio should return a dict with all required keys."""
    wav_path = str(tmp_path / "test.wav")
    _make_sine_wav(wav_path, duration=3.0)

    from api.pipeline import process_audio

    result = process_audio(wav_path, species="test_bird", duration=3.0)

    required_keys = {
        "species", "sr", "hop_length", "duration_s",
        "features_used", "t", "xyz", "energy", "spectral_centroid",
    }
    assert required_keys.issubset(result.keys())
    assert result["species"] == "test_bird"
    assert len(result["xyz"]) == len(result["t"]) == len(result["energy"])
    assert all(len(pt) == 3 for pt in result["xyz"])
    assert result["sr"] == 22050


def test_process_audio_xyz_normalised(tmp_path):
    """All xyz coordinates should be in [-1, 1]."""
    wav_path = str(tmp_path / "test2.wav")
    _make_sine_wav(wav_path, duration=3.0)

    from api.pipeline import process_audio

    result = process_audio(wav_path, duration=3.0)
    coords = np.array(result["xyz"])
    assert coords.max() <= 1.0 + 1e-6
    assert coords.min() >= -1.0 - 1e-6


def test_process_audio_energy_clipped(tmp_path):
    """Energy values must all be in [0, 1]."""
    wav_path = str(tmp_path / "test3.wav")
    _make_sine_wav(wav_path, duration=3.0)

    from api.pipeline import process_audio

    result = process_audio(wav_path, duration=3.0)
    energy = np.array(result["energy"])
    assert energy.min() >= 0.0
    assert energy.max() <= 1.0


def test_process_audio_too_short_raises(tmp_path):
    """Very short audio should raise ValueError."""
    import wave, struct  # noqa: E401

    wav_path = str(tmp_path / "short.wav")
    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(struct.pack("<100h", *([0] * 100)))

    from api.pipeline import process_audio

    with pytest.raises(ValueError, match="too short"):
        process_audio(wav_path, duration=1.0)


def test_species_store_classify(tmp_path):
    """SpeciesStore.classify should return k results sorted by distance."""
    import json
    import math

    # Build a tiny 3-species data file
    n = 10
    data = {}
    for i, name in enumerate(["robin", "blackbird", "wren"]):
        xyz = [[math.sin(j + i), math.cos(j + i), 0.1 * j] for j in range(n)]
        data[name] = {
            "species": name, "sr": 22050, "hop_length": 512,
            "duration_s": 1.0, "features_used": [],
            "t": list(range(n)), "xyz": xyz,
            "energy": [0.5] * n, "spectral_centroid": [0.5] * n,
        }

    data_path = tmp_path / "test_data.json"
    data_path.write_text(json.dumps(data))

    from api.store import SpeciesStore

    store = SpeciesStore(data_path)
    assert len(store) == 3

    # Query identical to robin → robin should be rank 1
    results = store.classify(data["robin"], k=3)
    assert len(results) == 3
    assert results[0]["species"] == "robin"
    assert results[0]["rank"] == 1
    assert results[0]["distance"] < 1e-6
    assert all(r["similarity_pct"] >= 0 for r in results)
