"""SpeciesStore — loads birdsong_data.json and exposes classify / list.

KNN classification uses the same centroid-distance approach as knn.js
so results are consistent between browser-side and server-side paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class SpeciesStore:
    """In-memory species index built from birdsong_data.json."""

    def __init__(self, data_path: Path) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._centroids: dict[str, np.ndarray] = {}

        if data_path.exists():
            with open(data_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            # Support both single-species and multi-species JSON
            if isinstance(raw.get("t"), list):
                key = str(raw.get("species", "birdsong"))
                raw = {key: raw}
            for key, manifold in raw.items():
                self._data[key] = manifold
                self._centroids[key] = _centroid(manifold["xyz"])
        else:
            import warnings
            warnings.warn(
                f"birdsong_data.json not found at {data_path}. "
                "Species list will be empty until the file is present.",
                stacklevel=2,
            )

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_metadata(self) -> list[dict[str, Any]]:
        """Return lightweight per-species metadata (no xyz/energy arrays)."""
        out = []
        for key, m in self._data.items():
            out.append(
                {
                    "species": key,
                    "duration_s": m.get("duration_s", 0.0),
                    "sr": m.get("sr", 22050),
                    "n_frames": len(m.get("t", [])),
                    "features_used": m.get("features_used", []),
                }
            )
        return sorted(out, key=lambda x: x["species"])

    def classify(
        self,
        manifold: dict[str, Any],
        k: int = 3,
    ) -> list[dict[str, Any]]:
        """Return top-k nearest species by centroid Euclidean distance.

        The query manifold centroid is compared against all stored
        species centroids. Matches `knn.js` classify() logic so
        server and browser scores are equivalent.

        Parameters
        ----------
        manifold:
            Dict with at least an "xyz" key (list of [x,y,z] triples).
        k:
            Number of results to return.

        Returns
        -------
        List of dicts: rank, species, distance, similarity_pct
        """
        if not self._centroids:
            return []

        query_centroid = _centroid(manifold["xyz"])

        scored: list[tuple[float, str]] = []
        for key, centroid in self._centroids.items():
            dist = float(np.linalg.norm(query_centroid - centroid))
            scored.append((dist, key))

        scored.sort(key=lambda x: x[0])
        top_k = scored[:k]

        return [
            {
                "rank": rank + 1,
                "species": species,
                "distance": round(dist, 6),
                "similarity_pct": _dist_to_similarity(dist),
            }
            for rank, (dist, species) in enumerate(top_k)
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroid(xyz: list[list[float]]) -> np.ndarray:
    """Mean of all 3D manifold points."""
    arr = np.asarray(xyz, dtype=np.float32)
    return arr.mean(axis=0)


def _dist_to_similarity(dist: float) -> int:
    """Map Euclidean distance [0, inf) → similarity percent [0, 100].

    Uses the same exponential decay as knn.js distToSimilarity():
        similarity = 100 * exp(-dist * 1.5)
    """
    return int(round(100.0 * np.exp(-dist * 1.5)))
