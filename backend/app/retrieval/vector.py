"""Vector store for appearance/semantic embeddings (cosine similarity).

Uses a pure-NumPy index so it works without extra binaries; the interface is
designed so a FAISS index can be swapped in later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._vectors: list[np.ndarray] = []

    @staticmethod
    def _normalize(vector) -> np.ndarray:
        v = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def add(self, key: str, vector) -> None:
        self._ids.append(key)
        self._vectors.append(self._normalize(vector))

    def search(self, vector, k: int = 5) -> list[tuple[str, float]]:
        if not self._vectors:
            return []
        query = self._normalize(vector)
        sims = np.stack(self._vectors) @ query
        order = np.argsort(-sims)[:k]
        return [(self._ids[int(i)], float(sims[int(i)])) for i in order]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "dim": self.dim,
                    "ids": self._ids,
                    "vectors": [v.tolist() for v in self._vectors],
                }
            )
        )

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.dim = data["dim"]
        self._ids = data["ids"]
        self._vectors = [np.asarray(v, dtype=np.float32) for v in data["vectors"]]

    def __len__(self) -> int:
        return len(self._ids)
