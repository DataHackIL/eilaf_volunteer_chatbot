"""Placeholder embedder that returns deterministic random vectors.

Stands in for a real (Hebrew-capable, finetunable) embedder so the pipeline can
run end-to-end today. The same text always maps to the same vector.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from chatbot.rag.embedder.base import Embedder


class RandomEmbedder(Embedder):
    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim = dim
        self.seed = seed

    @property
    def fingerprint(self) -> str:
        return f"{type(self).__qualname__}:dim={self.dim}:seed={self.seed}"

    def _vector_seed(self, text: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(self._vector_seed(text))
            vectors[i] = rng.standard_normal(self.dim)
        return vectors
