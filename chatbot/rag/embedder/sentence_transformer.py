"""Pretrained sentence-transformer embedder (Hebrew-capable, prefix-free).

Defaults to a multilingual MiniLM model: small, fast, supports Hebrew, and
symmetric (no query/passage prefixes), so it fits the plain ``encode()``
interface. Swap ``model_name`` for a stronger model later (e.g. e5, which needs
``query:``/``passage:`` prefixes).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chatbot.rag.embedder.base import Embedder
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        # Imported lazily: keeps the heavy torch/transformers import off the
        # path of code that only needs the lightweight RandomEmbedder.

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)

    @property
    def fingerprint(self) -> str:
        # For a frozen HF model the name pins the weights. A local checkpoint
        # (e.g. a finetuned model) may keep the same path across finetunes, so
        # fold in its newest file mtime to catch an in-place overwrite that
        # leaves the name unchanged — otherwise stale embeddings get reused.
        parts = [type(self).__qualname__, self.model_name]
        local = Path(self.model_name)
        if local.exists():
            mtimes = [p.stat().st_mtime for p in local.rglob("*") if p.is_file()]
            if mtimes:
                parts.append(f"mtime={max(mtimes):.0f}")
        return ":".join(parts)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
