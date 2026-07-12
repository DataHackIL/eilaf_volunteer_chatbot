"""Pretrained sentence-transformer embedder (Hebrew-capable, prefix-free).

Defaults to a multilingual MiniLM model: small, fast, supports Hebrew, and
symmetric (no query/passage prefixes), so it fits the plain ``encode()``
interface. Swap ``model_name`` for a stronger model later (e.g. e5, which needs
``query:``/``passage:`` prefixes).
"""

from __future__ import annotations

from collections.abc import Sequence

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

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
