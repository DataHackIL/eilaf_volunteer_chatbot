"""E5 embedder: multilingual-e5 with ``query:`` / ``passage:`` prefixes.

The e5 family is *asymmetric*: it expects a short instruction prefix that differs
between a search query and an indexed passage. Retrieval quality drops sharply if
the prefixes are omitted or swapped, so queries and documents must go through the
dedicated :meth:`encode_queries` / :meth:`encode_documents` paths (which the
pipeline and the embedding cache use) rather than the bare :meth:`encode`.

Defaults to ``multilingual-e5-large`` — stronger on Hebrew legal text than the
MiniLM baseline and, crucially, a 512-token window (vs MiniLM's 128), so the
coarsened ~500-char chunks embed in full. It runs on CPU; the only real cost is
the one-time corpus encode, which :mod:`chatbot.rag.pipeline.embedding_cache`
persists. Swap ``model_name`` for ``multilingual-e5-base`` to trade some accuracy
for a faster encode while iterating.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from chatbot.rag.embedder.sentence_transformer import SentenceTransformerEmbedder

DEFAULT_MODEL = "intfloat/multilingual-e5-large"


class E5Embedder(SentenceTransformerEmbedder):
    """Prefix-aware sentence-transformer for the e5 model family.

    Inherits model loading, normalisation and the (model-name-based) fingerprint
    from :class:`SentenceTransformerEmbedder`; the class name folds into that
    fingerprint too, so switching to/from e5 invalidates the cached embeddings.
    """

    QUERY_PREFIX = "query: "
    PASSAGE_PREFIX = "passage: "

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        super().__init__(model_name, device)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_prefixed(texts, self.QUERY_PREFIX)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode_prefixed(texts, self.PASSAGE_PREFIX)

    def _encode_prefixed(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        return super().encode([f"{prefix}{text}" for text in texts])
