"""Embedder interface: maps texts to vectors. Swap implementations freely."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Embedder(ABC):
    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` float32 array of embeddings."""
        raise NotImplementedError

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode corpus passages for indexing.

        Split from :meth:`encode_queries` for *asymmetric* models that treat a
        query and an indexed passage differently — e.g. e5's ``query:`` /
        ``passage:`` instruction prefixes. The default treats both alike, so
        symmetric embedders need only implement :meth:`encode`.
        """
        return self.encode(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Encode search queries. See :meth:`encode_documents`."""
        return self.encode(texts)

    @property
    def fingerprint(self) -> str:
        """Stable identity of this embedder's output space.

        Used to invalidate cached embeddings: two embedders sharing a
        fingerprint must produce interchangeable vectors for the same text, so
        override this to fold in anything that changes the vectors — model
        name, finetuned weights, encode config. The default (the class name) is
        enough only for an embedder with no output-affecting state.
        """
        return type(self).__qualname__
