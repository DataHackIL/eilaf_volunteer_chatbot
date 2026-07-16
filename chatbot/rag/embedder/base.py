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
