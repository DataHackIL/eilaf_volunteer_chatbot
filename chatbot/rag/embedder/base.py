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
