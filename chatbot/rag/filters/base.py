"""Metadata filter interface: re-scores retrieved chunks against user facts.

A "fact" is a closed-form answer the user optionally supplied (age, gender,
locality). A filter turns those facts plus each chunk's ``meta`` into an
additive adjustment to its similarity score, so retrieval can prefer — or drop
— chunks whose eligibility metadata (mis)matches the person being helped.
Swap implementations to change the soft/hard policy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from chatbot.rag.documents import Document


class MetadataFilter(ABC):
    @abstractmethod
    def adjust(self, documents: Sequence[Document], facts: dict) -> np.ndarray:
        """Return a ``(len(documents),)`` additive score adjustment.

        ``facts`` holds only the fields the user actually answered — a blank
        answer is absent, so it constrains nothing. A chunk whose ``meta`` lacks
        a field (NaN) is never penalised on that field. Contradictions may be
        soft (a finite penalty) or hard (``-inf``, i.e. excluded).
        """
        raise NotImplementedError
