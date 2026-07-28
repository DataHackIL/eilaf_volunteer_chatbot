"""Reranker interface: re-scores retrieved candidates by lexical relevance.

The embedder ranks the whole corpus by *semantic* similarity; a reranker then
re-orders a small candidate pool by a second, cheaper signal — here BM25 lexical
overlap — so exact-term matches (statute numbers, named benefits, Hebrew terms
the embedder blurs together) float back to the top. The same per-document scores
double as a relevance gauge: threshold them to estimate retrieval precision (see
:mod:`chatbot.rag.eval.precision`). Swap implementations freely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from chatbot.rag.documents import Document


class Reranker(ABC):
    @abstractmethod
    def score(self, query: str, documents: Sequence[Document]) -> np.ndarray:
        """Return a ``(len(documents),)`` array of relevance scores for ``query``.

        Higher is more relevant. Scores are comparable across calls of the same
        fitted reranker (they share its corpus statistics), so a caller can both
        re-order a candidate pool and threshold the scores for a precision
        estimate. Scoring an empty ``documents`` returns a length-0 array.
        """
        raise NotImplementedError
