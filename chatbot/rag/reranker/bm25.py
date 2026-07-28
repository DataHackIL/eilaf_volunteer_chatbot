"""Okapi BM25 reranker — a self-contained lexical scorer over the corpus.

BM25 needs corpus-wide statistics (document frequencies, average length) to weigh
terms, so the reranker is *fitted* on the full corpus at construction and then
scores any subset against those fixed stats. Fitting on the corpus (not on the
handful of candidates handed to :meth:`score`) is what makes a rare term like a
statute number carry real weight — its idf comes from the whole collection.

Implemented directly (no ``rank_bm25`` dependency): the formula is ~20 lines and
we want control over Hebrew tokenization and over reusing the fitted stats for
the precision gauge. Tokenization is whitespace/punctuation splitting on Unicode
word characters — adequate for Hebrew (whitespace-segmented), though it does not
strip the attached prefixes (ו/ה/ב/כ/ל/מ/ש); good enough as a lexical signal,
and swappable if a proper Hebrew tokenizer lands later.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

from chatbot.rag.documents import Document
from chatbot.rag.reranker.base import Reranker

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercased Unicode word tokens; ``lower`` is a no-op on Hebrew but folds
    any interleaved Latin/ASCII (source names, digits) to one case."""
    return _TOKEN.findall(text.lower())


class BM25Reranker(Reranker):
    def __init__(
        self,
        corpus: Sequence[Document | str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Fit BM25 statistics on ``corpus`` (Documents or raw strings).

        ``k1`` controls term-frequency saturation (higher = tf keeps mattering);
        ``b`` controls length normalisation (1 = full, 0 = none). The defaults
        are the standard Okapi values.
        """
        self.k1 = k1
        self.b = b
        texts = [d.text if isinstance(d, Document) else d for d in corpus]
        self._doc_tf: list[Counter[str]] = [Counter(_tokenize(t)) for t in texts]
        self._doc_len = np.array([sum(tf.values()) for tf in self._doc_tf], dtype=np.float64)
        n_docs = len(self._doc_tf)
        self._avgdl = float(self._doc_len.mean()) if n_docs else 0.0
        # Document frequency per term, then the (always non-negative) BM25 idf.
        df: Counter[str] = Counter()
        for tf in self._doc_tf:
            df.update(tf.keys())
        self._idf: dict[str, float] = {
            term: np.log(1 + (n_docs - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def _bm25(self, terms: Sequence[str], tf: Counter[str], dl: float) -> float:
        if not dl or not self._avgdl:
            return 0.0
        denom_len = self.k1 * (1 - self.b + self.b * dl / self._avgdl)
        total = 0.0
        for term in terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = self._idf.get(term, 0.0)  # out-of-corpus query term: no weight
            total += idf * (f * (self.k1 + 1)) / (f + denom_len)
        return total

    def score(self, query: str, documents: Sequence[Document]) -> np.ndarray:
        """BM25 score of ``query`` against each of ``documents`` (any subset).

        Documents are re-tokenised here rather than looked up by identity, so the
        caller may pass freshly built or sliced Documents; scoring uses the fitted
        corpus idf/avgdl, so scores stay comparable to :meth:`score_corpus`.
        """
        terms = _tokenize(query)
        scores = []
        for doc in documents:
            tokens = _tokenize(doc.text)
            scores.append(self._bm25(terms, Counter(tokens), float(len(tokens))))
        return np.array(scores, dtype=np.float64)

    def score_corpus(self, query: str) -> np.ndarray:
        """BM25 score of ``query`` against every fitted corpus document, in order.

        Reuses the token counts computed at fit time (no re-tokenisation), so the
        precision gauge can cheaply find the best possible lexical match per
        query for normalisation. Aligned to the corpus passed to ``__init__``.
        """
        terms = _tokenize(query)
        return np.array(
            [self._bm25(terms, tf, dl) for tf, dl in zip(self._doc_tf, self._doc_len)],
            dtype=np.float64,
        )
