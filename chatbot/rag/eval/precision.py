"""Threshold-based retrieval precision, gauged with the BM25 reranker.

We have no relevance labels yet, so precision is estimated *lexically*: a matched
chunk counts as relevant when its BM25 score for the query clears a threshold,
and precision@k is the fraction of the pipeline's top-k matches that clear it.

Because raw BM25 scores are unbounded and query-dependent (a query with rare
terms scores higher everywhere), each match is normalised by the *best possible*
BM25 score for that query — the top score over the whole corpus. The threshold
``tau`` then reads as a fraction of that best-possible match ("at least half as
lexically on-topic as the single best chunk in the corpus"), so one ``tau`` is
comparable across queries.

Caveats worth remembering: this measures lexical overlap, not truth — it rewards
term-matching and can't see a correct answer phrased in other words, and using
the reranker's own signal to grade a reranker-ordered pipeline is a consistency
check, not an unbiased accuracy. It's a cheap, label-free dial to compare
retrieval configurations, not a substitute for a labelled eval set.

CLI (see :mod:`chatbot.rag.eval.__main__`)::

    printf 'שאלה אחת\\nשאלה שנייה\\n' | python -m chatbot.rag.eval --tau 0.5
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from chatbot.rag.pipeline.pipeline import RAGPipeline
from chatbot.rag.reranker import BM25Reranker


@dataclass
class PrecisionReport:
    """Per-query and mean threshold precision for one run."""

    tau: float
    per_query: list[tuple[str, float]]

    @property
    def mean(self) -> float:
        vals = [p for _, p in self.per_query]
        return float(np.mean(vals)) if vals else 0.0


def bm25_precision_at_k(
    pipeline: RAGPipeline,
    queries: Sequence[str],
    tau: float = 0.5,
    reranker: BM25Reranker | None = None,
    **retrieve_kwargs,
) -> PrecisionReport:
    """Estimate precision@k of ``pipeline`` over ``queries`` via BM25 threshold.

    For each query, the pipeline's top-k *matches* (pre-expansion; see
    :meth:`RAGPipeline.matches`) are scored with BM25 and normalised by the best
    corpus score for that query; precision is the share scoring above ``tau``.
    ``reranker`` defaults to the pipeline's own; pass one explicitly to gauge a
    reranker-less pipeline. Extra keyword args (``context``, ``context_weight``,
    ``facts``) pass through to ``matches`` so a run mirrors live retrieval.
    """
    scorer = reranker or pipeline.reranker
    if not isinstance(scorer, BM25Reranker):
        raise ValueError(
            "bm25_precision_at_k needs a BM25Reranker (pipeline.reranker is "
            f"{type(scorer).__name__}); pass reranker=BM25Reranker(pipeline.documents)."
        )
    per_query: list[tuple[str, float]] = []
    for query in queries:
        matches = pipeline.matches(query, **retrieve_kwargs)
        best = float(scorer.score_corpus(query).max()) if matches else 0.0
        if not matches or best <= 0:
            # No matches, or a query with no in-corpus terms: nothing to clear
            # the bar, so precision is 0 by definition.
            per_query.append((query, 0.0))
            continue
        normalised = scorer.score(query, matches) / best
        per_query.append((query, float(np.mean(normalised > tau))))
    return PrecisionReport(tau=tau, per_query=per_query)
