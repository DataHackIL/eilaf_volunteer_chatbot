"""Entry point: `python -m chatbot.rag.eval` gauges precision over stdin queries.

Reads one query per line, builds the default static-dir pipeline (BM25 reranker
included) and prints each query's threshold precision plus the mean::

    printf 'שאלה אחת\\nשאלה שנייה\\n' | python -m chatbot.rag.eval --tau 0.5
"""

from __future__ import annotations

import argparse
import sys

from chatbot.rag.eval.precision import bm25_precision_at_k
from chatbot.rag.pipeline.pipeline import RAGPipeline


def _main() -> None:
    parser = argparse.ArgumentParser(description="BM25 threshold precision gauge.")
    parser.add_argument("--tau", type=float, default=0.5, help="relevance threshold in [0, 1]")
    args = parser.parse_args()

    queries = [line.strip() for line in sys.stdin if line.strip()]
    if not queries:
        print("No queries on stdin (one per line).", file=sys.stderr)
        raise SystemExit(1)

    pipeline = RAGPipeline.from_static_dir()
    if not pipeline.documents:
        print("No documents loaded; cannot gauge precision.", file=sys.stderr)
        raise SystemExit(1)

    report = bm25_precision_at_k(pipeline, queries, tau=args.tau)
    for query, precision in report.per_query:
        print(f"{precision:.2f}  {query}")
    print(f"\nmean P@{pipeline.top_k} (tau={report.tau}): {report.mean:.3f}")


if __name__ == "__main__":
    _main()
