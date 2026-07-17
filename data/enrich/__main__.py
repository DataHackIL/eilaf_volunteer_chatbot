"""Enrich every scraped section-tree: ``python -m data.enrich``.

Reads ``data/static/raw/*.json``, writes annotated copies to
``data/static/enriched/`` (raw is never touched), reusing the previous enriched
copy so re-runs only bill Claude for new/changed segments.

Cost control (Batches has no built-in spend cap):
  --dry-run     report how many segments would be sent to Claude; call nothing.
  --limit N     send at most N new segments to Claude this run; defer the rest
                (a later run picks them up), so you can enrich within a budget.
"""

from __future__ import annotations

import argparse
import json

from chatbot.rag.pipeline.pipeline import DEFAULT_STATIC_DIR, RAW_STATIC_DIR
from data.enrich.claude_enricher import ClaudeSegmentEnricher
from data.enrich.enrich_store import annotate_tree


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m data.enrich")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many segments would be sent to Claude; do not call it or write.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="send at most N new segments to Claude this run (defer the rest).",
    )
    args = parser.parse_args()

    if not args.dry_run:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

    raw_dir = RAW_STATIC_DIR
    enriched_dir = DEFAULT_STATIC_DIR
    raw_files = sorted(raw_dir.glob("*.json"))
    if not raw_files:
        print(f"No raw section-trees in {raw_dir}")
        return

    enriched_dir.mkdir(parents=True, exist_ok=True)
    enricher = ClaudeSegmentEnricher()
    remaining = args.limit
    total_pending = 0

    for raw_path in raw_files:
        raw_tree = json.loads(raw_path.read_text(encoding="utf-8"))
        out_path = enriched_dir / raw_path.name
        existing = (
            json.loads(out_path.read_text(encoding="utf-8"))
            if out_path.exists()
            else None
        )

        max_new = 0 if args.dry_run else remaining
        tree, n_pending, billed = annotate_tree(
            raw_tree, enricher, existing, max_new=max_new
        )
        total_pending += n_pending

        if args.dry_run:
            print(f"{raw_path.name}: {n_pending} segments would be sent to Claude")
            continue

        out_path.write_text(
            json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        deferred = n_pending - billed
        print(
            f"{raw_path.name}: enriched ({billed} sent to Claude"
            + (f", {deferred} deferred" if deferred else "")
            + ")"
        )
        if remaining is not None:
            remaining -= billed
            if remaining <= 0:
                print(f"--limit reached; {total_pending - billed}+ segments deferred.")
                break

    if args.dry_run:
        print(f"total: {total_pending} segments would be sent to Claude.")


if __name__ == "__main__":
    main()
