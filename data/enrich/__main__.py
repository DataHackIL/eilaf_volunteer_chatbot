"""Enrich every scraped section-tree: ``python -m data.enrich``.

Reads ``data/static/raw/*.json``, writes annotated copies to
``data/static/enriched/`` (raw is never touched), reusing the previous enriched
copy so re-runs only bill the LLM for new/changed segments.

Modes / cost control (neither back-end has a built-in spend cap):
  --provider P  which LLM back-end to enrich with: ``claude`` (default) or
                ``gemini`` (Google AI free tier).
  --dry-run     report how many segments would be sent to the LLM; call nothing.
  --no-llm      rules-only pass: drop obvious junk, leave the rest untagged, and
                write every file — no API key needed. The deferred segments get
                tagged on a later full run.
  --limit N     send at most N new segments to the LLM this run; defer the rest
                (a later run picks them up), so you can enrich within a budget.
"""

from __future__ import annotations

import argparse
import json

from chatbot.rag.pipeline.pipeline import DEFAULT_STATIC_DIR, RAW_STATIC_DIR
from data.enrich.claude_enricher import ClaudeSegmentEnricher
from data.enrich.enrich_store import annotate_tree
from data.enrich.gemini_enricher import GeminiSegmentEnricher

# Maps --provider to the enricher class; add a back-end here in one place.
_ENRICHERS = {
    "claude": ClaudeSegmentEnricher,
    "gemini": GeminiSegmentEnricher,
}


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m data.enrich")
    parser.add_argument(
        "--provider",
        choices=sorted(_ENRICHERS),
        default="claude",
        help="which LLM back-end to enrich with (default: claude).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many segments would be sent to the LLM; do not call it or write.",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="rules-only pass: drop obvious junk, leave the rest untagged, write all files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="send at most N new segments to the LLM this run (defer the rest).",
    )
    args = parser.parse_args()

    # A non-positive limit means "send nothing to the LLM" — same as --no-llm.
    no_llm = args.no_llm or (args.limit is not None and args.limit <= 0)
    needs_llm = not (args.dry_run or no_llm)

    if needs_llm:
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
    enricher = _ENRICHERS[args.provider]()
    # Only a positive limit drives the stop-early budget; dry-run/no-llm send 0.
    remaining = None if not needs_llm else args.limit
    total_pending = 0

    for raw_path in raw_files:
        raw_tree = json.loads(raw_path.read_text(encoding="utf-8"))
        out_path = enriched_dir / raw_path.name
        existing = (
            json.loads(out_path.read_text(encoding="utf-8"))
            if out_path.exists()
            else None
        )

        max_new = remaining if needs_llm else 0
        tree, n_pending, billed = annotate_tree(
            raw_tree, enricher, existing, max_new=max_new
        )
        total_pending += n_pending

        if args.dry_run:
            print(f"{raw_path.name}: {n_pending} segments would be sent to the LLM")
            continue

        out_path.write_text(
            json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        deferred = n_pending - billed
        if no_llm:
            print(
                f"{raw_path.name}: rules-only "
                f"({deferred} segments deferred to a future LLM run)"
            )
        else:
            print(
                f"{raw_path.name}: enriched ({billed} sent to {args.provider}"
                + (f", {deferred} deferred" if deferred else "")
                + ")"
            )
        if remaining is not None:
            remaining -= billed
            if remaining <= 0:
                print(f"--limit reached; {total_pending - billed}+ segments deferred.")
                break

    if args.dry_run:
        print(f"total: {total_pending} segments would be sent to the LLM.")


if __name__ == "__main__":
    main()
