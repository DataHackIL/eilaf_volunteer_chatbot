"""Benchmark query translators: does an Arabic question find what its Hebrew twin finds?

The bot answers from a Hebrew corpus, and the same question asked in Arabic used
to retrieve different passages than in Hebrew. This script measures that gap and
how much each translator closes it, without needing relevance labels: for every
pair in the question set, the Hebrew question's top-k matches are the reference,
and each configuration is scored on how much of that reference its Arabic
retrieval recovers.

Configurations, one row each in the summary:

* ``none``: the Arabic question retrieved as is (the behaviour before
  translation existed). This is the gap to close.
* each translator, scored twice: ``translated`` retrieves on the Hebrew
  translation alone (what the bot does by default); ``fused`` merges it with
  the original-language ranking by reciprocal-rank fusion, the original
  weighted by ``--original-weight`` (default 1.0, i.e. equal footing).
  If ``fused`` ever beats ``translated``, raise ``DEFAULT_ORIGINAL_WEIGHT``
  in ``chatbot/rag/pipeline/pipeline.py``.

Columns:

* ``overlap@k``: mean Jaccard overlap between the configuration's top-k ids and
  the reference's (1.0 = exactly the same passages).
* ``top1@k``: share of questions where the reference's best passage is anywhere
  in the configuration's top-k.
* ``chrF``: character n-gram F-score (0-100) of the translation against the
  Hebrew question, which doubles as a reference translation. A rough
  translation-quality signal, independent of retrieval.
* ``sec``: mean seconds per translation on this machine. Run it on the target
  box too: the measured latency is only meaningful on the CPU that will serve.

Translator specs (``--translator``, repeatable; ``none`` is always included):

    nllb:/path/to/ct2-dir             NLLB on CTranslate2 (Palestinian Arabic)
    nllb:/path/to/ct2-dir@arb_Arab    ... treating input as Modern Standard Arabic
    opus                              Helsinki-NLP/opus-mt-ar-he
    opus:<hf model id>                another Marian model
    llm:<model>@<base url>            OpenAI-compatible endpoint, e.g.
                                      llm:gemma3:4b@http://localhost:11434/v1
                                      (key from EILAF_TRANSLATOR_API_KEY)
    glossary+<any spec above>         the same, with the Arabic dialect glossary
                                      applied first (what the bot runs), read
                                      from data/static/glossary_ar.csv; add
                                      @<csv> at the very end for another file:
                                      glossary+nllb:/path@ajp_Arab@my.csv

The glossary was built from the failures on the draft pairs, so its gain on
that same set is optimistic. Judge it on questions it hasn't seen.

Example::

    python scripts/translation_bench.py \\
        --translator nllb:$HOME/models/nllb-600m-ct2 \\
        --translator opus \\
        --details /tmp/bench-details.csv

The question set is ``data/eval/translation_pairs.csv`` (see the README there).
Pairs with an empty Arabic column are skipped.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from chatbot.rag.pipeline.pipeline import DEFAULT_STATIC_DIR, RAGPipeline  # noqa: E402
from chatbot.rag.translator import (  # noqa: E402
    Glossary,
    GlossaryTranslator,
    LLMTranslator,
    NLLBTranslator,
    OpusMTTranslator,
    Translator,
)

DEFAULT_PAIRS = REPO / "data" / "eval" / "translation_pairs.csv"
ARABIC = "Arabic"


@dataclass
class Pair:
    id: str
    hebrew: str
    arabic: str


def load_pairs(path: Path) -> list[Pair]:
    # utf-8-sig: the file carries a BOM so Excel shows the Arabic and Hebrew
    # properly, and reading it back must not glue the BOM onto the first header.
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = [
        Pair(r["id"].strip(), r["hebrew"].strip(), r["arabic_palestinian"].strip())
        for r in rows
        if r.get("hebrew", "").strip() and r.get("arabic_palestinian", "").strip()
    ]
    if not pairs:
        sys.exit(f"no complete pairs in {path}")
    return pairs


def build_translator(spec: str) -> Translator:
    if spec.startswith("glossary+"):
        inner_spec = spec.removeprefix("glossary+")
        if inner_spec.endswith(".csv"):
            inner_spec, _, path = inner_spec.rpartition("@")
            return GlossaryTranslator(
                build_translator(inner_spec), Glossary.from_csv(path)
            )
        return GlossaryTranslator(build_translator(inner_spec))
    kind, _, rest = spec.partition(":")
    if kind == "nllb":
        model_dir, _, code = rest.partition("@")
        if not model_dir:
            sys.exit("nllb needs a model directory: nllb:/path/to/ct2-dir")
        return NLLBTranslator(model_dir, **({"arabic_code": code} if code else {}))
    if kind == "opus":
        return OpusMTTranslator({"arabic": rest} if rest else None)
    if kind == "llm":
        model, _, url = rest.partition("@")
        if not (model and url):
            sys.exit("llm needs a model and a URL: llm:<model>@<base url>")
        return LLMTranslator(
            url, model, api_key=os.environ.get("EILAF_TRANSLATOR_API_KEY")
        )
    sys.exit(f"unknown translator spec {spec!r}")


class _Recording(Translator):
    """Wraps a translator to time each call and keep its output for the report."""

    def __init__(self, inner: Translator):
        self.inner = inner
        self.outputs: dict[str, str] = {}
        self.seconds: dict[str, float] = {}

    @property
    def name(self) -> str:
        return self.inner.name

    def to_hebrew(self, text: str, source: str) -> str:
        if text not in self.outputs:
            started = time.perf_counter()
            self.outputs[text] = self.inner.to_hebrew(text, source)
            self.seconds[text] = time.perf_counter() - started
        return self.outputs[text]


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, Popović 2015), whitespace removed, 0-100."""
    hyp, ref = hypothesis.replace(" ", ""), reference.replace(" ", "")
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        h = Counter(hyp[i : i + n] for i in range(len(hyp) - n + 1))
        r = Counter(ref[i : i + n] for i in range(len(ref) - n + 1))
        if not h or not r:
            continue
        hit = sum((h & r).values())
        precisions.append(hit / sum(h.values()))
        recalls.append(hit / sum(r.values()))
    if not precisions:
        return 0.0
    p, r = statistics.mean(precisions), statistics.mean(recalls)
    if p == 0 and r == 0:
        return 0.0
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r)


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


@dataclass
class Row:
    label: str
    overlaps: list[float] = field(default_factory=list)
    top1: list[bool] = field(default_factory=list)
    chrf: list[float] = field(default_factory=list)
    seconds: list[float] = field(default_factory=list)

    def add(self, ids: list[str], reference: list[str]) -> float:
        overlap = jaccard(ids, reference)
        self.overlaps.append(overlap)
        self.top1.append(bool(reference) and reference[0] in ids)
        return overlap


def _ids(docs) -> list[str]:
    return [d.id or d.text[:40] for d in docs]


def _fmt(values: list[float], spec: str) -> str:
    return format(statistics.mean(values), spec) if values else "-"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--translator", action="append", default=[], metavar="SPEC")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--original-weight",
        type=float,
        default=1.0,
        help="weight of the original-language pass in the fused rows",
    )
    parser.add_argument("--static-dir", type=Path, default=DEFAULT_STATIC_DIR)
    parser.add_argument(
        "--details", type=Path, help="write a per-question CSV (translations included)"
    )
    args = parser.parse_args()

    pairs = load_pairs(args.pairs)
    print(f"{len(pairs)} pairs from {args.pairs}", file=sys.stderr)
    # Retrieval only; the generator is never called. Loading reuses the
    # embedding cache, which must already be warm (python -m
    # chatbot.rag.pipeline.warm_cache), or this first pays a full corpus encode.
    pipeline = RAGPipeline.from_static_dir(
        args.static_dir, top_k=args.top_k, original_weight=args.original_weight
    )

    def matches(query: str, language: str) -> list[str]:
        return _ids(pipeline.matches(query, language=language))

    references = {p.id: matches(p.hebrew, "Hebrew") for p in pairs}
    details: list[dict] = []
    rows: list[Row] = []

    pipeline.translator = None
    baseline = Row("none (Arabic as is)")
    for p in pairs:
        overlap = baseline.add(matches(p.arabic, ARABIC), references[p.id])
        details.append({"id": p.id, "config": "none", "overlap": f"{overlap:.2f}"})
    rows.append(baseline)

    for spec in args.translator:
        print(f"loading {spec}", file=sys.stderr)
        recording = _Recording(build_translator(spec))
        translated = Row(f"{recording.name} translated")
        fused = Row(f"{recording.name} fused (w={args.original_weight:g})")
        pipeline.translator = recording
        for p in pairs:
            try:
                hebrew = recording.to_hebrew(p.arabic, ARABIC)
            except Exception as exc:  # noqa: BLE001 — report and keep benchmarking
                print(f"  {p.id}: translation failed: {exc}", file=sys.stderr)
                continue
            score = chrf(hebrew, p.hebrew)
            translated.chrf.append(score)
            translated.seconds.append(recording.seconds[p.arabic])
            t_overlap = translated.add(matches(hebrew, "Hebrew"), references[p.id])
            # Same translator through the pipeline: the recording wrapper's cache
            # means this reuses the translation above instead of redoing it.
            f_overlap = fused.add(matches(p.arabic, ARABIC), references[p.id])
            details.append(
                {
                    "id": p.id,
                    "config": recording.name,
                    "translation": hebrew,
                    "chrF": f"{score:.1f}",
                    "seconds": f"{recording.seconds[p.arabic]:.2f}",
                    "overlap_translated": f"{t_overlap:.2f}",
                    "overlap": f"{f_overlap:.2f}",
                }
            )
        fused.chrf, fused.seconds = translated.chrf, translated.seconds
        rows += [translated, fused]

    k = args.top_k
    print(f"\n| config | overlap@{k} | top1@{k} | chrF | sec |")
    print("|---|---|---|---|---|")
    for row in rows:
        top1 = _fmt([float(x) for x in row.top1], ".0%")
        print(
            f"| {row.label} | {_fmt(row.overlaps, '.2f')} | {top1} "
            f"| {_fmt(row.chrf, '.1f')} | {_fmt(row.seconds, '.2f')} |"
        )

    if args.details:
        by_id = {p.id: p for p in pairs}
        fields = [
            "id",
            "config",
            "hebrew",
            "arabic",
            "translation",
            "chrF",
            "seconds",
            "overlap_translated",
            "overlap",
        ]
        with args.details.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for d in details:
                pair = by_id[d["id"]]
                writer.writerow({"hebrew": pair.hebrew, "arabic": pair.arabic, **d})
        print(f"\nper-question details: {args.details}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
