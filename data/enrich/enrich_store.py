"""Drive enrichment over a scraped section-tree, incrementally.

``annotate_tree`` takes a raw tree (and the previously enriched copy, if any) and
returns an annotated deep copy — the raw tree is never mutated. Each text-bearing
node gets a ``useful`` flag and, when the segment is constrained, a ``meta`` dict
of group tags. Segments already annotated in the previous copy are reused, and
obvious junk is rejected by rule, so only genuinely new segments reach Claude.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator

from data.enrich.base import SegmentEnricher
from data.enrich.rules import is_obvious_junk


def _text_nodes(tree: dict) -> Iterator[dict]:
    """Yield the tree's text-bearing nodes depth-first (stable order)."""

    def walk(nodes: list) -> Iterator[dict]:
        for node in nodes:
            if (node.get("text") or "").strip():
                yield node
            yield from walk(node.get("children", []))

    yield from walk(tree.get("children", []))


def _existing_annotations(
    existing_tree: dict | None,
) -> dict[str, tuple[bool, dict | None]]:
    """Map segment text -> (useful, meta) from a previously enriched tree."""
    cache: dict[str, tuple[bool, dict | None]] = {}
    if not existing_tree:
        return cache
    for node in _text_nodes(existing_tree):
        if "useful" in node:
            cache[node["text"].strip()] = (node.get("useful", True), node.get("meta"))
    return cache


def _apply(node: dict, useful: bool, meta: dict | None) -> None:
    node["useful"] = bool(useful)
    if meta:
        node["meta"] = meta
    else:
        node.pop("meta", None)


def annotate_tree(
    raw_tree: dict,
    enricher: SegmentEnricher,
    existing_tree: dict | None = None,
    max_new: int | None = None,
    chunk_size: int | None = None,
    on_chunk: Callable[[dict], None] | None = None,
) -> tuple[dict, int, int, int]:
    """Return ``(enriched_copy, n_pending, n_sent, n_annotated)``.

    Reuses annotations from ``existing_tree`` and rejects obvious junk by rule;
    only the remainder (``n_pending``) needs Claude. ``max_new`` caps how many of
    those are actually sent this run (``n_sent`` ≤ ``max_new``); the rest are
    left un-annotated so a later run picks them up — this is the budget lever.

    ``n_sent`` counts attempts (what costs money and what the budget spends);
    ``n_annotated`` counts the answers actually recorded. They differ when the
    back-end fails on a segment: that segment is left un-annotated and stays
    pending, so a later run retries it.
    ``max_new=0`` sends nothing (``enricher`` is never called), which powers the
    dry-run preflight.

    ``chunk_size`` sends the segments to the enricher ``chunk_size`` at a time
    and calls ``on_chunk(tree)`` after each chunk is applied, so the caller can
    persist progress: a 7k-segment file is otherwise all-or-nothing, and a
    crash (or an aborting back-end) late in the run loses every answer already
    paid for. Each segment is still sent at most once per call.
    """
    tree = copy.deepcopy(raw_tree)
    cache = _existing_annotations(existing_tree)

    pending_nodes: list[dict] = []
    pending_texts: list[str] = []
    for node in _text_nodes(tree):
        text = node["text"].strip()
        if text in cache:
            useful, meta = cache[text]
            _apply(node, useful, meta)
        elif is_obvious_junk(text):
            _apply(node, False, None)
        else:
            pending_nodes.append(node)
            pending_texts.append(text)

    send_nodes = pending_nodes if max_new is None else pending_nodes[:max_new]
    send_texts = pending_texts if max_new is None else pending_texts[:max_new]
    annotated = 0
    step = chunk_size if chunk_size and chunk_size > 0 else max(len(send_texts), 1)
    for start in range(0, len(send_texts), step):
        chunk_nodes = send_nodes[start : start + step]
        annotations = enricher.enrich(send_texts[start : start + step])
        # strict: one annotation per sent segment is the enricher contract;
        # a mismatch would silently skip nodes, so fail loudly instead.
        for node, annotation in zip(chunk_nodes, annotations, strict=True):
            # A segment the back-end never answered for (``ok=False``) is left
            # untouched, so it stays pending and a later run retries it. Writing
            # the enricher's defaults would be indistinguishable from a real
            # "useful, unconstrained" verdict and would never be revisited.
            if not annotation.ok:
                continue
            _apply(node, annotation.useful, annotation.tags() or None)
            annotated += 1
        if on_chunk is not None:
            on_chunk(tree)

    return tree, len(pending_texts), len(send_texts), annotated
