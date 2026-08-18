"""Load the static knowledge store into text chunks.

Files in the enriched store (``data/static/enriched/``) are read and returned as
:class:`Document` objects:
- ``.json`` → a scraper section-tree (nevo/kolzchut): one Document per
  text-bearing node, prefixed with its heading breadcrumb for context.
- ``.docx`` → one Document per non-empty paragraph (a "phrase").
- ``.pdf`` / ``.txt`` → text split into paragraph-packed chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF — better Hebrew/RTL extraction than pypdf
from docx import Document as DocxDocument

# Joins a node's ancestor headings into the breadcrumb prefixed to each chunk.
_CRUMB = " › "


@dataclass
class Document:
    """A single chunk of text plus where it came from.

    ``meta`` holds closed-form eligibility tags used by the metadata filter
    (``type``/``area`` today; ``min_age``/``max_age``/``gender``/``locality``
    once the deferred extraction pass fills them). A missing key means NaN —
    unconstrained — and never causes the chunk to be filtered out.

    ``id`` / ``parent_id`` locate the chunk in its source section-tree, so
    retrieval can reattach a matched node's immediate parent (see
    :meth:`~chatbot.rag.pipeline.pipeline.RAGPipeline.retrieve`). ``id`` is a
    path-based id (``source#0/2/1`` = child 0 → child 2 → child 1); ``parent_id``
    is the immediate structural parent's id (``None`` at the top level, or for
    non-tree docx/pdf/txt chunks). A ``parent_id`` may reference a heading-only
    node that never became a Document — such a parent simply won't resolve.
    """

    text: str
    source: str = ""
    meta: dict = field(default_factory=dict)
    id: str = ""
    parent_id: str | None = None


def _read_pdf(path: Path) -> str:
    doc = fitz.open(path)
    try:
        return "\n\n".join(page.get_text() for page in doc)
    finally:
        doc.close()


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_docx_paragraphs(path: Path) -> list[str]:
    doc = DocxDocument(str(path))
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def _chunk(text: str, max_chars: int) -> list[str]:
    """Pack paragraphs (split on blank lines) into chunks of up to ``max_chars``."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for paragraph in paragraphs:
        if buf and len(buf) + len(paragraph) + 1 > max_chars:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n{paragraph}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def _root_meta(data: dict) -> dict:
    """Seed metadata from the tree's structured root fields.

    ``type``/``area`` come from the scrapers (kolzchut). ``locality`` is a
    dataset-level tag: national sources omit it (unconstrained); a municipal
    dataset sets it once on the root and it inherits down to every segment via
    :func:`_node_meta`, so the metadata filter can restrict it to that locality.

    ``track`` works the same way for sources devoted to one compensation track
    (hostile acts, road accidents…), where per-segment inference is unreliable —
    an isolated sentence about a monthly allowance names no track, but the page
    it came from does. The enricher may still override it per segment.
    """
    meta = {}
    for key in ("type", "area", "locality", "track"):
        value = (data.get(key) or "").strip()
        if value:
            meta[key] = value
    return meta


def _node_meta(node: dict, parent_meta: dict) -> dict:
    """Metadata for one node: the parent's tags, overridden by the node's own.

    The enrich stage (``data/enrich``) parses age/gender/locality out of the
    Hebrew prose and stores them under ``node["meta"]``; here they override the
    inherited parent tags. Un-enriched nodes carry no ``meta`` and simply inherit
    the parent's — so the loader still works on a raw (un-enriched) tree.
    """
    meta = dict(parent_meta)
    node_meta = node.get("meta")
    if isinstance(node_meta, dict):
        meta.update({key: value for key, value in node_meta.items() if value is not None})
    return meta


def _flatten_json_tree(data: dict, source: str, merge_chars: int = 0) -> list[Document]:
    """Flatten a scraper section-tree into heading-prefixed Documents.

    Handles both scraper schemas: nevo nodes carry a ``marker`` (e.g. "6."),
    kolzchut nodes a ``heading``; both nest via a recursive ``children`` list
    and hold body text in ``text``. Each emitted Document is prefixed with its
    ancestor breadcrumb, so an isolated chunk still says which right / section it
    belongs to. Metadata flows root → child (children inherit, then override) so
    a chunk carries its eligibility tags.

    ``merge_chars`` sets the size-driven rollup that coarsens over-granular trees
    (e.g. the health-basket schedules, which enumerate one clause per node): a
    subtree whose combined useful text fits within ``merge_chars`` collapses into
    a *single* Document — its descendants' text joined (each sub-node kept with
    its own marker so ``(1)…(2)…`` structure survives) under the subtree root's
    id/crumb — instead of one Document per clause. Larger subtrees recurse as
    before, so genuinely long sections stay split. ``merge_chars=0`` disables the
    rollup: every text-bearing node becomes its own Document (the finer-grained
    substrate the id/parent tests pin). A collapsed subtree carries only its root
    node's metadata, so keep ``merge_chars`` near the embedder's context window.
    """
    title = data.get("title", "")
    documents: list[Document] = []
    root_meta = _root_meta(data)

    def node_id(path: tuple[int, ...]) -> str:
        return f"{source}#{'/'.join(map(str, path))}"

    def label(node: dict) -> str:
        parts = (node.get("marker", ""), node.get("heading", ""))
        return " ".join(part for part in parts if part).strip()

    def subtree_text(node: dict, is_root: bool) -> list[str]:
        """Useful text in ``node``'s subtree, in reading order.

        Descendants keep their own marker/heading inline so enumerated structure
        survives the merge; the root's label is omitted here — it rides in the
        breadcrumb prefix that :func:`emit` prepends.
        """
        parts: list[str] = []
        if node.get("useful", True):
            body = (node.get("text") or "").strip()
            if body:
                tag = "" if is_root else label(node)
                parts.append(f"{tag} {body}".strip() if tag else body)
        for child in node.get("children", []):
            parts.extend(subtree_text(child, is_root=False))
        return parts

    def emit(
        text: str, crumbs: list[str], meta: dict, id: str, parent_id: str | None
    ) -> None:
        text = (text or "").strip()
        if not text:
            return
        prefix = _CRUMB.join(crumb for crumb in crumbs if crumb)
        documents.append(
            Document(
                text=f"{prefix}\n{text}" if prefix else text,
                source=source,
                meta=dict(meta),
                id=id,
                parent_id=parent_id,
            )
        )

    # kolzchut intro paragraph — top-level content, so no structural parent.
    emit(data.get("lead", ""), [title], root_meta, id=f"{source}#lead", parent_id=None)

    def walk(nodes: list, crumbs: list[str], meta: dict, parent_path: tuple[int, ...]) -> None:
        # A top-level node's parent is the title/root, which is not a Document —
        # so ``parent_id`` is None there and won't resolve.
        parent_id = node_id(parent_path) if parent_path else None
        # Buffer of adjacent leaf clauses awaiting bin-packing (see below). Only
        # siblings under *this* parent are ever packed together, so packing never
        # crosses a section boundary; ``pack_id`` is the first buffered leaf's id.
        pack: list[str] = []
        pack_id: str | None = None

        def flush() -> None:
            nonlocal pack_id
            if pack:
                emit("\n".join(pack), crumbs, meta, pack_id, parent_id)
                pack.clear()
                pack_id = None

        for index, node in enumerate(nodes):
            path = parent_path + (index,)
            child_crumbs = crumbs + [label(node)]
            node_meta = _node_meta(node, meta)
            merged = "\n".join(subtree_text(node, is_root=True))
            small = merge_chars and 0 < len(merged) <= merge_chars
            if small and not node.get("children"):
                # A small *leaf* clause: bin-pack it with adjacent leaf siblings
                # into one Document of up to ``merge_chars`` — this is what
                # actually collapses the schedules' long flat enumerations. Keep
                # the clause's own marker inline (its crumb is the shared parent).
                unit = "\n".join(subtree_text(node, is_root=False))
                if pack and sum(len(t) for t in pack) + len(unit) > merge_chars:
                    flush()
                if pack_id is None:
                    pack_id = node_id(path)
                pack.append(unit)
                continue
            flush()  # a non-packable node ends the current bin
            if small:
                # A small *subtree* (a whole sub-section that fits): roll it up
                # into one Document instead of one-per-descendant-clause.
                emit(merged, child_crumbs, node_meta, node_id(path), parent_id)
                continue  # subtree consumed — don't also recurse into it
            # Too big (or rollup off): emit this node's own text, recurse.
            # Skip segments the enrich stage judged useless (headers, stubs,
            # meaningless lines); a missing flag (un-enriched) keeps the node.
            if node.get("useful", True):
                emit(node.get("text", ""), child_crumbs, node_meta, node_id(path), parent_id)
            walk(node.get("children", []), child_crumbs, node_meta, path)
        flush()  # emit any leaves buffered at the end of the sibling list

    walk(data.get("children", []), [title], root_meta, ())
    return documents


def load_documents(
    static_dir: str | Path, max_chars: int = 500, merge_chars: int = 500
) -> list[Document]:
    """Read every ``.json`` / ``.docx`` / ``.pdf`` / ``.txt`` in ``static_dir``.

    ``max_chars`` packs prose (``.pdf`` / ``.txt``) into chunks; ``merge_chars``
    coarsens section-trees (``.json``), collapsing small subtrees into one
    Document — see :func:`_flatten_json_tree`. Set ``merge_chars=0`` to keep the
    finest granularity (one Document per node).
    """
    static_dir = Path(static_dir)
    documents: list[Document] = []
    paths = sorted(
        [
            *static_dir.glob("*.json"),
            *static_dir.glob("*.docx"),
            *static_dir.glob("*.pdf"),
            *static_dir.glob("*.txt"),
        ]
    )
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            documents.extend(_flatten_json_tree(data, source=path.name, merge_chars=merge_chars))
        elif suffix == ".docx":
            texts = _read_docx_paragraphs(path)
            documents.extend(
                Document(text=text, source=path.name, id=f"{path.name}#{i}")
                for i, text in enumerate(texts)
            )
        else:
            raw = _read_pdf(path) if suffix == ".pdf" else _read_txt(path)
            documents.extend(
                Document(text=text, source=path.name, id=f"{path.name}#{i}")
                for i, text in enumerate(_chunk(raw, max_chars))
            )
    return documents
