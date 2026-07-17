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
    """

    text: str
    source: str = ""
    meta: dict = field(default_factory=dict)


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
    """
    meta = {}
    for key in ("type", "area", "locality"):
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


def _flatten_json_tree(data: dict, source: str) -> list[Document]:
    """Flatten a scraper section-tree into heading-prefixed Documents.

    Handles both scraper schemas: nevo nodes carry a ``marker`` (e.g. "6."),
    kolzchut nodes a ``heading``; both nest via a recursive ``children`` list
    and hold body text in ``text``. Each text-bearing node becomes one Document
    whose text is prefixed with its ancestor breadcrumb, so an isolated chunk
    still says which right / section it belongs to. Metadata flows root → child
    (children inherit, then override) so a chunk carries its eligibility tags.
    """
    title = data.get("title", "")
    documents: list[Document] = []
    root_meta = _root_meta(data)

    def label(node: dict) -> str:
        parts = (node.get("marker", ""), node.get("heading", ""))
        return " ".join(part for part in parts if part).strip()

    def emit(text: str, crumbs: list[str], meta: dict) -> None:
        text = (text or "").strip()
        if not text:
            return
        prefix = _CRUMB.join(crumb for crumb in crumbs if crumb)
        documents.append(
            Document(
                text=f"{prefix}\n{text}" if prefix else text,
                source=source,
                meta=dict(meta),
            )
        )

    emit(data.get("lead", ""), [title], root_meta)  # kolzchut intro paragraph

    def walk(nodes: list, crumbs: list[str], meta: dict) -> None:
        for node in nodes:
            child_crumbs = crumbs + [label(node)]
            node_meta = _node_meta(node, meta)
            # Skip segments the enrich stage judged useless (headers, stubs,
            # meaningless lines); a missing flag (un-enriched) keeps the node.
            if node.get("useful", True):
                emit(node.get("text", ""), child_crumbs, node_meta)
            walk(node.get("children", []), child_crumbs, node_meta)

    walk(data.get("children", []), [title], root_meta)
    return documents


def load_documents(static_dir: str | Path, max_chars: int = 500) -> list[Document]:
    """Read every ``.json`` / ``.docx`` / ``.pdf`` / ``.txt`` in ``static_dir``."""
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
            documents.extend(_flatten_json_tree(data, source=path.name))
        elif suffix == ".docx":
            texts = _read_docx_paragraphs(path)
            documents.extend(Document(text=text, source=path.name) for text in texts)
        else:
            raw = _read_pdf(path) if suffix == ".pdf" else _read_txt(path)
            documents.extend(
                Document(text=text, source=path.name) for text in _chunk(raw, max_chars)
            )
    return documents
