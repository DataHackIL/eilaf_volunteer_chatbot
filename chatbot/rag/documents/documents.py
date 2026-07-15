"""Load the static knowledge store into text chunks.

Files in ``data/static/`` are read and returned as :class:`Document` objects:
- ``.json`` → a scraper section-tree (nevo/kolzchut): one Document per
  text-bearing node, prefixed with its heading breadcrumb for context.
- ``.docx`` → one Document per non-empty paragraph (a "phrase").
- ``.pdf`` / ``.txt`` → text split into paragraph-packed chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF — better Hebrew/RTL extraction than pypdf
from docx import Document as DocxDocument

# Joins a node's ancestor headings into the breadcrumb prefixed to each chunk.
_CRUMB = " › "


@dataclass
class Document:
    """A single chunk of text plus where it came from."""

    text: str
    source: str = ""


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


def _flatten_json_tree(data: dict, source: str) -> list[Document]:
    """Flatten a scraper section-tree into heading-prefixed Documents.

    Handles both scraper schemas: nevo nodes carry a ``marker`` (e.g. "6."),
    kolzchut nodes a ``heading``; both nest via a recursive ``children`` list
    and hold body text in ``text``. Each text-bearing node becomes one Document
    whose text is prefixed with its ancestor breadcrumb, so an isolated chunk
    still says which right / section it belongs to.
    """
    title = data.get("title", "")
    documents: list[Document] = []

    def label(node: dict) -> str:
        parts = (node.get("marker", ""), node.get("heading", ""))
        return " ".join(part for part in parts if part).strip()

    def emit(text: str, crumbs: list[str]) -> None:
        text = (text or "").strip()
        if not text:
            return
        prefix = _CRUMB.join(crumb for crumb in crumbs if crumb)
        documents.append(
            Document(text=f"{prefix}\n{text}" if prefix else text, source=source)
        )

    emit(data.get("lead", ""), [title])  # kolzchut intro paragraph

    def walk(nodes: list, crumbs: list[str]) -> None:
        for node in nodes:
            child_crumbs = crumbs + [label(node)]
            emit(node.get("text", ""), child_crumbs)
            walk(node.get("children", []), child_crumbs)

    walk(data.get("children", []), [title])
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
