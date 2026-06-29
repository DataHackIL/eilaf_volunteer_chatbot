"""Load the static knowledge store into text chunks.

Files in ``data/static/`` are read and returned as :class:`Document` objects:
- ``.docx`` → one Document per non-empty paragraph (a "phrase").
- ``.pdf`` / ``.txt`` → text split into paragraph-packed chunks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF — better Hebrew/RTL extraction than pypdf
from docx import Document as DocxDocument


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


def load_documents(static_dir: str | Path, max_chars: int = 500) -> list[Document]:
    """Read every ``.docx`` / ``.pdf`` / ``.txt`` in ``static_dir`` as Documents."""
    static_dir = Path(static_dir)
    documents: list[Document] = []
    paths = sorted(
        [
            *static_dir.glob("*.docx"),
            *static_dir.glob("*.pdf"),
            *static_dir.glob("*.txt"),
        ]
    )
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".docx":
            texts = _read_docx_paragraphs(path)
        else:
            raw = _read_pdf(path) if suffix == ".pdf" else _read_txt(path)
            texts = _chunk(raw, max_chars=max_chars)
        for text in texts:
            documents.append(Document(text=text, source=path.name))
    return documents
