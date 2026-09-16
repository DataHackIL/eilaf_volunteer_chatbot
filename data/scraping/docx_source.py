import json
from pathlib import Path

from docx import Document as DocxDocument
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.oxml import parse_xml
from docx.oxml.ns import qn

from data.scraping.base import BaseScraper

# Word's built-in separator lines live in footnotes.xml alongside the real notes.
_SEPARATOR_TYPES = ("separator", "continuationSeparator")


def _read_footnotes(doc) -> dict[str, dict]:
    """Footnote id -> ``{"text", "url"}`` for every real footnote in ``doc``.

    python-docx has no footnote API, but it keeps ``word/footnotes.xml`` as a
    raw part, so the notes are read from its XML. ``url`` is the note's first
    hyperlink target (``None`` if it has none); the drafts cite web pages, and a
    user can follow a URL where they could not open the docx.
    """
    try:
        part = doc.part.part_related_by(RT.FOOTNOTES)
    except KeyError:
        return {}
    notes = {}
    for note in parse_xml(part.blob).iterchildren(qn("w:footnote")):
        if note.get(qn("w:type")) in _SEPARATOR_TYPES:
            continue
        text = "".join(t.text or "" for t in note.iter(qn("w:t"))).strip()
        url = next(
            (
                part.rels[rid].target_ref
                for link in note.iter(qn("w:hyperlink"))
                if (rid := link.get(qn("r:id"))) in part.rels
            ),
            None,
        )
        if text or url:
            notes[note.get(qn("w:id"))] = {"text": text or url, "url": url}
    return notes


class DocxToJsonConverter(BaseScraper):
    """Convert a local ``.docx`` into the same section-tree JSON the web
    scrapers emit, so every source the RAG pipeline ingests is JSON.

    The example sources are hand-authored drafts with no reliable heading
    styles (only the odd ``Heading`` paragraph, most section titles left as
    ``Normal``), so inferring a real hierarchy would misclassify. This
    therefore produces a *flat* tree: the document title plus one node per
    paragraph, with consecutive Word list items grouped into a single node.
    A style-aware hierarchy can be layered on later if source docs adopt
    proper heading styles.

    Each node carries a ``source`` list: the footnotes anchored in its
    paragraph(s), as ``{"text", "url"}`` citations. The docx itself is never
    shown to users, so these are what an answer can point them to.
    """

    def __init__(self, path, *args, **kwargs):
        super().__init__(str(path), *args, **kwargs)
        self._path = Path(path)
        self._extension = "json"

    def scrape(self) -> str:
        """Read the docx's non-empty paragraphs as a JSON list.

        Mirrors the HTTP scrapers' ``scrape() -> raw text`` contract so
        ``parse()``/``format()`` stay symmetrical; the flag marks Word list
        items so paragraphs and bullet runs can be regrouped downstream, and
        ``source`` holds the citations of the footnotes the paragraph references.
        """
        doc = DocxDocument(str(self._path))
        footnotes = _read_footnotes(doc)
        paragraphs = []
        for p in doc.paragraphs:
            if not p.text.strip():
                continue
            ids = [ref.get(qn("w:id")) for ref in p._p.iter(qn("w:footnoteReference"))]
            paragraphs.append(
                {
                    "text": p.text.strip(),
                    "list": p.style.name == "List Paragraph",
                    "source": [footnotes[i] for i in ids if i in footnotes],
                }
            )
        return json.dumps(paragraphs, ensure_ascii=False)

    def parse(self, content: str) -> dict:
        """Flat section tree: one node per paragraph, bullet runs grouped."""
        paragraphs = json.loads(content)
        children: list[dict] = []
        bullets: list[str] = []
        bullet_sources: list[dict] = []

        def flush() -> None:
            if bullets:
                children.append(
                    {
                        "heading": "",
                        "text": "\n".join(bullets),
                        "source": list(bullet_sources),
                        "children": [],
                    }
                )
                bullets.clear()
                bullet_sources.clear()

        for paragraph in paragraphs:
            if paragraph["list"]:
                bullets.append(paragraph["text"])
                bullet_sources.extend(
                    s for s in paragraph["source"] if s not in bullet_sources
                )
            else:
                flush()
                children.append(
                    {
                        "heading": "",
                        "text": paragraph["text"],
                        "source": paragraph["source"],
                        "children": [],
                    }
                )
        flush()
        return {"title": self._path.stem, "type": "docx", "children": children}

    def format(self, content: str) -> str:
        return json.dumps(self.parse(content), ensure_ascii=False, indent=2)

    def __call__(self) -> None:
        """Convert and save as ``<docx stem>.json`` in the static directory."""
        self.save(self.format(self.scrape()), f"{self._path.stem}.{self._extension}")


if __name__ == "__main__":
    import glob

    for docx_path in glob.glob("/home/kneidell/Documents/eilaf/*.docx"):
        DocxToJsonConverter(docx_path)()
