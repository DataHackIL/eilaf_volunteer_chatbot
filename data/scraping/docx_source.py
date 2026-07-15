import json
from pathlib import Path

from docx import Document as DocxDocument

from data.scraping.base import BaseScraper


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
    """

    def __init__(self, path, *args, **kwargs):
        super().__init__(str(path), *args, **kwargs)
        self._path = Path(path)
        self._extension = "json"

    def scrape(self) -> str:
        """Read the docx's non-empty paragraphs as a JSON list.

        Mirrors the HTTP scrapers' ``scrape() -> raw text`` contract so
        ``parse()``/``format()`` stay symmetrical; the flag marks Word list
        items so paragraphs and bullet runs can be regrouped downstream.
        """
        doc = DocxDocument(str(self._path))
        paragraphs = [
            {"text": p.text.strip(), "list": p.style.name == "List Paragraph"}
            for p in doc.paragraphs
            if p.text.strip()
        ]
        return json.dumps(paragraphs, ensure_ascii=False)

    def parse(self, content: str) -> dict:
        """Flat section tree: one node per paragraph, bullet runs grouped."""
        paragraphs = json.loads(content)
        children: list[dict] = []
        bullets: list[str] = []

        def flush() -> None:
            if bullets:
                children.append(
                    {"heading": "", "text": "\n".join(bullets), "children": []}
                )
                bullets.clear()

        for paragraph in paragraphs:
            if paragraph["list"]:
                bullets.append(paragraph["text"])
            else:
                flush()
                children.append(
                    {"heading": "", "text": paragraph["text"], "children": []}
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
