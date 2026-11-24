"""Convert a published PDF into the section-tree JSON the web scrapers emit.

The loader can read a ``.pdf`` directly, but only as flat paragraph chunks with
no headings, no citations and no enrichment (enrichment reads ``raw/*.json``).
Converting at scrape time gives a PDF the same breadcrumb-prefixed, cited,
enrichable segments as every other source.

PDF text extraction is the hard part, and Hebrew makes it harder. A PDF stores
positioned glyphs, not text, and the stock extraction modes all fail on the
sources here in different ways:

* a Word export splits words into separately-positioned runs ("נפגע י"), and
  ``get_text()`` reports each run as its own line;
* an InDesign layout reports a wrapped paragraph's visual lines in an order
  where numbers and hyphens land on the wrong side ("פי חוק … הדיווח-על");
* ``sort=True`` puts every line into left-to-right order, reversing the Hebrew.

So each block's lines are rebuilt here from the glyphs themselves: characters
are ordered right to left by position, spaces come from the gaps between them,
and runs of left-to-right script (digits, Latin, URLs) are put back in reading
order — a small, sufficient subset of the Unicode bidi algorithm for running
text. Layout is handled the same way, by geometry: blocks are read column by
column, right to left, and text repeated on most pages (running headers, a
navigation sidebar, page numbers) is dropped as chrome.
"""

import json
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter
from urllib.parse import unquote, urlsplit

import fitz  # PyMuPDF
import requests

from data.scraping.base import BaseScraper, fold_barren_headings

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# A gap between neighbouring glyphs wider than this fraction of the font size
# is a word space; narrower ones are kerning inside a word.
_SPACE_GAP = 0.2
# How far (pt) a stored space glyph's centre may stray outside the gap between
# two letters and still count as the space between them.
_SPACE_SLACK = 0.5
# Glyphs whose baselines differ by less than this fraction of the font size
# sit on one visual line.
_SAME_LINE = 0.5
# Consecutive blocks in one column closer than this many font sizes apart are
# the wrapped lines of one paragraph, joined with a space rather than a newline.
_PARAGRAPH_GAP = 0.8
# Text on at least this share of pages is page furniture, not content…
_CHROME_SHARE = 0.4
# …provided the document is long enough for repetition to mean anything.
_CHROME_MIN_PAGES = 3
# A block at least this much larger than the body font is a heading candidate
# (the fallback when the PDF has no bookmarks)…
_HEADING_SCALE = 1.15
# …unless it runs longer than a heading would.
_HEADING_MAX_CHARS = 120
# Share of words a block must share with a bookmark title to be its heading;
# bookmark titles are typed separately and differ in order and punctuation.
_TOC_MATCH = 0.6
# Bookmarks that name a page rather than a section ("עמוד 3").
_PAGE_BOOKMARK = re.compile(r"^\s*עמוד\s+\d+\s*$")
# A printed table of contents: its title, and an entry line ending in a page
# number ("זכויות נפגעי עבירה בהליך הפלילי, כללי 3").
_CONTENTS_TITLE = "תוכן העניינים"
_CONTENTS_ENTRY = re.compile(r"^(?P<title>.*\D)\s+(?P<page>\d{1,3})$")
# Blocks narrower than this share of the page define its columns; a wider one
# (a spread's title, just under half a landscape page) would bridge two.
_COLUMN_WIDTH = 0.45
_BULLETS = ("•", "▪", "◦", "")


def _direction(char: str) -> str:
    """``"R"`` (Hebrew/Arabic), ``"L"`` (Latin, digits) or ``"N"`` (neutral)."""
    bidi = unicodedata.bidirectional(char)
    if bidi in ("R", "AL"):
        return "R"
    if bidi in ("L", "EN", "AN"):
        return "L"
    return "N"


def _logical_line(chars: list[dict]) -> str:
    """One visual line's glyphs as reading-order text.

    ``chars`` are ``{"c", "x0", "x1", "size"}`` dicts in any order. They are
    laid out right to left by position, a space is put wherever the gap between
    two glyphs is wider than :data:`_SPACE_GAP` of the font size, and each run
    of left-to-right characters — with the neutrals between them, so "08:00-20:00"
    or a URL stays whole — is reversed back into reading order.

    The PDF's own space glyphs count only where they sit *between* two letters.
    Justified layouts (InDesign) need them, their word gaps being too tight to
    measure; Word exports position some inside a neighbouring letter's box,
    which is where the "נפגע י" splits come from. Invisible format characters
    (a byte-order mark, bidi marks) are dropped.
    """
    spaces = sorted((ch["x0"] + ch["x1"]) / 2 for ch in chars if ch["c"].isspace())
    glyphs = sorted(
        (ch for ch in chars if not ch["c"].isspace() and unicodedata.category(ch["c"]) != "Cf"),
        key=lambda ch: -ch["x1"],
    )
    text: list[str] = []
    for prev, cur in zip([None, *glyphs], glyphs):
        if prev is not None:
            gap_from, gap_to = cur["x1"] - _SPACE_SLACK, prev["x0"] + _SPACE_SLACK
            stored = bisect_left(spaces, gap_to) > bisect_right(spaces, gap_from)
            if stored or prev["x0"] - cur["x1"] > _SPACE_GAP * max(cur["size"], 1):
                text.append(" ")
        text.append(cur["c"])

    kinds = [_direction(c) for c in text]
    out: list[str] = []
    i = 0
    while i < len(text):
        if kinds[i] != "L":
            out.append(text[i])
            i += 1
            continue
        end = i
        j = i
        while j < len(text) and kinds[j] != "R":
            if kinds[j] == "L":
                end = j
            j += 1
        out.extend(reversed(text[i:end + 1]))
        i = end + 1
    return " ".join("".join(out).split())


def _block_text(block: dict) -> tuple[str, float]:
    """A PyMuPDF ``rawdict`` block's text, and its dominant font size."""
    chars = [
        {
            "c": ch["c"],
            "x0": ch["bbox"][0],
            "x1": ch["bbox"][2],
            "y": ch["origin"][1],
            "size": span["size"],
        }
        for line in block.get("lines", [])
        for span in line["spans"]
        for ch in span["chars"]
    ]
    if not chars:
        return "", 0.0
    sizes = Counter()
    for ch in chars:
        if not ch["c"].isspace():
            sizes[round(ch["size"], 1)] += 1
    size = sizes.most_common(1)[0][0] if sizes else chars[0]["size"]

    lines: list[list[dict]] = []
    for ch in sorted(chars, key=lambda ch: ch["y"]):
        if lines and abs(lines[-1][0]["y"] - ch["y"]) < _SAME_LINE * max(size, 1):
            lines[-1].append(ch)
        else:
            lines.append([ch])
    texts = [t for t in (_logical_line(line) for line in lines) if t]
    joined = ""
    for t in texts:
        sep = "\n" if t.startswith(_BULLETS) else " "
        joined = f"{joined}{sep}{t}" if joined else t
    return joined, size


def _norm(text: str) -> str:
    """Text reduced to its letters, for matching repeats and bookmark titles."""
    return "".join(ch for ch in text if ch.isalpha())


def _words(text: str) -> set[str]:
    return {w for w in (_norm(w) for w in text.split()) if w}


def _reading_order(blocks: list[dict], page_width: float) -> list[dict]:
    """``blocks`` in reading order: column by column, right to left.

    Columns are found from the narrow blocks (under :data:`_COLUMN_WIDTH` of the
    page) by merging ones whose horizontal extents overlap. A block straddling
    several columns — a page title over a two-column spread — cuts the page into
    bands and is read at its vertical position, ahead of the columns below it.
    """
    narrow = [b for b in blocks if b["x1"] - b["x0"] < _COLUMN_WIDTH * page_width]
    columns: list[list[float]] = []
    for b in sorted(narrow, key=lambda b: b["x0"]):
        if columns and b["x0"] < columns[-1][1]:
            columns[-1][1] = max(columns[-1][1], b["x1"])
        else:
            columns.append([b["x0"], b["x1"]])

    def column_of(b: dict) -> int | None:
        hits = [i for i, (x0, x1) in enumerate(columns) if b["x0"] < x1 and b["x1"] > x0]
        return hits[0] if len(hits) == 1 else None

    spanning = sorted((b for b in blocks if column_of(b) is None), key=lambda b: b["y0"])
    ordered: list[dict] = []
    top = float("-inf")
    for cut in [*spanning, None]:
        bottom = cut["y0"] if cut else float("inf")
        band = [b for b in blocks if column_of(b) is not None and top <= b["y0"] < bottom]
        # Right-most column first; top to bottom within it.
        band.sort(key=lambda b: (-columns[column_of(b)][1], b["y0"]))
        ordered.extend(band)
        if cut:
            ordered.append(cut)
            top = cut["y0"]
    return ordered


class PDFScraper(BaseScraper):
    """Download a PDF and convert it into a heading-nested section tree.

    Headings come from the PDF's bookmarks when it has them (each matched to
    the block on its page that shares most of its words), else from font size:
    a short block well above the body size opens a section, the larger sizes
    the outer levels. Output mirrors the other scrapers — a root carrying
    ``title``/``url``/``type``/``lead`` over ``{"heading", "level", "text",
    "children"}`` nodes — and each node cites the document at the page its
    section starts on, as "<title>, עמ' N" linking to ``<url>#page=N`` (which
    browsers' PDF viewers open at that page).

    ``title`` names the document in citations; PDFs rarely carry a usable one
    in their metadata, so it is best given explicitly.
    """

    def __init__(self, *args, title: str = "", track: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._extension = "json"
        self.title = title
        # As in gov_il.py: stamps the whole document with one compensation track.
        self.track = track

    def scrape(self) -> bytes:
        response = requests.get(self.target_url, headers=_HEADERS, timeout=120)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise RuntimeError(f"{self.target_url!r} did not return a PDF")
        return response.content

    def _blocks(self, doc) -> list[dict]:
        """Every text block of ``doc`` in reading order, page furniture removed."""
        pages = []
        for number, page in enumerate(doc, start=1):
            found = []
            for block in page.get_text("rawdict")["blocks"]:
                if block["type"] != 0:
                    continue
                text, size = _block_text(block)
                if not _norm(text):
                    continue  # page numbers, rules, stray punctuation
                x0, y0, x1, y1 = block["bbox"]
                found.append({"text": text, "size": size, "page": number,
                              "x0": x0, "y0": y0, "x1": x1, "y1": y1})
            pages.append(found)

        repeats = Counter()
        for found in pages:
            repeats.update({_norm(b["text"]) for b in found})
        chrome = (
            {text for text, count in repeats.items() if count >= _CHROME_SHARE * len(pages)}
            if len(pages) >= _CHROME_MIN_PAGES
            else set()
        )
        blocks = []
        for page, found in zip(doc, pages):
            kept = [b for b in found if _norm(b["text"]) not in chrome]
            blocks.extend(_reading_order(kept, page.rect.width))
        return blocks

    @staticmethod
    def _printed_contents(blocks: list[dict]) -> tuple[list[tuple[int, str, int]], set[int]]:
        """Entries of a printed table of contents, and the blocks that make it up.

        A Word export often has no bookmarks but does have a contents page:
        a "תוכן העניינים" block followed, on the same page, by lines that end
        in a page number. Entries come back as ``(1, title, page)``, the shape of
        a top-level bookmark.
        """
        start = next(
            (i for i, b in enumerate(blocks) if _norm(b["text"]) == _norm(_CONTENTS_TITLE)),
            None,
        )
        if start is None:
            return [], set()
        entries, used = [], {start}
        for i in range(start + 1, len(blocks)):
            if blocks[i]["page"] != blocks[start]["page"]:
                break
            match = _CONTENTS_ENTRY.match(blocks[i]["text"])
            if match:
                entries.append((1, match["title"].strip(" .…"), int(match["page"])))
                used.add(i)
        return entries, used

    @staticmethod
    def _match_entries(entries: list[tuple[int, str, int]], blocks: list[dict], skip: set[int]) -> dict[int, int]:
        """Block index -> heading level, for contents entries found in the text.

        Each entry is matched to the block on its page sharing most of its
        words; a printed contents page may be a page off from the PDF's own
        numbering, so the neighbouring pages are tried after it.
        """
        headings: dict[int, int] = {}
        for level, title, page in entries:
            want = _words(title)
            if not want:
                continue
            for candidate in (page, page + 1, page - 1):
                best, best_score = None, 0.0
                for i, b in enumerate(blocks):
                    if b["page"] != candidate or i in headings or i in skip:
                        continue
                    if len(b["text"]) > 2 * len(title) + 20:
                        continue
                    have = _words(b["text"])
                    score = len(want & have) / len(want | have) if have else 0.0
                    if score > best_score:
                        best, best_score = i, score
                if best is not None and best_score >= _TOC_MATCH:
                    headings[best] = level + 1  # a level-1 entry is a top section
                    break
        return headings

    def _headings(self, doc, blocks: list[dict]) -> tuple[dict[int, int], set[int]]:
        """Block index -> heading level, and the blocks to leave out of the text.

        Bookmarks win when the PDF has them. Otherwise a printed contents page
        gives the top-level sections, and font size the headings inside them —
        from the first section on, since a cover page's large type is not a
        heading. With neither, font size alone.

        Left out: the contents page itself, and repeats of a heading's text
        elsewhere (a section title running at the top of each of its pages).
        """
        bookmarks = [entry for entry in doc.get_toc() if not _PAGE_BOOKMARK.match(entry[1])]
        if bookmarks:
            headings, skip = self._match_entries(bookmarks, blocks, set()), set()
        else:
            entries, skip = self._printed_contents(blocks)
            headings = self._match_entries(entries, blocks, skip)
            by_size = self._size_headings(blocks)
            if headings:
                first = min(headings)
                for i, level in by_size.items():
                    if i > first and i not in headings and i not in skip:
                        headings[i] = min(level + 1, 6)
            else:
                headings = {i: level for i, level in by_size.items() if i not in skip}
        titles = {_norm(blocks[i]["text"]) for i in headings}
        skip |= {i for i, b in enumerate(blocks) if i not in headings and _norm(b["text"]) in titles}
        return headings, skip

    @staticmethod
    def _size_headings(blocks: list[dict]) -> dict[int, int]:
        """Block index -> heading level, from font size."""
        weights = Counter()
        for b in blocks:
            weights[b["size"]] += len(b["text"])
        if not weights:
            return {}
        body = weights.most_common(1)[0][0]
        candidates = {
            i: b["size"]
            for i, b in enumerate(blocks)
            if b["size"] >= _HEADING_SCALE * body and len(b["text"]) <= _HEADING_MAX_CHARS
        }
        sizes = sorted(set(candidates.values()), reverse=True)
        return {i: min(2 + sizes.index(size), 6) for i, size in candidates.items()}

    def parse(self, content: bytes) -> dict:
        doc = fitz.open(stream=content, filetype="pdf")
        try:
            blocks = self._blocks(doc)
            headings, skip = self._headings(doc, blocks)
            meta_title = (doc.metadata or {}).get("title", "").strip()
        finally:
            doc.close()

        stem = unquote(urlsplit(self.target_url).path.rsplit("/", 1)[-1])
        title = self.title or meta_title or stem
        root = {"title": title, "url": self.target_url, "type": "pdf", "lead": "", "children": []}
        if self.track:
            root["track"] = self.track

        def citation(page: int) -> list[dict]:
            return [{"text": f"{title}, עמ' {page}", "url": f"{self.target_url}#page={page}"}]

        root["source"] = [{"text": title, "url": self.target_url}]
        body = {"heading": "", "level": 2, "text": "", "children": [],
                "source": citation(blocks[0]["page"] if blocks else 1)}
        root["children"].append(body)
        # Open sections, deepest last; the body node sits below every heading
        # level so text before the first heading lands in it.
        stack = [(1, body)]
        previous = None
        for i, block in enumerate(blocks):
            if i in skip:
                continue
            if i in headings:
                level = headings[i]
                node = {"heading": block["text"], "level": level, "text": "",
                        "children": [], "source": citation(block["page"])}
                while stack[-1][0] >= level:
                    stack.pop()
                (stack[-1][1] if len(stack) > 1 else root)["children"].append(node)
                stack.append((level, node))
                previous = None
                continue
            target = stack[-1][1]
            same_paragraph = (
                previous is not None
                and previous["page"] == block["page"]
                and 0 <= block["y0"] - previous["y1"] < _PARAGRAPH_GAP * block["size"]
                and block["x0"] < previous["x1"] and block["x1"] > previous["x0"]
                and not block["text"].startswith(_BULLETS)
            )
            sep = " " if same_paragraph else "\n"
            target["text"] = f'{target["text"]}{sep}{block["text"]}' if target["text"] else block["text"]
            previous = block

        for child in root["children"]:
            fold_barren_headings(child)
        root["children"] = [c for c in root["children"] if c["text"] or c["children"]]
        return root

    def format(self, content: bytes) -> str:
        """Return the reconstructed section tree as indented JSON."""
        return json.dumps(self.parse(content), ensure_ascii=False, indent=2)


_BLOB = "https://www.gov.il/BlobFolder"

# gov.il-hosted PDFs from the sources sheet: (url, title as cited).
_PDFS = (
    (f"{_BLOB}/generalpage/prkfiles2/he/logo1.pdf",
     "פרקליטות המדינה, דפי מידע לנפגעי עבירה"),
    (f"{_BLOB}/dynamiccollectorresultitem/molsa-yated-mental-therapy/he/Yated_Documents_yated-mental-therapy.pdf",
     "תוכנית יתד, מעני טיפול נפשי לצעירים וצעירות"),
)

if __name__ == "__main__":
    for url, title in _PDFS:
        PDFScraper(url, title=title)()
