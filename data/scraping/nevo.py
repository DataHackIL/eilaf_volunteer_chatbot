import json
import re

from bs4 import BeautifulSoup, NavigableString, Tag

from data.scraping.base import BaseHTMLScraper

# Hebrew letters used as clause markers, e.g. "(א)".
_HEB = "אבגדהוזחטיכלמנסעפצקרשת"
# A section marker: digits with an optional Hebrew suffix and a dot, e.g. "6.",
# "9." or "3א." / "3 א." (the amended-insert variant).
_SECTION_RE = re.compile(r"^\d+\s*[" + _HEB + r"]{0,2}[.．]")
# A parenthesised sub-clause marker: "(א)", "(ב)", "(1)", "(א1)", "(א)(1)"…
_SUB_RE = re.compile(r"^(\((?:[" + _HEB + r"]{1,2}\d?|\d+[" + _HEB + r"]?)\))\s*")


def _kind(marker: str) -> str:
    """Classify a marker into a nesting 'kind'.

    Israeli legal numbering alternates between Hebrew-letter levels (סעיף קטן,
    פסקת משנה) and digit levels (פסקה), so the kind — not the DOM — drives depth.
    """
    if _SECTION_RE.match(marker):
        return "section"
    inner = marker.strip("()").strip()
    if inner and all(c in _HEB for c in inner):
        return "letter"
    if inner.isdigit():
        return "digit"
    return "other"  # definitions / unnumbered continuations


def _attach_subclause(marker, text, heading, anchor, path):
    """Attach a parenthesised sub-clause to the deepest same-kind level.

    Returns the (possibly truncated) path with the new node pushed on. The node
    becomes a sibling of the deepest level of its own kind, else nests one deeper.
    """
    kind = _kind(marker)
    depth = next((i for i in range(len(path) - 1, -1, -1)
                  if path[i][0] == kind), None)
    path = path if depth is None else path[:depth]
    parent = path[-1][1] if path else anchor
    node = {"marker": marker, "heading": heading, "text": text, "children": []}
    parent["children"].append(node)
    return path + [(kind, node)], node


class NevoScraper(BaseHTMLScraper):
    """Base for Nevo legislation scrapers.

    Nevo serves laws in two unrelated HTML templates; each subclass parses one
    into the same nested tree of chapters → sections → sub-clauses. Nodes are
    ``{"marker", "heading", "text", "children"}`` dicts; chapters use
    ``{"type": "chapter", "title", "children"}`` and legislative-history lines
    are collected on an optional ``"notes"`` list.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extension = "json"

    def parse(self, content: str) -> dict:
        """Parse raw HTML into the nested tree. Implemented per template."""
        raise NotImplementedError

    def format(self, content: str) -> str:
        """Return the reconstructed hierarchy as indented JSON."""
        return json.dumps(self.parse(content), ensure_ascii=False, indent=2)


def _marker_of(div: Tag) -> str:
    """Return the leading bare-text marker of a unit div (e.g. '6.', '(1)', '(א)').

    On the modern template each section/clause is a ``<div>`` whose marker sits
    as a bare text node before the ``<p>`` body; definitions have no marker.
    """
    for child in div.children:
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                return text
        elif isinstance(child, Tag):
            break  # reached the <p>/<defenition> before any marker text
    return ""


class NevoModernScraper(NevoScraper):
    """Scraper for the modern, semantic Nevo template (e.g. law_html/law00/*).

    The law is a *flat* list of ``<div>`` units whose marker is a bare text node
    (``6.`` → ``(א)`` → ``(1)`` …); hierarchy lives in that marker plus the
    ``<h3>`` chapter and ``<h6>`` marginal-heading tags, not in DOM nesting.
    """

    def parse(self, content: str) -> dict:
        soup = BeautifulSoup(content, "html.parser")
        title_el = soup.find("h1")
        content_el = title_el.parent
        root = {"title": title_el.get_text(strip=True), "children": []}

        chapter = None       # current <h3> node, or None before the first one
        heading = ""         # pending <h6> marginal heading for the next unit
        section = None       # current top-level section node
        anchor = root        # node a depth-1 sub-clause attaches under
        path = []            # stack of (kind, node) for the open sub-clause levels

        for el in content_el.children:
            if not isinstance(el, Tag):
                continue
            if el.name == "h3":
                chapter = {"type": "chapter", "title": el.get_text(strip=True),
                           "children": []}
                root["children"].append(chapter)
                section, anchor, path = None, chapter, []
            elif el.name == "h6":
                heading = el.get_text(strip=True)
            elif el.name == "div" and el.find("p"):
                marker = _marker_of(el)
                text = el.find("p").get_text(strip=True)
                kind = _kind(marker)

                if kind == "section":
                    # Nevo merges the section number with its first subsection,
                    # e.g. "9. (א)"; split it so the (א) level is explicit and
                    # later subsections nest as its siblings rather than deeper.
                    m = _SECTION_RE.match(marker)
                    section = {"type": "section", "marker": m.group().strip(),
                               "heading": heading, "text": "", "children": []}
                    (chapter or root)["children"].append(section)
                    anchor, path, heading = section, [], ""
                    rest = marker[m.end():].strip()
                    if _kind(rest) in ("letter", "digit"):
                        path, _ = _attach_subclause(rest, text, "", section, [])
                    else:
                        section["text"] = text
                elif kind == "other":
                    # definition / continuation: child of the current section,
                    # and a fresh anchor for any sub-clauses it introduces.
                    node = {"marker": marker, "heading": heading, "text": text,
                            "children": []}
                    (section or chapter or root)["children"].append(node)
                    anchor, path, heading = node, [], ""
                else:
                    path, _ = _attach_subclause(marker, text, heading, anchor, path)
                    heading = ""
        return root


# Legislative-history annotations interleaved in the legacy body: citation
# lines plus the change-description keywords that introduce/precede old text.
_NOTE_RE = re.compile(
    r'^(מיום|עד יום|תיקון מס|ס"ח|ה"ח|ק"ת|פ"ח|דיני מדינה|י"פ|הודעה'
    r"|הוספת|החלפת|מחיקת|ביטול|השמטת|החלת|הוראת שעה|תחילה|הנוסח הקודם)")


def _colors(p: Tag) -> set:
    """Set of CSS text colours used inside a paragraph's spans."""
    return {m.group(1)
            for s in p.find_all("span")
            for m in [re.search(r"color:\s*([a-z]+)", s.get("style", ""))]
            if m}


def _split_section(text: str):
    """Split a legacy section line into (number, first-subclause-or-None, body).

    e.g. '3. (א) כל תושב…' → ('3.', '(א)', 'כל תושב…').
    """
    m = _SECTION_RE.match(text)
    num, rest = m.group().strip(), text[m.end():]
    sub = _SUB_RE.match(rest)
    if sub:
        return num, sub.group(1), rest[sub.end():].strip()
    return num, None, rest.strip()


class NevoLegacyScraper(NevoScraper):
    """Scraper for the legacy Word-export Nevo template (e.g. law_html/law01/*).

    These consolidated laws have no semantic headings: markers are inline at the
    start of ``<p class="P00">`` text, chapters are ``medium2-header``
    paragraphs, section headings are coloured green, and amendment history
    (red ``מיום`` lines + citations) is interleaved. History is bucketed into
    each node's ``notes``; replacement *re-quotes* of old text may still leak.
    """

    def parse(self, content: str) -> dict:
        soup = BeautifulSoup(content, "html.parser")
        root = {"title": soup.title.get_text(strip=True), "children": []}

        ps = soup.find_all("p")
        body_start = next(i for i, p in enumerate(ps)
                          if p.get("class") and "medium2-header" in p["class"])

        chapter = None
        section = None
        anchor = root
        path = []
        heading = ""
        notes_target = root  # last real node, to hang history notes on

        for p in ps[body_start:]:
            cls = set(p.get("class") or [])
            if cls & {"footnote", "MsoFootnoteText"}:
                continue
            text = p.get_text(" ", strip=True)
            if not text:
                continue
            col = _colors(p)

            if "medium2-header" in cls:
                chapter = {"type": "chapter", "title": text, "children": []}
                root["children"].append(chapter)
                section, anchor, path, heading = None, chapter, [], ""
                notes_target = chapter
                continue

            sec_m = _SECTION_RE.match(text)
            # Green lines are section headings (like <h6> on the modern template).
            if "green" in col and not sec_m:
                heading = text
                continue
            # Citation / change-description lines are legislative history.
            if _NOTE_RE.match(text) or ("red" in col and text.startswith(("מיום", "עד יום"))):
                notes_target.setdefault("notes", []).append(text)
                continue

            if sec_m:
                num, sub, body = _split_section(text)
                section = {"type": "section", "marker": num, "heading": heading,
                           "text": "", "children": []}
                (chapter or root)["children"].append(section)
                anchor, path, heading = section, [], ""
                if sub:
                    path, notes_target = _attach_subclause(sub, body, "", section, [])
                else:
                    section["text"] = body
                    notes_target = section
            elif _SUB_RE.match(text):
                marker = _SUB_RE.match(text).group(1)
                body = text[_SUB_RE.match(text).end():].strip()
                path, notes_target = _attach_subclause(marker, body, "", anchor, path)
            else:
                # marker-less body line: a definition or a continuation.
                node = {"marker": "", "heading": "", "text": text, "children": []}
                (anchor or chapter or root)["children"].append(node)
                notes_target = node
        return root


if __name__ == "__main__":
    # Example usage: scrape a URL with the scraper matching its template.
    NevoModernScraper("https://www.nevo.co.il/law_html/law00/71835.htm")()
    NevoLegacyScraper("https://www.nevo.co.il/law_html/law01/036_001.htm")()
