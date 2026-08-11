import json
from urllib.parse import unquote, urlsplit

import requests
from bs4 import BeautifulSoup, Tag

from data.scraping.base import BaseHTMLScraper

# KolZchut runs one wiki per language, each with its own script path: a
# /he/<page> article is served by /w/he/api.php (there is no shared /w/api.php).
# Deriving the endpoint from the URL's language segment keeps Arabic pages
# (/ar/<page>) working through the same class.
API_PATH_TEMPLATE = "{scheme}://{netloc}/w/{lang}/api.php"
DEFAULT_LANG = "he"

# Chrome / navigation / transient site-notice blocks that are not article
# content; removed wholesale before the section tree is built.
_STRIP_SELECTORS = ", ".join((
    ".mw-editsection",                                # per-heading [edit] links
    ".toc-box", ".noprint", ".rs_skip",              # table of contents
    ".kolsherut-links-section", ".help-wrapper-all",  # "help services" callouts
    ".article-see-also",                             # "ראו גם" link lists
    ".reference", ".mw-references-wrap",              # footnote markers + reflist
    "ol.references", "sup.reference",
    ".mw-empty-elt", "style",
))

# Heading tag -> nesting depth. KolZchut hierarchy lives in these levels, so
# (unlike nevo.py) no legal-marker inference is needed.
_HEADING_LEVELS = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def _norm(text: str) -> str:
    """Whitespace-normalised text, for comparing note boxes across the page."""
    return " ".join(text.split())


class KolZchutScraper(BaseHTMLScraper):
    """Scrape a KolZchut (MediaWiki) page into a nested section tree.

    KolZchut is a MediaWiki site, so rather than scrape the rendered page we
    query the ``action=parse`` API, which returns the section list plus the
    article HTML. Hierarchy comes straight from the ``<h2>/<h3>/<h4>`` levels;
    only KolZchut's own chrome (TOC, edit links, help-service and site-notice
    boxes) is stripped. Output mirrors ``nevo.py``: an indented-JSON tree of
    ``{"heading", "anchor", "level", "text", "children"}`` nodes under a root
    carrying ``title``/``url``/``type``/``area``/``lead``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extension = "json"

    def _page_title(self) -> str:
        """MediaWiki page title from the target URL's last path segment."""
        slug = self.target_url.rstrip("/").rsplit("/", 1)[-1]
        return unquote(slug)

    def _api_endpoint(self) -> str:
        """API endpoint of the language wiki that serves the target URL."""
        parts = urlsplit(self.target_url)
        segments = [seg for seg in parts.path.split("/") if seg]
        lang = segments[0] if len(segments) > 1 else DEFAULT_LANG
        return API_PATH_TEMPLATE.format(
            scheme=parts.scheme, netloc=parts.netloc, lang=lang
        )

    def scrape(self) -> str:
        """Fetch the parsed article from the MediaWiki API as raw JSON text."""
        headers = {"User-Agent": "EilafVolunteerBot/0.1 (RAG ingest)"}
        params = {
            "action": "parse",
            "page": self._page_title(),
            "prop": "text|sections|displaytitle|properties",
            "format": "json",
            "formatversion": "2",
        }
        response = requests.get(self._api_endpoint(), params=params, headers=headers)
        response.raise_for_status()
        return response.text

    def parse(self, content: str) -> dict:
        """Parse the API JSON into the nested section tree."""
        payload = json.loads(content)
        if "error" in payload:
            info = payload["error"].get("info", payload["error"])
            raise RuntimeError(f"KolZchut API error for {self.target_url!r}: {info}")
        data = payload["parse"]

        # formatversion=2 returns properties as a dict; guard older list form.
        props = data.get("properties") or {}
        if isinstance(props, list):
            props = {p["name"]: p["value"] for p in props}

        soup = BeautifulSoup(data["text"], "html.parser")
        body = soup.select_one(".mw-parser-output") or soup
        for junk in body.select(_STRIP_SELECTORS):
            junk.decompose()

        title = BeautifulSoup(
            data.get("displaytitle") or data.get("title", ""), "html.parser"
        ).get_text(strip=True)
        root = {
            "title": title,
            "url": self.target_url,
            "type": props.get("ArticleType", ""),
            "area": props.get("ArticleContentArea", ""),
            "lead": "",
            "children": [],
        }

        # The lead's real content is the .article-intro box; anything else before
        # the first heading (site notices, service links) is chrome, so skip it.
        intro = body.select_one(".article-intro")
        if intro:
            root["lead"] = intro.get_text(" ", strip=True)

        # MediaWiki output is flat — headings and content blocks are siblings —
        # so walk in document order and nest sections by heading level via a
        # stack. The root sits at sentinel level 1 so every h2 attaches under it.
        # Site-notice boxes (e.g. a transient emergency banner) sit in the lead
        # *and* get echoed inside a section; the lead copies are recorded here so
        # the body echoes can be dropped while genuine per-section notes stay.
        stack = [(1, root)]
        seen_heading = False
        lead_banners = set()
        for el in body.find_all(recursive=False):
            if not isinstance(el, Tag):
                continue
            level = _HEADING_LEVELS.get(el.name)
            if level:
                seen_heading = True
                headline = el.select_one(".mw-headline")
                node = {
                    "heading": (headline or el).get_text(" ", strip=True),
                    "anchor": headline.get("id", "") if headline else "",
                    "level": level,
                    "text": "",
                    "children": [],
                }
                while stack[-1][0] >= level:
                    stack.pop()
                stack[-1][1]["children"].append(node)
                stack.append((level, node))
                continue

            is_note = "wr-note" in (el.get("class") or [])
            if not seen_heading:
                if is_note:  # lead chrome: article-intro is captured separately
                    lead_banners.add(_norm(el.get_text(" ", strip=True)))
                continue

            text = el.get_text(" ", strip=True)
            if not text or (is_note and _norm(text) in lead_banners):
                continue
            node = stack[-1][1]
            node["text"] = f'{node["text"]}\n{text}'.strip()
        return root

    def format(self, content: str) -> str:
        """Return the reconstructed section tree as indented JSON."""
        return json.dumps(self.parse(content), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Example usage: a (זכות) rights page and a (חוק) law/portal page.
    KolZchutScraper("https://www.kolzchut.org.il/he/%D7%93%D7%9E%D7%99_%D7%90%D7%91%D7%98%D7%9C%D7%94")()
    KolZchutScraper("https://www.kolzchut.org.il/he/%D7%97%D7%95%D7%A7_%D7%94%D7%91%D7%99%D7%98%D7%95%D7%97_%D7%94%D7%9C%D7%90%D7%95%D7%9E%D7%99")()
