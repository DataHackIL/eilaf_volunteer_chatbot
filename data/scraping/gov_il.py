"""Scrape gov.il content pages via the API the site's own front end calls.

``www.gov.il`` is a single-page app behind Cloudflare bot management: the HTML
body is an empty ``<div id="root">``, and the request is rejected on its TLS
fingerprint anyway (a browser ``User-Agent`` does not help; there is no
``robots.txt`` or ``sitemap.xml`` — the SPA catch-all serves the homepage for
both). So instead of the page we call the content API the page shell itself
loads, published in ``/ContentpageWebApi/client-config.js`` together with the
public client id below. That API lives on a different host which is *not* behind
Cloudflare, so plain ``requests`` reaches it.

Two endpoints are used:

* content   — one page (optionally chapter-by-chapter), see :class:`GovILScraper`
* discovery — which pages exist under a topic/office, see :func:`list_pages`

Both need ``x-client-id`` *and* an ``Origin`` header; without ``Origin`` the
gateway answers ``500 {"faultstring": "RF-OriginError"}``, which reads like a
server fault rather than a rejected request.
"""

import json
import time
from urllib.parse import parse_qs, urlsplit

import requests
from bs4 import BeautifulSoup, Tag

from data.scraping.base import BaseHTMLScraper

CONTENT_API = (
    "https://openapi-gc.digital.gov.il/pub/cio/govil/rest/contentpage/v1"
    "/api/content-pages/{slug}"
)
LANDING_API = (
    "https://openapi-gc.digital.gov.il/pub/cio/govil/rest/landingpage/v1"
    "/{culture}/api/{kind}/get/{slug}"
)
SITE = "https://www.gov.il"
# Public constant baked into the gov.il front end's client-config.js — it
# identifies the site, not a user. Still gov.il's, so keep the request rate low.
CLIENT_ID = "9KFgciHHGDyNiqz5MdQS0eK2ApeJYMc6YnElUICpN1atirZc"
HEADERS = {"x-client-id": CLIENT_ID, "Origin": SITE, "Referer": f"{SITE}/"}

DEFAULT_CULTURE = "he"
# Landing-page kinds addressed by their own path segment; anything else after
# /departments/ is an office slug (e.g. /he/departments/ministry_of_justice...).
LANDING_KINDS = ("topics", "units")
# The tab listing guides and information pages. Sibling tabs ("publication",
# "legalInfo", "policies") are PDFs and notices rather than page content.
PAGES_TAB = "generalPages"

# Politeness pause between consecutive API calls.
_PAUSE = 0.3

# Heading tag -> nesting depth, as in kol_zchut.py: gov.il's body fragments are
# flat HTML whose hierarchy lives entirely in these levels.
_HEADING_LEVELS = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
# Sentinel depth for a chapter node, below every heading level, so an <h2>
# opening a chapter's body nests *under* it instead of replacing it.
_CHAPTER_DEPTH = 1
# Some editors mark a section with a wholly-bold paragraph rather than an <hN>
# tag — the page renders the two identically. Recognising it matters: a page
# written that way would otherwise come out as one unsplit wall of text. Such
# headings sit one level below every real one, so consecutive ones stay
# siblings while still nesting under whatever real heading is open.
_PSEUDO_LEVEL = max(_HEADING_LEVELS.values()) + 1
# Above this length a bold paragraph is emphasised prose, not a heading.
_PSEUDO_MAX_CHARS = 120

_FAQ_HEADING = "שאלות ותשובות"


def _get(url: str, **params) -> dict | list:
    """GET a gov.il API endpoint and return the decoded payload.

    The gateway reports a missing page as HTTP 200 carrying an ``AlertCode``
    body (the front end redirects to /error on it), so that is raised here.
    """
    response = requests.get(url, params=params or None, headers=HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("AlertCode"):
        raise RuntimeError(
            f"gov.il API error for {url!r}: "
            f"{payload.get('AlertCode')} {payload.get('Message', '')}".strip()
        )
    time.sleep(_PAUSE)
    return payload


def _text(el: Tag) -> str:
    """Readable text of one body element.

    List items are newline-separated so enumerations survive as separate lines;
    everything else collapses to a single spaced line. ``&nbsp;`` (U+00A0) is
    normalised to a plain space — it is pervasive in gov.il's editor output and
    would otherwise reach the embedder as an odd character.
    """
    if el.name in ("ul", "ol"):
        parts = [li.get_text(" ", strip=True) for li in el.find_all("li")]
    else:
        parts = [el.get_text(" ", strip=True)]
    return "\n".join(p.replace("\xa0", " ").strip() for p in parts if p.strip())


def _heading_of(el: Tag) -> tuple[str, int] | None:
    """``(heading, level)`` if ``el`` opens a section, else ``None``.

    Real ``<hN>`` tags first, then the wholly-bold-paragraph convention (see
    :data:`_PSEUDO_LEVEL`).
    """
    text = el.get_text(" ", strip=True).replace("\xa0", " ")
    level = _HEADING_LEVELS.get(el.name)
    if level:
        return (text, level) if text else None
    if el.name != "p" or not text or len(text) > _PSEUDO_MAX_CHARS:
        return None
    bold = " ".join(b.get_text(" ", strip=True) for b in el.find_all(("strong", "b")))
    bold = " ".join(bold.replace("\xa0", " ").split())
    return (text, _PSEUDO_LEVEL) if bold == " ".join(text.split()) else None


def _fold_barren_headings(node: dict) -> None:
    """Turn headings that introduce nothing back into their parent's text.

    gov.il editors sometimes mark a plain sentence as an ``<h4>`` for emphasis.
    Left as a heading it would carry its own prose *as a heading* over an empty
    body, and the loader — which only emits text-bearing nodes — would drop it
    entirely. Folding bottom-up, so a heading whose children all fold is
    reconsidered once it has absorbed them.
    """
    kept = []
    for child in node.get("children", []):
        _fold_barren_headings(child)
        if child["text"] or child["children"]:
            kept.append(child)
        elif child["heading"]:
            node["text"] = f'{node["text"]}\n{child["heading"]}'.strip()
    node["children"] = kept


def _first_title(section: dict | None, key: str) -> str:
    """Title of the first entry under ``section[key]`` (a gov.il metadata list)."""
    entries = (section or {}).get(key) or []
    return entries[0].get("title", "").strip() if entries else ""


class GovILScraper(BaseHTMLScraper):
    """Scrape one gov.il content page into a nested section tree.

    The API returns the article as HTML fragments that keep their own
    ``<h2>/<h3>`` headings, so — as in ``kol_zchut.py`` — hierarchy comes
    straight from those levels. Multi-chapter guides are fetched chapter by
    chapter (``chapterIndex``) and merged into one tree, each chapter a
    top-level node named after its entry in the page's chapter nav.

    Output mirrors the other scrapers: an indented-JSON tree of
    ``{"heading", "level", "text", "children"}`` nodes under a root carrying
    ``title``/``url``/``type``/``area``/``lead``.
    """

    def __init__(self, *args, track: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self._extension = "json"
        # Set when the whole page belongs to one compensation track (see
        # data.enrich.base.Track). It lands on the tree root and the loader
        # inherits it down to every segment, which per-segment enrichment cannot
        # do reliably: an isolated sentence about an allowance names no track.
        self.track = track

    def _culture_and_slug(self) -> tuple[str, str]:
        """Culture and page slug from the target URL's path.

        The culture is the first segment (``/he/...``, ``/ar/...``) and the slug
        the last, which covers every page form gov.il uses: ``/he/pages/{slug}``,
        ``/he/departments/guides/{slug}``, ``/he/service/{slug}``.
        """
        segments = [seg for seg in urlsplit(self.target_url).path.split("/") if seg]
        culture = segments[0] if len(segments) > 1 else DEFAULT_CULTURE
        return culture, segments[-1]

    def scrape(self) -> str:
        """Fetch every chapter of the page, as one JSON array of API payloads.

        The first chapter is the bare URL; the rest carry ``chapterIndex``,
        whose values are read off the chapter nav rather than assumed (they are
        1-based and start at 2 — ``chapterIndex=0`` and ``1`` both return the
        first chapter).
        """
        culture, slug = self._culture_and_slug()
        url = CONTENT_API.format(slug=slug)
        first = _get(url, culture=culture)
        chapters = [first]

        nav = ((first.get("contentMain") or {}).get("sideNav") or {}).get("tagItems") or []
        for item in nav[1:]:
            index = parse_qs(urlsplit(item.get("url", "")).query).get("chapterIndex")
            if index:
                chapters.append(_get(url, culture=culture, chapterIndex=index[0]))
        return json.dumps(chapters, ensure_ascii=False)

    def parse(self, content: str) -> dict:
        """Build the nested section tree from the fetched chapter payloads."""
        chapters = json.loads(content)
        head = chapters[0].get("contentHead") or {}
        root = {
            "title": (head.get("title") or "").strip(),
            "url": self.target_url,
            # Seeds the loader's metadata (see documents._root_meta): "סוג" is
            # the gov.il page kind (מדריך / מידע), "נושא" its subject area —
            # the analogues of kolzchut's ArticleType / ArticleContentArea.
            "type": _first_title(head.get("promotedMetaData"), "סוג"),
            "area": _first_title(head.get("metaData"), "נושא"),
            "lead": (head.get("description") or "").strip(),
            "children": [],
        }
        if self.track:
            root["track"] = self.track

        nav = ((chapters[0].get("contentMain") or {}).get("sideNav") or {}).get("tagItems") or []
        for index, payload in enumerate(chapters):
            # A single-chapter page has no chapter nav; it still gets a wrapper
            # node, with an empty heading that the loader drops from the
            # breadcrumb — so its text reads as belonging straight to the title.
            heading = nav[index].get("title", "").strip() if index < len(nav) else ""
            chapter = {"heading": heading, "level": 2, "text": "", "children": []}
            self._fill_chapter(chapter, payload.get("contentMain") or {})
            _fold_barren_headings(chapter)
            if chapter["text"] or chapter["children"]:
                root["children"].append(chapter)
        return root

    def _fill_chapter(self, chapter: dict, main: dict) -> None:
        """Append one chapter's content blocks to its node.

        Prose and Q&A blocks interleave on the page, ordered by their shared
        ``order`` field, so they are merged on it rather than concatenated.
        """
        blocks = [("html", b.get("order", 0), b) for b in main.get("htmlContents") or []]
        blocks += [("faq", b.get("order", 0), b) for b in main.get("faqs") or []]

        # Open heading levels, deepest last; the chapter sits below every real
        # heading level so all of them nest under it.
        stack = [(_CHAPTER_DEPTH, chapter)]
        for kind, _, block in sorted(blocks, key=lambda b: b[1]):
            if kind == "html":
                self._fill_html(stack, block.get("sectionData") or "")
            else:
                self._fill_faq(chapter, block.get("sectionData") or {})

    @staticmethod
    def _fill_html(stack: list, html: str) -> None:
        """Walk one HTML fragment, nesting its sections by heading level."""
        soup = BeautifulSoup(html, "html.parser")
        # Inline footnote markers ("[1]") interrupt sentences mid-flow. Only the
        # in-text links go; the footnote list itself is real prose and stays —
        # its entries are anchors without an href, so they are untouched.
        for marker in soup.select('a[href^="#"]'):
            marker.decompose()

        for el in soup.find_all(recursive=False):
            if not isinstance(el, Tag):
                continue
            opened = _heading_of(el)
            if opened:
                heading, level = opened
                node = {"heading": heading, "level": level, "text": "", "children": []}
                while stack[-1][0] >= level:
                    stack.pop()
                stack[-1][1]["children"].append(node)
                stack.append((level, node))
                continue
            text = _text(el)
            if not text:
                continue
            node = stack[-1][1]
            node["text"] = f'{node["text"]}\n{text}'.strip()

    @staticmethod
    def _fill_faq(chapter: dict, section: dict) -> None:
        """Attach a Q&A block as one section with a child per question."""
        items = section.get("dataItems") or []
        children = []
        for item in items:
            answer = BeautifulSoup(item.get("regionContent") or "", "html.parser")
            text = "\n".join(
                filter(None, (_text(el) for el in answer.find_all(recursive=False)))
            )
            if text:
                children.append({
                    "heading": (item.get("headTitle") or "").strip(),
                    "level": 3,
                    "text": text,
                    "children": [],
                })
        if children:
            chapter["children"].append({
                "heading": (section.get("headTitle") or "").strip() or _FAQ_HEADING,
                "level": 2,
                "text": "",
                "children": children,
            })

    def format(self, content: str) -> str:
        """Return the reconstructed section tree as indented JSON."""
        return json.dumps(self.parse(content), ensure_ascii=False, indent=2)


def list_pages(landing_url: str, tab: str = PAGES_TAB) -> list[str]:
    """URLs of the content pages listed on a gov.il topic or office page.

    gov.il publishes no sitemap, so this is how a set of pages is discovered.
    The landing page's JSON carries, per tab, a ``sourceUrl`` that already
    embeds the right office/topic ids — following it avoids hardcoding any of
    them here. Only pages hosted on gov.il are returned; these lists also carry
    outbound links (e.g. to campus.gov.il) that are not content pages.
    """
    parts = urlsplit(landing_url)
    segments = [seg for seg in parts.path.split("/") if seg]
    culture = segments[0] if len(segments) > 1 else DEFAULT_CULTURE
    kind = next((seg for seg in segments if seg in LANDING_KINDS), "offices")
    landing = _get(LANDING_API.format(culture=culture, kind=kind, slug=segments[-1]))

    tabs = ((landing.get("loadingTabsStrip") or {}).get("stripModel") or {}).get("items") or []
    urls = []
    for item in tabs:
        source = item.get("sourceUrl") or ""
        if f"/{tab}?" not in source:
            continue
        for entry in _get(source):
            url = (entry.get("url") or "").strip()
            if not url:
                continue
            if "://" in url:  # absolute: an outbound link, not a gov.il page
                continue
            urls.append(f"{SITE}/{url.lstrip('/')}")
    return urls


# Pages about crime victims generally — they apply to this bot's population as
# written, so they carry no track.
_GENERAL = (
    "nifgaey-avera",            # rights, criminal-process stages, support services
    "guide-victims-of-killing",  # the סנ"ה programme for families of killing victims
)

# Pages whose entitlements run through the Ministry of Defence / National
# Insurance on the hostile-acts track. Their prose reads like ordinary
# victim-support material, which is exactly why they are stamped: unstamped,
# retrieval would happily offer them for violence inside an Arab community,
# where the eligibility does not carry over. The tag makes them cost nothing by
# default and still available once a user says the hostile-acts track applies.
_HOSTILE_ACTS = (
    "guide-bereaved-families",
    "guide-victims-of-hostile-acts-background",
)

# Not scraped at all — different populations from the ones this bot serves,
# recorded so the omission reads as a decision rather than an oversight:
# human_trafficking, victims-of-sexual-offenses.

if __name__ == "__main__":
    # The source URLs this module was last run with. ``list_pages`` sweeps a
    # whole topic or office, but read the notes above before widening the list:
    #
    #   for url in list_pages("https://www.gov.il/he/departments/topics/victims_of_crime_legal_aid"):
    #       GovILScraper(url)()
    for slug in _GENERAL:
        GovILScraper(f"https://www.gov.il/he/pages/{slug}")()
    for slug in _HOSTILE_ACTS:
        GovILScraper(f"https://www.gov.il/he/pages/{slug}", track="hostile_acts")()
