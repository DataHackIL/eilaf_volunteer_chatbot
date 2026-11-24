"""Scrape National Insurance (btl.gov.il) benefit pages into a section tree.

btl.gov.il is a SharePoint site that serves plain HTML (no SPA, no bot wall), so
the pages are read directly. A benefit is published as a *lobby* page
(``.../<Benefit>/Pages/default.aspx``) holding a short intro and a list of links
to the benefit's sub-pages — eligibility, how to claim, amounts — where the
actual content lives. A lobby is therefore fetched together with its sub-pages
and merged into one tree, the way gov_il.py merges a guide's chapters.
"""

import json
import time
from collections import deque
from urllib.parse import unquote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from data.scraping.base import (
    BaseHTMLScraper,
    cite,
    clean_text,
    fill_sections,
    fold_barren_headings,
)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# Politeness pause between page fetches: a lobby can list ~30 sub-pages.
_PAUSE = 0.5
# The lobby's list of sub-page links (SharePoint prefixes the id with its control path).
_CHILDREN = "[id$=SiteNodesControl_ChildrensDiv] a[href]"
# Lobby links to sign-language video renditions of pages scraped as text
# here; their own body is just the video player's fallback line.
_SKIP_LINK = "בשפת הסימנים"
# Page chrome inside the content block: print / send-to-friend / share buttons.
_STRIP = ".content-toolbar"
_CITATION = "ביטוח לאומי"
# Sentinel depth for a sub-page node, below every heading level, so the page's
# own <h2> sections nest under it.
_PAGE_DEPTH = 1


def _benefit_folder(url: str) -> str:
    """The path a benefit's pages share, e.g. ``/benefits/Disability/``.

    Everything before the ``Pages/`` segment, lowercased because SharePoint
    links mix the case of the same folder.
    """
    path = unquote(urlsplit(url).path).lower()
    return path.split("/pages/", 1)[0] + "/"


def _get(url: str) -> tuple[str, str]:
    """``(final URL, HTML)``: some lobby links are redirects to another section."""
    response = requests.get(url, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    time.sleep(_PAUSE)
    return response.url, response.text


def _path(url: str) -> str:
    return unquote(urlsplit(url).path).lower()


class BTLScraper(BaseHTMLScraper):
    """Scrape one btl.gov.il benefit page, with its sub-pages when it is a lobby.

    Output mirrors gov_il.py: a root carrying ``title``/``url``/``type``/
    ``lead`` (the lobby's intro), then one node per sub-page holding that
    page's heading-nested sections. Lobbies nest — the survivors' lobby links to
    an eligibility lobby that links to the actual eligibility pages — so a
    sub-page that is itself a lobby becomes a node holding its intro, then its
    own sub-pages. A plain content page comes out as a single page node with an
    empty heading, which the loader drops from the breadcrumb.

    Only pages inside the top lobby's benefit folder are followed: lobbies also
    link to *other* benefits (disability → vocational rehabilitation), which are
    scraped from their own lobby, not duplicated here. A link that redirects out
    of the folder (the vocational lobby's links to the war-updates hub)
    contributes that one page but is not crawled further. Pages are deduplicated
    on the URL they finally resolve to, which two such redirects can share.

    Every node cites its page as "ביטוח לאומי, <benefit>, <page>".
    """

    # Lobby nesting followed below the top page; the deepest the site goes
    # today is three (survivors → eligibility → who counts as a survivor).
    MAX_DEPTH = 3

    def __init__(self, *args, track: str = "", page_tracks: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._extension = "json"
        # As in gov_il.py: stamps the whole benefit with one compensation track.
        self.track = track
        # Sub-page title -> track, for lobbies that mix tracks: a general
        # benefit with a page for hostile-acts victims. The tag goes on that
        # page's node and the loader inherits it down to its sections.
        self.page_tracks = page_tracks or {}

    def scrape(self) -> str:
        """Fetch the page and its in-folder sub-pages, as nested JSON.

        Each page is ``{"url", "html", "children"}``, ``children`` being its
        sub-pages in the order its lobby lists them.
        """
        folder = _benefit_folder(self.target_url)
        host = urlsplit(self.target_url).netloc
        top = {"url": self.target_url, "children": []}
        seen = {_path(self.target_url)}
        # Breadth-first, so a page lobbies at several depths all list belongs
        # to the shallowest one. Depth-first, the first sub-lobby's crawl would
        # claim its siblings — a sub-lobby linking back to most of its parent's
        # pages is common here.
        queue = deque([(top, 0)])
        while queue:
            page, depth = queue.popleft()
            requested = page["url"]
            page["url"], page["html"] = _get(requested)
            # Compared as paths: requests re-encodes the Hebrew file names, so the
            # final URL differs in spelling from the link even without a redirect.
            if _path(page["url"]) != _path(requested):
                if _path(page["url"]) in seen:
                    page["html"] = None  # a redirect onto a page already taken
                    continue
                seen.add(_path(page["url"]))
            if depth >= self.MAX_DEPTH or not _path(page["url"]).startswith(folder):
                continue
            for link in BeautifulSoup(page["html"], "html.parser").select(_CHILDREN):
                child = urljoin(page["url"], link["href"])
                if urlsplit(child).netloc != host or _SKIP_LINK in link.get_text():
                    continue
                if _path(child) in seen or not _path(child).startswith(folder):
                    continue
                seen.add(_path(child))
                sub = {"url": child, "children": []}
                page["children"].append(sub)
                queue.append((sub, depth + 1))

        def prune(page: dict) -> dict:
            page["children"] = [prune(sub) for sub in page["children"] if sub["html"]]
            return page

        return json.dumps(prune(top), ensure_ascii=False)

    @staticmethod
    def _read_page(html: str) -> tuple[str, BeautifulSoup | None]:
        """A page's title and its content block (``None`` when it has none)."""
        soup = BeautifulSoup(html, "html.parser")
        main = soup.select_one("#mainContent") or soup
        title_el = main.find("h1")
        title = clean_text(title_el.get_text(" ", strip=True)) if title_el else ""
        body = main.select_one(".opener-txt")
        if body:
            for junk in body.select(_STRIP):
                junk.decompose()
        return title, body

    def _sections(self, page: dict, heading: str, citation: str) -> dict:
        """A node holding one page's own heading-nested sections, all cited to it."""
        _, body = self._read_page(page["html"])
        node = {"heading": heading, "level": 2, "text": "", "children": []}
        if body is not None:
            fill_sections(body, node, _PAGE_DEPTH)
            fold_barren_headings(node)
        cite(node, citation, page["url"])
        return node

    def _page_node(self, page: dict, citation: str) -> dict | None:
        """A sub-page's node: its own sections, then its sub-pages' nodes.

        The sections are cited before the sub-pages are attached, since
        :func:`cite` stamps a whole subtree and each sub-page cites itself.
        ``None`` for a page with nothing to say and no sub-pages that do.
        """
        title, _ = self._read_page(page["html"])
        citation = ", ".join(filter(None, (citation, title)))
        node = self._sections(page, title, citation)
        node["children"].extend(self._subpages(page, citation))
        if title in self.page_tracks:
            node["track"] = self.page_tracks[title]
        return node if node["text"] or node["children"] else None

    def _subpages(self, page: dict, citation: str) -> list[dict]:
        nodes = (self._page_node(sub, citation) for sub in page["children"])
        return [node for node in nodes if node]

    def parse(self, content: str) -> dict:
        page = json.loads(content)
        title, _ = self._read_page(page["html"])
        root = {
            "title": title,
            "url": self.target_url,
            "type": "btl",
            "lead": "",
            "children": [],
        }
        if self.track:
            root["track"] = self.track
        root_citation = ", ".join(filter(None, (_CITATION, title)))
        cite(root, root_citation, self.target_url)

        own = self._sections(page, "", root_citation)
        if page["children"]:
            # A lobby's own body is a one-paragraph intro: the root's lead, the
            # role gov.il's page description plays.
            root["lead"] = own["text"]
            own["text"] = ""
        if own["children"] or own["text"]:
            root["children"].append(own)
        root["children"].extend(self._subpages(page, root_citation))
        return root

    def format(self, content: str) -> str:
        """Return the merged section tree as indented JSON."""
        return json.dumps(self.parse(content), ensure_ascii=False, indent=2)


_BASE = "https://www.btl.gov.il/benefits"

# Sub-pages on a separate compensation track (see data.enrich.base.Track),
# keyed by their title as the lobby lists it. They read like the general
# benefit around them, which is why they are stamped rather than left to the
# enricher: retrieval would otherwise offer hostile-acts rehabilitation to a
# victim of violence inside the community, where it does not apply.
_PAGE_TRACKS = {
    "שיקום לנפגעי פעולות איבה": "hostile_acts",
    "שיקום למשפחות חללי פעולות איבה": "hostile_acts",
    "רפורמת ״נפש אחת״ לנפגעי איבה": "hostile_acts",
    "זכויות בני משפחה של חטופים ששבו": "hostile_acts",
    "עדכונים ומידע בעקבות מלחמת חרבות ברזל": "hostile_acts",
    "מקבל תגמול ממשרד הביטחון": "military",
    "חייל בשירות חובה, חייל מתנדב ומשרת בשירות לאומי": "military",
}

# Benefit lobbies, each scraped with all its sub-pages. The survivors' lobby
# already includes the death-grant page (מענק פטירה).
_LOBBIES = (
    f"{_BASE}/Disability/Pages/default.aspx",               # general disability
    f"{_BASE}/Vocational_Rehabilitation/Pages/default.aspx",
    f"{_BASE}/Survivors_%20Insurance/Pages/default.aspx",   # survivors' pension
)

# Single content pages outside any lobby scraped above.
_PAGES = (
    f"{_BASE}/Pages/domesticviolence.aspx",  # allowance for a child orphaned by domestic violence
)

if __name__ == "__main__":
    for url in (*_LOBBIES, *_PAGES):
        BTLScraper(url, page_tracks=_PAGE_TRACKS)()
