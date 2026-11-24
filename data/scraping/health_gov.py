"""Scrape Ministry of Health information pages (me.health.gov.il) into a section tree.

The site is an Umbraco CMS serving plain HTML, so pages are read directly. Unlike
gov.il's or btl.gov.il's flat article bodies, an Umbraco page is a stack of
component blocks — accordions, icon cards, tabbed panels — each wrapping its own
headings and paragraphs, so the tree is built from a document-order walk of the
article (see :func:`data.scraping.base.fill_sections`) rather than from its
top-level children.
"""

import json
import re

from bs4 import BeautifulSoup

from data.scraping.base import (
    BaseHTMLScraper,
    cite,
    clean_text,
    fill_sections,
    fold_barren_headings,
)

# The article body; everything outside it is site navigation.
_ARTICLE = "#article-content"
# Components inside the article that are not its content: the "quick read"
# box (a digest that repeats the article's own text, at times verbatim), the
# "you might also be interested in" link table, the "did this page help you?"
# widget, reference cards that only link to other pages, and the title header.
_STRIP = ", ".join((
    "#twoMinutesV2",
    ".external-component",
    ".feedback-component",
    ".ref-block",
    ".article-content-header",
))
# Contact cards (the health funds' support lines) whose name is only in the
# logo: the card's accessible label is the one place it is spelled out.
_LABELLED_CARDS = ".info[aria-label]"
_URL = re.compile(r"https?://\S+")
_CITATION = "משרד הבריאות"
# Sentinel depth for the body node, below every heading level.
_BODY_DEPTH = 1


class HealthGovScraper(BaseHTMLScraper):
    """Scrape one me.health.gov.il article into a nested section tree.

    Output mirrors the other scrapers: a root carrying ``title``/``url``/
    ``type``/``lead`` over one empty-headed body node, which the loader drops
    from the breadcrumb, holding the article's heading-nested sections. Every
    node cites the page as "משרד הבריאות, <title>".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extension = "json"

    def parse(self, content: str) -> dict:
        soup = BeautifulSoup(content, "html.parser")
        article = soup.select_one(_ARTICLE)
        if article is None:
            raise RuntimeError(f"no {_ARTICLE} on {self.target_url!r}: page layout changed?")
        title_el = article.find("h1")
        title = clean_text(title_el.get_text(" ", strip=True)) if title_el else ""
        for junk in article.select(_STRIP):
            junk.decompose()
        for card in article.select(_LABELLED_CARDS):
            card.string = clean_text(_URL.sub("", card["aria-label"]))

        root = {
            "title": title,
            "url": self.target_url,
            "type": "health.gov.il",
            "lead": "",
            "children": [],
        }
        node = {"heading": "", "level": 2, "text": "", "children": []}
        fill_sections(article, node, _BODY_DEPTH)
        fold_barren_headings(node)
        if node["text"] or node["children"]:
            root["children"].append(node)
        cite(root, ", ".join(filter(None, (_CITATION, title))), self.target_url)
        return root

    def format(self, content: str) -> str:
        """Return the reconstructed section tree as indented JSON."""
        return json.dumps(self.parse(content), ensure_ascii=False, indent=2)


_BASE = "https://me.health.gov.il/mental-health/therapy-rehabilitation/public-care/community-treatment"

if __name__ == "__main__":
    # Public mental-health care, and the resilience centres (מרכזי חוסן).
    HealthGovScraper(f"{_BASE}/treatment-services/")()
    HealthGovScraper(f"{_BASE}/resilience-center/")()
