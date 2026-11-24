import os
from pathlib import Path

from chatbot.rag.pipeline.pipeline import RAW_STATIC_DIR

# Heading tag -> nesting depth, shared by the scrapers whose hierarchy lives in
# <hN> levels rather than in DOM nesting.
HEADING_LEVELS = {"h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}


def cite(node: dict, text: str, url: str) -> None:
    """Set ``node``'s and all its descendants' ``source`` to one citation.

    For sources cited by page, where every section of a page (or chapter) points
    the user at the same place.
    """
    node["source"] = [{"text": text, "url": url}]
    for child in node.get("children", []):
        cite(child, text, url)


def clean_text(text: str) -> str:
    """``text`` with ``&nbsp;`` and zero-width marks normalised away.

    Both are pervasive in CMS editor output and would otherwise reach the
    embedder as odd characters.
    """
    text = text.replace("\xa0", " ").replace("​", "").replace("‏", "")
    return " ".join(text.split())


def fill_sections(container, node: dict, depth: int = 1) -> None:
    """Append ``container``'s content to ``node`` as a heading-nested tree.

    Walks the container's descendants in document order, so it works whether
    the CMS emits headings and paragraphs as flat siblings or wraps them in
    cards, accordions and tabs: every ``<hN>`` opens a section at its level and
    every text block lands in the section open at that point. ``depth`` is the
    sentinel level ``node`` sits at, below every heading level so all of them
    nest under it.

    Text blocks are ``<p>``, list items, and table rows (cells joined with
    " | ", since an amounts table read as prose loses which figure is which).
    Only the innermost are read: a list item wrapping paragraphs or a heading
    (an icon card) is not a block itself, its contents are, so each piece of
    text is read once and a card's title still opens its section. A heading
    that wraps another heading (an accordion header holding its own title)
    counts once, at the outer tag's level: that is the one placing the section
    in the page.
    """
    stack = [(depth, node)]
    headings = tuple(HEADING_LEVELS)
    blocks = ("p", "li", "tr")
    for el in container.find_all((*headings, *blocks)):
        if el.name in HEADING_LEVELS:
            if el.find_parent(headings):
                continue
            heading = clean_text(el.get_text(" ", strip=True))
            if not heading:
                continue
            level = HEADING_LEVELS[el.name]
            section = {"heading": heading, "level": level, "text": "", "children": []}
            while stack[-1][0] >= level:
                stack.pop()
            stack[-1][1]["children"].append(section)
            stack.append((level, section))
            continue
        if el.find_parent(headings) or el.find((*headings, *blocks, "table")):
            continue  # part of a heading, or a wrapper whose contents are read
        if any(parent.name in blocks and not parent.find((*headings, *blocks, "table"))
               for parent in el.parents):
            continue  # inside a block that was read whole
        if el.name == "tr":
            cells = [clean_text(c.get_text(" ", strip=True)) for c in el.find_all(("th", "td"))]
            text = " | ".join(c for c in cells if c)
        else:
            text = clean_text(el.get_text(" ", strip=True))
        if text:
            target = stack[-1][1]
            target["text"] = f'{target["text"]}\n{text}'.strip()


def fold_barren_headings(node: dict) -> None:
    """Turn headings that introduce nothing back into their parent's text.

    Editors sometimes mark a plain sentence as a heading for emphasis; left as
    one it would carry its prose *as a heading* over an empty body, and the
    loader — which only emits text-bearing nodes — would drop it entirely.
    Folding bottom-up, so a heading whose children all fold is reconsidered
    once it has absorbed them.
    """
    kept = []
    for child in node.get("children", []):
        fold_barren_headings(child)
        if child["text"] or child["children"]:
            kept.append(child)
        elif child["heading"]:
            node["text"] = f'{node["text"]}\n{child["heading"]}'.strip()
    node["children"] = kept


class BaseScraper:
    def __init__(self, target_url: str, static_dir: os.PathLike = RAW_STATIC_DIR):
        self.target_url = target_url
        self.static_dir = Path(static_dir)
        self._extension = 'txt'

    def scrape(self) -> str:
        """Scrape the target URL and return the scraped content."""
        raise NotImplementedError

    def format(self, content: str) -> str:
        """Format the scraped content for storage."""
        raise NotImplementedError
    
    def save(self, content: str, filename: str) -> None:
        """Save the formatted content to a file in the static directory."""
        self.static_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.static_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def __call__(self) -> None:
        """Scrape, format, and save the content."""
        content = self.scrape()
        formatted_content = self.format(content)
        filename = f"{self.target_url.replace('/', '_')}.{self._extension}"
        self.save(formatted_content, filename)


class BaseHTMLScraper(BaseScraper):
    def scrape(self) -> str:
        """Scrape the target URL using HTTP GET and return the response text."""
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }


        response = requests.get(self.target_url, headers=headers)
        response.raise_for_status()
        return response.text
    
if __name__ == "__main__":
    # Example usage: scrape a URL and save the content to the static directory.
    scraper = BaseHTMLScraper("https://www.nevo.co.il/law_html/law00/71835.htm")
    scraper()