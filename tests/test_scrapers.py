"""Offline tests for the btl.gov.il, me.health.gov.il and PDF scrapers.

Each parser is fed a small hand-built page reproducing the structure that made
it tricky, and every tree it returns is checked against the contract the loader
relies on (:func:`assert_tree_contract`). Nothing here touches the network.
"""

from __future__ import annotations

import json

from bs4 import BeautifulSoup

from chatbot.rag.documents.documents import _flatten_json_tree
from data.scraping.base import fill_sections
from data.scraping.btl import BTLScraper
from data.scraping.health_gov import HealthGovScraper
from data.scraping.pdf_source import _logical_line, _reading_order


def _walk(node: dict):
    for child in node.get("children", []):
        yield child
        yield from _walk(child)


def assert_tree_contract(tree: dict) -> None:
    """What the loader needs from any scraper's tree.

    A titled root with children; every node a heading/text/children dict citing
    at least one ``{"text", "url"}`` source; and flattening yields documents,
    each with a citation.
    """
    assert tree["title"]
    assert tree["children"]
    for node in [tree, *_walk(tree)]:
        assert node["source"], node
        assert all(c["text"] and c["url"] for c in node["source"]), node
        assert isinstance(node["children"], list)
    for node in _walk(tree):
        assert {"heading", "text", "children"} <= set(node), node
    docs = _flatten_json_tree(tree, source="t.json", merge_chars=0)
    assert docs
    assert all(d.citations for d in docs)


# ── fill_sections ────────────────────────────────────────────────────────────


def test_fill_sections_walks_nested_components():
    """Cards and accordions wrap their own headings; order, not nesting, decides."""
    html = """
    <div>
      <p>intro</p>
      <div class="card"><h2>Section A</h2><div><p>a body</p></div></div>
      <div class="card"><h2 class="header"><button><h3>Inner title</h3></button></h2>
        <ul><li><div><h4>Card</h4><p>card body</p></div></li><li>plain item</li></ul>
      </div>
      <table><tr><th>degree</th><th>amount</th></tr><tr><td>100%</td><td>4,711</td></tr></table>
    </div>"""
    node = {"heading": "", "level": 2, "text": "", "children": []}
    fill_sections(BeautifulSoup(html, "html.parser").div, node)

    assert node["text"] == "intro"
    a, inner = node["children"]
    assert (a["heading"], a["text"]) == ("Section A", "a body")
    # The wrapping <h2> counts once, at its own level; the card nests under it.
    assert inner["heading"] == "Inner title"
    card = inner["children"][0]
    # Text after the card stays in the section still open, in document order;
    # each piece read once — the list item holding the card is not read whole.
    assert card["heading"] == "Card"
    assert card["text"].splitlines() == ["card body", "plain item", "degree | amount", "100% | 4,711"]


# ── btl.gov.il ───────────────────────────────────────────────────────────────


def _btl_page(title: str, body: str) -> str:
    return f"""<html><body><div id="mainContent"><h1>{title}</h1>
    <div class="opener-txt"><div class="content-toolbar">הדפס שלח לחבר</div>{body}</div>
    </div></body></html>"""


def _btl_lobby_json() -> str:
    lobby = "https://www.btl.gov.il/benefits/X/Pages/default.aspx"
    page = {
        "url": lobby,
        "html": _btl_page("קצבה", "<p>פתיח הקצבה.</p>"),
        "children": [
            {
                "url": "https://www.btl.gov.il/benefits/X/Pages/elig.aspx",
                "html": _btl_page("תנאי זכאות", "<p>מי זכאי.</p><h2>חריגים</h2><p>חריג.</p>"),
                "children": [],
            },
            {
                "url": "https://www.btl.gov.il/benefits/X/Pages/hostile.aspx",
                "html": _btl_page("לנפגעי פעולות איבה", "<p>מסלול נפרד.</p>"),
                "children": [
                    {
                        "url": "https://www.btl.gov.il/benefits/X/Pages/hostile-sub.aspx",
                        "html": _btl_page("השתתפות במימון", "<p>מימון לימודים.</p>"),
                        "children": [],
                    }
                ],
            },
        ],
    }
    return json.dumps(page, ensure_ascii=False)


def test_btl_lobby_intro_becomes_lead_and_pages_cite_themselves():
    scraper = BTLScraper(
        "https://www.btl.gov.il/benefits/X/Pages/default.aspx",
        page_tracks={"לנפגעי פעולות איבה": "hostile_acts"},
    )
    tree = scraper.parse(_btl_lobby_json())
    assert_tree_contract(tree)

    assert tree["title"] == "קצבה"
    assert tree["lead"] == "פתיח הקצבה."
    elig, hostile = tree["children"]
    assert elig["heading"] == "תנאי זכאות"
    assert elig["source"][0]["url"].endswith("/elig.aspx")
    assert elig["source"][0]["text"] == "ביטוח לאומי, קצבה, תנאי זכאות"
    # The page's own <h2> nests under its page node, citing the same page.
    assert elig["children"][0]["heading"] == "חריגים"
    assert elig["children"][0]["source"] == elig["source"]
    # The toolbar chrome never reaches the text.
    assert all("הדפס" not in n["text"] for n in _walk(tree))
    # A nested sub-page cites itself, not its parent.
    sub = hostile["children"][0]
    assert sub["source"][0]["url"].endswith("/hostile-sub.aspx")


def test_btl_page_track_reaches_every_segment_under_it():
    scraper = BTLScraper(
        "https://www.btl.gov.il/benefits/X/Pages/default.aspx",
        page_tracks={"לנפגעי פעולות איבה": "hostile_acts"},
    )
    docs = _flatten_json_tree(scraper.parse(_btl_lobby_json()), source="t.json", merge_chars=0)
    tracks = {d.text.splitlines()[-1]: d.meta.get("track") for d in docs}
    assert tracks["מסלול נפרד."] == "hostile_acts"
    assert tracks["מימון לימודים."] == "hostile_acts"  # inherited by the sub-page
    assert tracks["מי זכאי."] is None


def test_btl_plain_page_is_one_unheaded_node():
    url = "https://www.btl.gov.il/benefits/Pages/single.aspx"
    page = {"url": url, "html": _btl_page("עמוד", "<p>תוכן.</p>"), "children": []}
    tree = BTLScraper(url).parse(json.dumps(page, ensure_ascii=False))
    assert_tree_contract(tree)
    assert tree["lead"] == ""
    assert [(c["heading"], c["text"]) for c in tree["children"]] == [("", "תוכן.")]


# ── me.health.gov.il ─────────────────────────────────────────────────────────


def test_health_strips_digest_and_chrome_and_names_contact_cards():
    html = """<html><body><div id="article-content">
      <div class="article-content-header"><h1>מרכזי חוסן</h1></div>
      <div id="twoMinutesV2"><h2>תקציר לקריאה מהירה</h2><p>הפתיח.</p></div>
      <div class="umb-block-list">
        <p>הפתיח.</p>
        <h2>איך פונים</h2><p>בטלפון.</p>
        <h3>מוקדים</h3>
        <ul><li><div class="info" aria-label="קופת חולים כללית *8703 https://www.clalit.co.il/">
          <img alt=""/><a href="tel:*8703">*8703</a></div></li></ul>
        <div class="feedback-component"><h3>האם המידע עזר?</h3><p>כן / לא</p></div>
        <div class="external-component"><h2>אולי יעניין אותך</h2><ul><li>קישור</li></ul></div>
      </div></div></body></html>"""
    tree = HealthGovScraper("https://me.health.gov.il/x/").parse(html)
    assert_tree_contract(tree)

    assert tree["title"] == "מרכזי חוסן"
    texts = [n["text"] for n in _walk(tree)]
    assert sum(t.count("הפתיח.") for t in texts) == 1  # the digest's copy is gone
    headings = [n["heading"] for n in _walk(tree)]
    assert "האם המידע עזר?" not in headings
    assert "אולי יעניין אותך" not in headings
    cards = next(n for n in _walk(tree) if n["heading"] == "מוקדים")
    assert cards["text"] == "קופת חולים כללית *8703"
    assert all(c["text"] == "משרד הבריאות, מרכזי חוסן" for c in tree["source"])


# ── PDF text reconstruction ──────────────────────────────────────────────────


def _glyphs(word_runs: list[tuple[str, float]], size: float = 10.0, width: float = 5.0):
    """Glyphs laid out right to left: each run starts at its x and grows leftward."""
    chars = []
    for text, right in word_runs:
        x = right
        for c in text:
            chars.append({"c": c, "x0": x - width, "x1": x, "size": size})
            x -= width
    return chars


def test_logical_line_orders_hebrew_right_to_left():
    # "שלום עולם" drawn right to left with a measurable word gap.
    chars = _glyphs([("שלום", 100.0), ("עולם", 75.0)])
    assert _logical_line(chars) == "שלום עולם"


def test_logical_line_keeps_left_to_right_runs_in_reading_order():
    # Hours drawn left to right inside a Hebrew line: "פתוח 08:00-20:00".
    hours = "08:00-20:00"
    chars = _glyphs([("פתוח", 200.0)])
    x = 100.0
    for c in hours:  # LTR: grows rightward from x=100
        chars.append({"c": c, "x0": x, "x1": x + 5.0, "size": 10.0})
        x += 5.0
    assert _logical_line(chars) == f"פתוח {hours}"


def test_logical_line_ignores_a_stored_space_inside_a_letter():
    # The Word-export case: a space glyph positioned inside the final letter.
    chars = _glyphs([("עבירה", 100.0)])
    he = chars[-1]
    chars.append({"c": " ", "x0": he["x0"] + 0.5, "x1": he["x0"] + 3.0, "size": 10.0})
    assert _logical_line(chars) == "עבירה"


def test_logical_line_keeps_a_stored_space_in_a_tight_gap():
    # The justified-layout case: words abut, and only the space glyph separates them.
    chars = _glyphs([("מסמך", 100.0), ("זה", 80.0)])
    chars.append({"c": " ", "x0": 79.5, "x1": 80.5, "size": 10.0})
    assert _logical_line(chars) == "מסמך זה"


def test_reading_order_reads_columns_right_to_left_under_a_spread_title():
    def block(name, x0, x1, y0):
        return {"text": name, "x0": x0, "x1": x1, "y0": y0, "y1": y0 + 10}

    blocks = [
        block("left body", 40, 370, 120),
        block("right body", 420, 680, 120),
        block("left head", 230, 350, 90),
        block("title", 230, 615, 5),  # just under half the page, straddling both
        block("right head", 540, 670, 90),
    ]
    order = [b["text"] for b in _reading_order(blocks, page_width=842)]
    assert order == ["title", "right head", "right body", "left head", "left body"]
