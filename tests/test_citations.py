"""Tests for citations: a tree node's ``source`` list reaching the model's prompt.

The docx drafts cite their facts in footnotes, and the docx is never shown to
users, so the citations are the only reference an answer can point them to.
These pin that they survive the converter's bullet grouping, every way the
loader reshapes nodes into Documents (split, subtree rollup, leaf packing), the
prompt rendering and the reply's source list — plus nevo's clause references.
"""

from __future__ import annotations

import json

from chatbot.rag.documents.documents import Document, _flatten_json_tree
from chatbot.rag.generator._prompt import build_user_message
from data.scraping.docx_source import DocxToJsonConverter
from data.scraping.nevo import _add_sources

LAW = {"text": "חוק זכויות נפגעי עבירה", "url": "https://www.nevo.co.il/law"}
PAGE = {"text": "אתר gov.il", "url": "https://www.gov.il/page"}
BARE = {"text": "https://www.gov.il/bare", "url": "https://www.gov.il/bare"}


def test_converter_groups_bullet_citations_without_duplicates(tmp_path):
    paragraphs = [
        {"text": "intro", "list": False, "source": [LAW]},
        {"text": "bullet 1", "list": True, "source": [PAGE]},
        {"text": "bullet 2", "list": True, "source": [PAGE, BARE]},
        {"text": "outro", "list": False, "source": []},
    ]
    converter = DocxToJsonConverter(tmp_path / "doc.docx", static_dir=tmp_path)
    children = converter.parse(json.dumps(paragraphs))["children"]

    assert [c["source"] for c in children] == [[LAW], [PAGE, BARE], []]


def tree(**a_extra) -> dict:
    return {
        "title": "T",
        "children": [
            {
                "heading": "A",
                "text": "section A",
                "source": [LAW],
                "children": [
                    {"heading": "A1", "text": "clause A1", "source": [PAGE]},
                    {"heading": "A2", "text": "clause A2", "source": [LAW]},
                ],
                **a_extra,
            },
            {"heading": "B", "text": "uncited B"},
        ],
    }


def test_each_node_keeps_its_own_citations():
    by_id = {d.id: d for d in _flatten_json_tree(tree(), source="s.json")}

    assert by_id["s.json#0"].citations == [LAW]
    assert by_id["s.json#0/0"].citations == [PAGE]
    assert by_id["s.json#1"].citations == []


def test_rolled_up_subtree_cites_its_root_plus_other_links():
    # A2's citation shares A's URL, so A's covers it; A1's links elsewhere.
    by_id = {d.id: d for d in _flatten_json_tree(tree(), source="s.json", merge_chars=500)}

    assert by_id["s.json#0"].citations == [LAW, PAGE]


def test_packed_leaves_cite_their_parent_plus_other_links():
    # A is too big to roll up whole, so its two leaf clauses pack into one bin.
    docs = _flatten_json_tree(tree(), source="s.json", merge_chars=25)
    packed = next(d for d in docs if "clause A1" in d.text)

    assert "clause A2" in packed.text
    assert packed.citations == [LAW, PAGE]


def test_same_url_clause_citations_collapse_to_the_section():
    section = {"text": "חוק, סעיף 7", "url": "https://nevo/law"}
    t = {
        "title": "T",
        "children": [
            {
                "marker": "7.",
                "text": "",
                "source": [section],
                "children": [
                    {"marker": f"({m})", "text": f"clause {m}",
                     "source": [{"text": f"חוק, סעיף 7({m})", "url": "https://nevo/law"}]}
                    for m in "אבג"
                ],
            }
        ],
    }
    (doc,) = _flatten_json_tree(t, source="s.json", merge_chars=500)

    assert doc.citations == [section]


def test_uncovered_packed_leaves_keep_their_own_citations():
    # Top-level leaves have no citing parent (the root), as in a docx tree.
    t = {
        "title": "T",
        "children": [
            {"text": "one", "source": [LAW]},
            {"text": "two", "source": [PAGE]},
        ],
    }
    (doc,) = _flatten_json_tree(t, source="s.json", merge_chars=500)

    assert doc.citations == [LAW, PAGE]


def test_useless_node_contributes_no_citations():
    t = tree()
    t["children"][0]["children"][0]["useful"] = False
    t["children"][0]["source"] = []
    by_id = {d.id: d for d in _flatten_json_tree(t, source="s.json", merge_chars=500)}

    assert by_id["s.json#0"].citations == [LAW]


def test_split_pieces_all_repeat_the_citation():
    t = {"title": "T", "children": [{"text": "משפט ארוך. " * 30, "source": [LAW]}]}
    docs = _flatten_json_tree(t, source="s.json", max_chars=60)

    assert len(docs) > 1
    assert all(d.citations == [LAW] for d in docs)


def test_prompt_lists_citations_under_their_passage():
    message = build_user_message(
        "q",
        [
            Document(text="cited", citations=[LAW, BARE]),
            Document(text="uncited"),
        ],
    )

    assert f"[1] cited\nמקור: {LAW['text']} ({LAW['url']})\nמקור: {BARE['url']}\n\n" in message
    assert "[2] uncited\n\n" in message


def test_whatsapp_sources_are_deduplicated_citations():
    from app.whatsapp.conversation import _format_sources

    contexts = [
        Document(text="a", source="a.json", citations=[LAW, BARE]),
        Document(text="b", source="b.json", citations=[LAW]),
        Document(text="c", source="c.json"),
    ]

    assert _format_sources(contexts) == (
        f"• {LAW['text']} ({LAW['url']})\n• {BARE['url']}"
    )
    assert _format_sources([Document(text="c", source="c.json")]) == ""


def test_nevo_cites_clause_paths_chapters_and_schedule_items():
    url = "https://www.nevo.co.il/law"
    t = {
        "title": "חוק",
        "children": [
            {"marker": "", "text": "preamble", "children": []},
            {
                "type": "chapter",
                "title": "פרק א'",
                "children": [
                    {
                        "type": "section", "marker": "3 א.", "text": "",
                        "children": [
                            {"marker": "(ב)", "text": "b", "children": [
                                {"marker": "(1)", "text": "b1", "children": []},
                                {"marker": "", "text": "definition", "children": []},
                            ]},
                        ],
                    },
                ],
            },
            {"type": "chapter", "title": "תוספת שניה", "children": []},
            {"type": "chapter", "title": "(סעיף 7(3))", "children": [
                {"type": "section", "marker": "28.", "text": "item", "children": []},
            ]},
        ],
    }
    _add_sources(t, url)

    def ref(node):
        (citation,) = node["source"]
        assert citation["url"] == url
        return citation["text"]

    preamble, chapter, schedule, authority = t["children"]
    section = chapter["children"][0]
    sub_b = section["children"][0]
    assert ref(preamble) == "חוק"
    assert ref(chapter) == "חוק, פרק א'"
    assert ref(section) == "חוק, סעיף 3א"
    assert ref(sub_b) == "חוק, סעיף 3א(ב)"
    assert ref(sub_b["children"][0]) == "חוק, סעיף 3א(ב)(1)"
    assert ref(sub_b["children"][1]) == "חוק, סעיף 3א(ב)"
    assert ref(schedule) == "חוק, תוספת שניה"
    # A pseudo-chapter inside a schedule stays in it; items are פרט, not סעיף.
    assert ref(authority) == "חוק, תוספת שניה"
    assert ref(authority["children"][0]) == "חוק, תוספת שניה, פרט 28"


def test_lead_and_top_level_packs_cite_the_root():
    page = {"text": "כל זכות, דף", "url": "https://kolzchut/page"}
    t = {
        "title": "T",
        "lead": "intro",
        "source": [page],
        "children": [
            {"heading": "a", "text": "one", "source": [page]},
            {"heading": "b", "text": "two", "source": [page]},
        ],
    }
    docs = _flatten_json_tree(t, source="s.json", merge_chars=500)

    assert [d.citations for d in docs] == [[page], [page]]


def test_govil_chapters_cite_their_chapter_page():
    from data.scraping.gov_il import GovILScraper

    def chapter(body):
        return {"htmlContents": [{"order": 0, "sectionData": f"<p>{body}</p>"}]}

    nav = {"tagItems": [
        {"title": "רקע", "url": "/he/pages/guide"},
        {"title": "זכאים", "url": "/he/pages/guide?chapterIndex=2"},
    ]}
    payloads = [
        {"contentHead": {"title": "מדריך", "description": "lead"},
         "contentMain": {**chapter("first"), "sideNav": nav}},
        {"contentMain": chapter("second")},
    ]
    url = "https://www.gov.il/he/pages/guide"
    tree = GovILScraper(url).parse(json.dumps(payloads))

    assert tree["source"] == [{"text": "gov.il, מדריך", "url": url}]
    first, second = tree["children"]
    assert first["source"] == [{"text": "gov.il, מדריך, רקע", "url": url}]
    assert second["source"] == [
        {"text": "gov.il, מדריך, זכאים", "url": f"{url}?chapterIndex=2"}
    ]
