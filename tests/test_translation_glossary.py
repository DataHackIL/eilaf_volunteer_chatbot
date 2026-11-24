"""The Arabic dialect glossary that runs in front of the query translator."""

from __future__ import annotations

from chatbot.rag.translator import Glossary, GlossaryTranslator, Translator
from chatbot.rag.translator.glossary import load_glossary

_ENTRIES = {
    "جوزي": "زوجي",
    "الطخ": "إطلاق النار",
    "الست": "المرأة",
    "مركز الصمود": "מרכז חוסן",
    "مركز": "مقر",
}


class _Echo(Translator):
    def to_hebrew(self, text, source):
        return text


def test_whole_words_only():
    g = Glossary(_ENTRIES)
    assert g.apply("جوزي بضربني") == "زوجي بضربني"
    assert g.apply("جوزيف") == "جوزيف"  # a name that starts with the term


def test_attached_prefixes_are_kept():
    g = Glossary(_ENTRIES)
    assert g.apply("وجوزي") == "وزوجي"
    assert g.apply("بالطخ") == "بإطلاق النار"


def test_lam_plus_article_contraction():
    assert Glossary(_ENTRIES).apply("للست") == "للمرأة"


def test_longest_term_wins_and_hebrew_takes_a_spelled_out_prefix():
    g = Glossary(_ENTRIES)
    assert g.apply("مركز الصمود") == "מרכז חוסן"
    assert g.apply("بمركز الصمود") == "في מרכז חוסן"
    assert g.apply("مركز الشرطة") == "مقر الشرطة"


def test_diacritics_do_not_block_a_match():
    assert Glossary(_ENTRIES).apply("جَوزي") == "زوجي"


def test_only_arabic_input_is_rewritten():
    t = GlossaryTranslator(_Echo(), Glossary(_ENTRIES))
    assert t.to_hebrew("جوزي", "Arabic") == "زوجي"
    assert t.to_hebrew("جوزي", "English") == "جوزي"


def test_glossary_csv_loads_with_bom_and_skips_blank_rows(tmp_path):
    path = tmp_path / "g.csv"
    path.write_text(
        "arabic,replacement,notes\nجَوزي,زوجي,husband\n,,\nمصاري,,no replacement\n",
        encoding="utf-8-sig",
    )
    assert load_glossary(path) == {"جوزي": "زوجي"}


def test_missing_glossary_is_empty_not_an_error(tmp_path, caplog):
    assert load_glossary(tmp_path / "absent.csv") == {}
    assert "no dialect glossary" in caplog.text
    assert Glossary.from_csv(tmp_path / "absent.csv").apply("جوزي") == "جوزي"
