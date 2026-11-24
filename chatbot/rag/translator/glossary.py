"""Dialect glossary: rewrites known trouble words before the translator sees them.

Translation models handle Modern Standard Arabic far better than spoken
Palestinian Arabic, and they don't know the Israeli names of institutions and
benefits. The benchmark showed both failing on exactly the words retrieval
depends on: جوزي ("my husband") came out as a name, انطخ ("was shot") as "freaked
out", مركز الصمود as "resistance center" rather than מרכז חוסן.

``GlossaryTranslator`` wraps any translator and, for Arabic input only, replaces
each glossary term before translating. A replacement is either:

* standard Arabic (جوزي → زوجي), which the model then translates properly, or
* the Hebrew term itself (مركز الصمود → מרכז חוסן). NLLB copies Hebrew it finds
  in the input straight through, so the corpus's own term reaches retrieval.

Terms match whole words, with up to two attached one-letter prefixes
(و ف ب ل ك: "and", "so", "in/with", "to/for", "like"), so وجوزي and بالطخ match
the entries جوزي and الطخ, and للست (ل + ال contracted) matches الست. Write
an entry with ال when the word takes it, since
the replacement often has to change shape (الطخ → إطلاق النار). Longer terms win
over shorter ones they contain. Diacritics and tatweel are stripped first.

The glossary is a CSV (``arabic,replacement,notes``) that speakers can extend
without touching code. It lives in the static store with the corpus
(``GLOSSARY_PATH``: ``data/static/glossary_ar.csv`` unless ``EILAF_STATIC_DIR``
moves it), is read once when the translator is built, and is not in git. A
missing file leaves translation running without a glossary, with a warning.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from chatbot.rag.translator.base import Translator

log = logging.getLogger(__name__)

# Harakat, shadda, sukun, superscript alef, and tatweel: users rarely type
# them, and when they do they would break an exact match.
_MARKS = re.compile("[ً-ْٰـ]")
_HEBREW = re.compile("[֐-׿]")
_PREFIXES = "وفبلك"
# A Hebrew replacement can't carry an Arabic letter glued to its front, so each
# prefix is spelled out as a separate word instead.
_PREFIX_WORDS = {"و": "و", "ف": "ف", "ب": "في", "ل": "إلى", "ك": "مثل"}


def strip_marks(text: str) -> str:
    return _MARKS.sub("", text)


def default_glossary_path() -> Path:
    # Imported here: the pipeline module imports this package.
    from chatbot.rag.pipeline.pipeline import GLOSSARY_PATH

    return GLOSSARY_PATH


def load_glossary(path: str | Path | None = None) -> dict[str, str]:
    """``{arabic term: replacement}`` from a glossary CSV (BOM tolerated).

    Defaults to the static store's glossary. A missing file gives an empty
    glossary and a warning: translation still works, just without the fixes.
    """
    path = Path(path) if path is not None else default_glossary_path()
    if not path.is_file():
        log.warning("no dialect glossary at %s; translating without one", path)
        return {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        entries = {
            strip_marks(row["arabic"].strip()): row["replacement"].strip()
            for row in csv.DictReader(f)
            if row.get("arabic", "").strip() and row.get("replacement", "").strip()
        }
    return entries


def _with_lil_forms(entries: dict[str, str]) -> dict[str, str]:
    """Add the ل + ال contraction: للست is "for the woman", from the entry الست.

    Arabic drops the alef when ل precedes the article, so the prefix rule alone
    never sees "ل" + "الست". Each ال-entry gets a لل-form whose replacement
    carries the same "to/for": glued on for Arabic (للمرأة, لإطلاق النار),
    spelled out for Hebrew.
    """
    derived = {}
    for term, replacement in entries.items():
        if not term.startswith("ال"):
            continue
        if _HEBREW.search(replacement):
            lil = f"{_PREFIX_WORDS['ل']} {replacement}"
        elif replacement.startswith("ال"):
            lil = "لل" + replacement[2:]
        else:
            lil = "ل" + replacement
        derived.setdefault("لل" + term[2:], lil)
    return {**derived, **entries}


class Glossary:
    def __init__(self, entries: dict[str, str]):
        entries = _with_lil_forms(entries)
        self.entries = entries
        if entries:
            # Longest first, so "مركز الصمود" wins over a shorter entry inside it.
            terms = sorted(entries, key=len, reverse=True)
            alternatives = "|".join(re.escape(t) for t in terms)
            self._pattern = re.compile(
                rf"(?<!\w)([{_PREFIXES}]{{0,2}}?)({alternatives})(?!\w)"
            )
        else:
            self._pattern = None

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> Glossary:
        return cls(load_glossary(path))

    def apply(self, text: str) -> str:
        text = strip_marks(text)
        if self._pattern is None:
            return text
        return self._pattern.sub(self._replace, text)

    def _replace(self, match: re.Match) -> str:
        prefix, term = match.group(1), match.group(2)
        replacement = self.entries[term]
        if not prefix:
            return replacement
        if _HEBREW.search(replacement):
            words = " ".join(_PREFIX_WORDS[p] for p in prefix)
            return f"{words} {replacement}"
        return prefix + replacement


class GlossaryTranslator(Translator):
    """Applies a :class:`Glossary` to Arabic input, then calls the inner translator."""

    def __init__(self, inner: Translator, glossary: Glossary | None = None):
        self.inner = inner
        self.glossary = glossary or Glossary.from_csv()

    @property
    def name(self) -> str:
        return f"glossary+{self.inner.name}"

    def to_hebrew(self, text: str, source: str) -> str:
        if source.strip().lower() == "arabic":
            text = self.glossary.apply(text)
        return self.inner.to_hebrew(text, source)
