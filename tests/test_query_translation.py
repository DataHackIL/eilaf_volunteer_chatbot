"""Tests for translate-then-retrieve: a non-Hebrew question is also retrieved in
Hebrew translation, and the two rankings are fused.

A keyword embedder makes retrieval deterministic: each document is about one
keyword, and a query scores highest against the documents whose keywords it
contains. A fake translator maps the "Arabic" question to a different keyword,
so each pass's winner is known in advance.
"""

from __future__ import annotations

import numpy as np

from chatbot.rag.documents.documents import Document
from chatbot.rag.embedder.base import Embedder
from chatbot.rag.generator.context_echo import ContextEchoGenerator
from chatbot.rag.pipeline.pipeline import DEFAULT_CONTEXT_ANCHOR, RAGPipeline
from chatbot.rag.reranker.base import Reranker
from chatbot.rag.translator import IdentityTranslator, Translator, is_hebrew

_VOCAB = ["alpha", "beta", "gamma", "delta"]
_DOCS = [Document(text=word, id=word, parent_id=None) for word in _VOCAB]


class _KeywordEmbedder(Embedder):
    def encode(self, texts):
        rows = []
        for text in texts:
            row = np.array([float(text.split().count(w)) for w in _VOCAB])
            rows.append(row + 1e-3)  # never all-zero, so cosine stays defined
        return np.array(rows, dtype=np.float32)


class _FakeTranslator(Translator):
    def __init__(self, table: dict[str, str] | None = None, fail: bool = False):
        self.table = table or {}
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def to_hebrew(self, text, source):
        self.calls.append((text, source))
        if self.fail:
            raise RuntimeError("backend down")
        return self.table.get(text, text)


class _RecordingReranker(Reranker):
    """Scores every candidate 0, so the order stays the embedder's; records queries."""

    def __init__(self):
        self.queries: list[str] = []

    def score(self, query, documents):
        self.queries.append(query)
        return np.zeros(len(documents))


def _pipeline(
    translator=None, reranker=None, top_k=1, original_weight=1.0
) -> RAGPipeline:
    return RAGPipeline(
        documents=_DOCS,
        embedder=_KeywordEmbedder(),
        generator=ContextEchoGenerator(),
        reranker=reranker,
        translator=translator,
        top_k=top_k,
        candidate_k=4,
        original_weight=original_weight,
    )


def _ids(docs):
    return [d.id for d in docs]


def test_is_hebrew():
    assert is_hebrew(None) and is_hebrew("") and is_hebrew("hebrew ")
    assert not is_hebrew("Arabic")


def test_hebrew_question_is_not_translated():
    translator = _FakeTranslator({"beta": "alpha"})
    pipe = _pipeline(translator)
    assert _ids(pipe.matches("beta", context=None, language="Hebrew")) == ["beta"]
    assert _ids(pipe.matches("beta", context=None)) == ["beta"]
    assert translator.calls == []


def test_no_translator_keeps_retrieval_unchanged():
    pipe = _pipeline(None)
    assert _ids(pipe.matches("beta", context=None, language="Arabic")) == ["beta"]


def test_fused_ranking_holds_both_passes():
    translator = _FakeTranslator({"beta": "alpha"})
    pipe = _pipeline(translator, top_k=2)
    ids = _ids(pipe.matches("beta", context=None, language="Arabic"))
    # alpha: the Hebrew translation's best match; beta: the original's.
    assert set(ids) == {"alpha", "beta"}
    assert translator.calls == [("beta", "Arabic")]


def test_weight_zero_ranks_on_the_translation_alone():
    original = "beta beta gamma"
    pipe = _pipeline(
        _FakeTranslator({original: "alpha delta"}), top_k=2, original_weight=0
    )
    fused = _ids(pipe.matches(original, context=None, language="Arabic"))
    assert fused == _ids(pipe.matches("alpha delta", context=None))


def test_a_lighter_original_pass_still_contributes():
    pipe = _pipeline(_FakeTranslator({"beta": "alpha"}), top_k=2, original_weight=0.5)
    assert set(_ids(pipe.matches("beta", context=None, language="Arabic"))) == {
        "alpha",
        "beta",
    }


def test_translated_pass_wins_a_tie():
    pipe = _pipeline(_FakeTranslator({"beta": "alpha"}), top_k=1)
    assert _ids(pipe.matches("beta", context=None, language="Arabic")) == ["alpha"]


def test_a_chunk_both_passes_rank_second_beats_each_pass_winner():
    # Hebrew pass: delta, gamma, alpha, beta. Original pass: beta, gamma, alpha,
    # delta. Neither puts gamma first, but 2/(60+2) beats 1/(60+1) + 1/(60+4).
    original = "beta beta beta gamma gamma alpha"
    translated = "delta delta delta gamma gamma alpha"
    pipe = _pipeline(_FakeTranslator({original: translated}), top_k=1)
    assert _ids(pipe.matches(original, context=None, language="Arabic")) == ["gamma"]


def test_translation_failure_falls_back_to_the_original():
    pipe = _pipeline(_FakeTranslator(fail=True))
    assert _ids(pipe.matches("beta", context=None, language="Arabic")) == ["beta"]


def test_reranker_only_sees_the_hebrew_query():
    reranker = _RecordingReranker()
    pipe = _pipeline(_FakeTranslator({"beta": "alpha"}), reranker=reranker)
    pipe.matches("beta", context=None, language="Arabic")
    assert reranker.queries == ["alpha"]


def test_translation_is_cached():
    translator = _FakeTranslator({"beta": "alpha"})
    pipe = _pipeline(translator)
    pipe.matches("beta", context=None, language="Arabic")
    pipe.retrieve("beta", context=None, language="Arabic")
    assert translator.calls == [("beta", "Arabic")]


def test_swapping_the_translator_drops_its_cached_translations():
    pipe = _pipeline(_FakeTranslator({"beta": "alpha"}))
    pipe.matches("beta", context=None, language="Arabic")
    second = _FakeTranslator({"beta": "gamma"})
    pipe.translator = second
    assert _ids(pipe.matches("beta", context=None, language="Arabic")) == ["gamma"]
    assert second.calls == [("beta", "Arabic")]


def test_user_context_is_translated_but_the_default_anchor_is_not():
    translator = _FakeTranslator({"beta": "alpha"})
    pipe = _pipeline(translator)
    pipe.matches("beta", context=DEFAULT_CONTEXT_ANCHOR, language="Arabic")
    assert translator.calls == [("beta", "Arabic")]
    pipe.matches("beta", context="delta", language="Arabic")
    assert ("delta", "Arabic") in translator.calls


def test_answer_retrieves_in_the_answer_language():
    translator = _FakeTranslator({"beta": "alpha"})
    pipe = _pipeline(translator)
    pipe.answer("beta", context=None, language="Arabic")
    assert translator.calls == [("beta", "Arabic")]


def test_identity_translator_reproduces_the_untranslated_matches():
    plain = _pipeline(None, top_k=2)
    same = _pipeline(IdentityTranslator(), top_k=2)
    query = "gamma delta"
    assert _ids(same.matches(query, context=None, language="Arabic")) == _ids(
        plain.matches(query, context=None, language="Arabic")
    )


def test_fuse_sums_reciprocal_ranks():
    fused = RAGPipeline._fuse([np.array([3, 1, 2]), np.array([1, 0])])
    # 1 is second in one list and first in the other, so it beats 3.
    assert list(fused[:2]) == [1, 3]
    assert set(fused) == {0, 1, 2, 3}


def test_hebrew_query_reports_what_was_searched():
    translator = _FakeTranslator({"beta": "alpha"})
    pipe = _pipeline(translator)
    assert pipe.hebrew_query("beta", "Hebrew") is None
    pipe.matches("beta", context=None, language="Arabic")
    assert pipe.hebrew_query("beta", "Arabic") == "alpha"
    assert translator.calls == [("beta", "Arabic")]  # served from the cache
    assert _pipeline(None).hebrew_query("beta", "Arabic") is None
    assert _pipeline(_FakeTranslator(fail=True)).hebrew_query("beta", "Arabic") is None
