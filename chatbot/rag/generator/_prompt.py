"""Shared prompt scaffolding for the LLM generator heads.

Both back-ends (Claude and Gemini) answer the same way — a grounded, Hebrew
answer built only from the retrieved passages — so the system instruction and
the context formatting live here and each generator supplies only the transport.
Mirrors how ``data.enrich.gemini_enricher`` reuses the Claude enricher's prompt:
one spec, two transports.
"""

from __future__ import annotations

from collections.abc import Sequence

from chatbot.rag.documents import Document, format_citation


# Meta-instruction is English (matches the rest of the codebase's prompts) but
# pins the *answer* language and forbids ungrounded content — the retrieval
# corpus is the only source of truth for a rights/benefits answer. The answer
# language is a parameter (Hebrew by default) so a multilingual front-end can
# reply in the user's chosen language; the corpus and citations stay unchanged.
def build_system(language: str = "Hebrew") -> str:
    """The grounding system instruction, phrased to answer in ``language``."""
    return (
        "You are a helpful assistant for an Israeli volunteer organisation, "
        "answering questions about rights and benefits.\n"
        f"Answer the user's question in {language}, using only the information in "
        "the numbered sources provided. If the sources do not contain the answer, "
        f"say so plainly in {language} and do not invent details, eligibility "
        "rules, or numbers. Keep the answer concise, and cite the passage "
        "number(s) you relied on in square brackets, e.g. [2]. The user cannot "
        "see the sources themselves, so when a passage you relied on lists where "
        "its information comes from, point the user to that reference, including "
        "its link when one is given."
    )

# Returned directly (no API call) when retrieval produced nothing to ground on.
NO_CONTEXT = "לא נמצא מידע רלוונטי לשאלה במאגר."

# Labels a passage's reference line; Hebrew, like the rest of the user message.
_SOURCE_LABEL = "מקור"


def _format_passage(index: int, doc: Document) -> str:
    """``[index] text``, followed by the passage's references, one per line."""
    lines = [f"[{index}] {doc.text}"]
    lines.extend(
        f"{_SOURCE_LABEL}: {ref}" for ref in map(format_citation, doc.citations) if ref
    )
    return "\n".join(lines)


def build_user_message(query: str, contexts: Sequence[Document]) -> str:
    """Render the retrieved passages + the question into one user message.

    Passages are numbered from 1 so the model can cite them ([1], [2], …); each
    ``doc.text`` already carries its heading breadcrumb from the loader, and any
    ``doc.citations`` follow it as ``מקור:`` lines the answer can refer the user to.
    """
    passages = "\n\n".join(
        _format_passage(i, doc) for i, doc in enumerate(contexts, start=1)
    )
    return f"מקורות:\n{passages}\n\nשאלה: {query}"
