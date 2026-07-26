"""Shared prompt scaffolding for the LLM generator heads.

Both back-ends (Claude and Gemini) answer the same way — a grounded, Hebrew
answer built only from the retrieved passages — so the system instruction and
the context formatting live here and each generator supplies only the transport.
Mirrors how ``data.enrich.gemini_enricher`` reuses the Claude enricher's prompt:
one spec, two transports.
"""

from __future__ import annotations

from collections.abc import Sequence

from chatbot.rag.documents import Document


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
        "number(s) you relied on in square brackets, e.g. [2]."
    )

# Returned directly (no API call) when retrieval produced nothing to ground on.
NO_CONTEXT = "לא נמצא מידע רלוונטי לשאלה במאגר."


def build_user_message(query: str, contexts: Sequence[Document]) -> str:
    """Render the retrieved passages + the question into one user message.

    Passages are numbered from 1 so the model can cite them ([1], [2], …); each
    ``doc.text`` already carries its heading breadcrumb from the loader.
    """
    passages = "\n\n".join(f"[{i}] {doc.text}" for i, doc in enumerate(contexts, start=1))
    return f"מקורות:\n{passages}\n\nשאלה: {query}"
