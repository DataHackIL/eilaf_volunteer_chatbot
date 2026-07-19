"""Generator that falls back across a chain of back-ends on rate-limit errors.

Wraps an ordered list of generators: it returns the first one's answer, but if a
back-end is rate-limited / quota-exhausted it advances to the next. Any *other*
error propagates immediately — a real bug shouldn't be silently swallowed as a
fall-over.

The immediate use is spreading load so no single free-tier limit gates every
request. For Gemini specifically, each model has its **own** per-day free quota
(the cap is per-project-per-model), so rotating models multiplies the free daily
budget without a second provider — see :func:`gemini_rotation`.
"""

from __future__ import annotations

from collections.abc import Sequence

from chatbot.rag.documents import Document
from chatbot.rag.generator.base import Generator


def is_rate_limited(exc: Exception) -> bool:
    """Heuristic: does ``exc`` look like a 429 / quota-exhausted error?

    Recognises the shapes the free-tier SDKs raise *without* importing them: a
    numeric 429 on ``code``/``status_code`` (google-genai ``ClientError``,
    openai ``RateLimitError``), or a tell-tale class name / message. Deliberately
    broad on the text so a newly added back-end's 429 still triggers fail-over.
    """
    if getattr(exc, "code", None) == 429 or getattr(exc, "status_code", None) == 429:
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("429", "rate limit", "ratelimit", "resource_exhausted", "quota")
    )


class FallbackGenerator(Generator):
    """Try each generator in order, advancing to the next on a rate-limit error.

    ``last_used`` holds the index of the generator that produced the last answer
    (handy for a UI / cost read-out); ``None`` until the first successful call.
    """

    def __init__(self, generators: Sequence[Generator]):
        if not generators:
            raise ValueError("FallbackGenerator needs at least one generator")
        self.generators = list(generators)
        self.last_used: int | None = None

    def generate(self, query: str, contexts: Sequence[Document]) -> str:
        last = len(self.generators) - 1
        for i, generator in enumerate(self.generators):
            try:
                answer = generator.generate(query, contexts)
            except Exception as exc:
                # Non-rate-limit errors, and a rate-limit on the final back-end,
                # propagate; otherwise move on to the next back-end.
                if i == last or not is_rate_limited(exc):
                    raise
                print(f"  fallback: generator #{i} rate-limited, trying #{i + 1}")
                continue
            self.last_used = i
            return answer
        raise AssertionError("unreachable")  # loop always returns or raises
