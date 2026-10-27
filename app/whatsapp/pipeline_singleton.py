"""Process-global RAG pipeline, built once and reused across every message.

The e5 embedder and the on-disk embedding cache (``data/static/.embeddings/``)
are loaded a single time on first access, then every WhatsApp turn, for every
user, reuses the same warm pipeline. The corpus is never re-embedded per message; the missing
per-user "session state" (which step a user is on) lives separately in
``conversation.py`` and never touches this singleton.

The answer engine is picked by ``EILAF_GENERATOR`` (``config.GENERATOR``):
Gemini rotation — free tier, model-rotating — unless it says otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from app.whatsapp import config
from chatbot.rag.generator import (
    BedrockGenerator,
    ClaudeGenerator,
    ContextEchoGenerator,
    Generator,
    OpenAICompatibleGenerator,
    gemini_rotation,
)
from chatbot.rag.pipeline import RAGPipeline

# One factory per name in ``config.ENGINE_KEYS``, which lists the keys each needs.
ENGINES: dict[str, Callable[[], Generator]] = {
    "gemini": gemini_rotation,
    "claude": ClaudeGenerator,
    "bedrock": BedrockGenerator,
    "cerebras": OpenAICompatibleGenerator,
    "echo": ContextEchoGenerator,
}


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Return the shared pipeline, building it on first call.

    ``from_static_dir`` hard-codes a ``ContextEchoGenerator`` (passing
    ``generator=`` collides with it), so we swap the generator afterwards — the
    head is a plain attribute, so swapping it costs nothing.
    """
    pipeline = RAGPipeline.from_static_dir()
    pipeline.generator = ENGINES[config.GENERATOR]()
    return pipeline
