"""Process-global RAG pipeline, built once and reused across every message.

The e5 embedder and the on-disk embedding cache (``data/static/.embeddings/``)
are loaded a single time on first access, then every WhatsApp turn, for every
user, reuses the same warm pipeline. The corpus is never re-embedded per message; the missing
per-user "session state" (which step a user is on) lives separately in
``conversation.py`` and never touches this singleton.

The answer engine is picked by ``EILAF_GENERATOR`` (``config.GENERATOR``):
Gemini rotation — free tier, model-rotating — unless it says otherwise. The
query translator, if any, is picked by ``EILAF_TRANSLATOR``.
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
from chatbot.rag.translator import (
    GlossaryTranslator,
    LLMTranslator,
    NLLBTranslator,
    OpusMTTranslator,
    Translator,
)

# One factory per name in ``config.ENGINE_KEYS``, which lists the keys each needs.
ENGINES: dict[str, Callable[[], Generator]] = {
    "gemini": gemini_rotation,
    "claude": ClaudeGenerator,
    "bedrock": BedrockGenerator,
    "cerebras": OpenAICompatibleGenerator,
    "echo": ContextEchoGenerator,
}


def build_translator() -> Translator | None:
    """The query translator ``EILAF_TRANSLATOR`` names, or None when it's blank.

    Every translator gets the Arabic dialect glossary in front of it, read once
    from the static store (``data/static/glossary_ar.csv``). The boot check
    (``config.wiring_problems``) has already vetted the settings.
    """
    inner = _build_inner_translator()
    return GlossaryTranslator(inner) if inner is not None else None


def _build_inner_translator() -> Translator | None:
    if config.TRANSLATOR == "nllb":
        return NLLBTranslator(config.TRANSLATOR_MODEL)
    if config.TRANSLATOR == "opus":
        if config.TRANSLATOR_MODEL:
            return OpusMTTranslator({"arabic": config.TRANSLATOR_MODEL})
        return OpusMTTranslator()
    if config.TRANSLATOR == "llm":
        return LLMTranslator(
            config.TRANSLATOR_URL,
            config.TRANSLATOR_MODEL,
            api_key=config.TRANSLATOR_API_KEY or None,
        )
    return None


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Return the shared pipeline, building it on first call.

    ``from_static_dir`` hard-codes a ``ContextEchoGenerator`` (passing
    ``generator=`` collides with it), so we swap the generator afterwards — the
    head is a plain attribute, so swapping it costs nothing.
    """
    pipeline = RAGPipeline.from_static_dir(translator=build_translator())
    pipeline.generator = ENGINES[config.GENERATOR]()
    return pipeline
