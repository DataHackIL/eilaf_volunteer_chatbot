"""Process-global RAG pipeline, built once and reused across every message.

The e5 embedder and the on-disk embedding cache (``data/static/.embeddings/``)
are loaded a single time on first access — mirroring Streamlit's
``@st.cache_resource`` — then every WhatsApp turn, for every user, reuses the
same warm pipeline. The corpus is never re-embedded per message; the missing
per-user "session state" (which step a user is on) lives separately in
``conversation.py`` and never touches this singleton.

The generator head is Gemini rotation (free tier, model-rotating) per the
project decision; swap it here to change the answer engine globally.
"""

from __future__ import annotations

from functools import lru_cache

from chatbot.rag.generator import gemini_rotation
from chatbot.rag.pipeline import RAGPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """Return the shared pipeline, building it on first call.

    ``from_static_dir`` hard-codes a ``ContextEchoGenerator`` (passing
    ``generator=`` collides with it), so we swap the generator afterwards — the
    same pattern the Streamlit app uses when the user picks an answer engine.
    """
    pipeline = RAGPipeline.from_static_dir()
    pipeline.generator = gemini_rotation()
    return pipeline
