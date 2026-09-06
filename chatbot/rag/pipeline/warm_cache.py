"""Build the corpus embedding cache ahead of time: ``python -m chatbot.rag.pipeline.warm_cache``.

Constructing a pipeline embeds the whole corpus and persists the result under
``<static_dir>/.embeddings`` (see
:mod:`chatbot.rag.pipeline.embedding_cache`), so simply building one — and
throwing it away — is what "warm the cache" means here. The point is to pay the
CPU-bound encode cost once, at deploy time, instead of on the first user
message: a cold cache is roughly an hour of encoding on CPU, which a WhatsApp
webhook cannot sit through.

The cache is fingerprinted by corpus *and* embedder, so this is a no-op (a
sub-second load) whenever neither has changed — safe to run on every boot.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from chatbot.rag.pipeline.pipeline import DEFAULT_STATIC_DIR, RAGPipeline


def main(static_dir: str | Path = DEFAULT_STATIC_DIR) -> int:
    """Encode the corpus into the on-disk cache; return a process exit code."""
    static_dir = Path(static_dir)
    started = time.monotonic()
    pipeline = RAGPipeline.from_static_dir(static_dir)
    elapsed = time.monotonic() - started

    if not pipeline.documents:
        # Not an error: a fresh clone with no corpus yet still starts (and
        # answers nothing), so warming is a no-op rather than a boot blocker.
        print(f"warm_cache: no documents in {static_dir}; nothing to embed")
        return 0
    # Deliberately vague about which happened: the cache layer doesn't report
    # hit vs miss, and the elapsed time says it plainly enough (seconds = hit).
    print(
        f"warm_cache: corpus ready — {len(pipeline.documents)} chunks in "
        f"{elapsed:.1f}s (cache: {static_dir / '.embeddings'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
