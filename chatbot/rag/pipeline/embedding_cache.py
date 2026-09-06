"""Disk cache for corpus embeddings.

Embedding the whole corpus is the dominant startup cost (minutes on CPU), yet
the corpus is static between runs. Persist the embedding matrix and reuse it
whenever neither the corpus nor the embedder has changed; otherwise recompute
and overwrite.

Staleness is decided by a fingerprint with two independent parts:

- the corpus — a hash of the chunk texts (order included, since retrieval maps
  result indices back to documents by position);
- the embedder — :attr:`~chatbot.rag.embedder.base.Embedder.fingerprint`, which
  each implementation defines to capture its own identity (model name, and
  finetuned weights once those change under a fixed name).

A mismatch on either part — or a row count that disagrees with the corpus —
forces a recompute.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from chatbot.rag.embedder.base import Embedder

_MATRIX_FILE = "embeddings.npy"
_META_FILE = "embeddings.meta.json"


def _corpus_hash(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")  # delimiter so different splits can't alias
    return digest.hexdigest()


def _fingerprint(embedder: Embedder, texts: Sequence[str]) -> dict:
    return {
        "embedder": embedder.fingerprint,
        "corpus": _corpus_hash(texts),
        "count": len(texts),
    }


def load_or_encode(
    embedder: Embedder, texts: Sequence[str], cache_dir: str | Path
) -> np.ndarray:
    """Return embeddings for ``texts``, reusing the on-disk cache when valid.

    On a hit, loads ``embeddings.npy`` (~sub-second) instead of re-encoding. On
    a miss, encodes and writes ``embeddings.npy`` + ``embeddings.meta.json``
    under ``cache_dir``.
    """
    texts = list(texts)
    cache_dir = Path(cache_dir)
    matrix_path = cache_dir / _MATRIX_FILE
    meta_path = cache_dir / _META_FILE
    want = _fingerprint(embedder, texts)

    if matrix_path.exists() and meta_path.exists():
        try:
            have = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            have = None
        if have == want:
            matrix = np.load(matrix_path)
            if matrix.shape[0] == len(texts):  # guard a truncated/partial write
                return matrix

    matrix = embedder.encode_documents(texts)
    _try_save(cache_dir, matrix_path, meta_path, want, matrix)
    return matrix


def _try_save(
    cache_dir: Path,
    matrix_path: Path,
    meta_path: Path,
    want: dict,
    matrix: np.ndarray,
) -> None:
    """Persist the matrix + fingerprint, tolerating an unwritable cache dir.

    The cache is an optimisation, never a correctness requirement, so failing to
    write it must not take the caller down. The case this exists for is a
    deliberately read-only mount: containers share this directory with the host
    but pin their own copy of the code, and a container running an older loader
    would otherwise recompute a *different* chunking and overwrite a newer,
    correct cache — last writer wins, and the two versions ping-pong. Mounting
    read-only stops that; this keeps such a container running (recomputing in
    memory each start) instead of crashing on PermissionError.
    """
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(matrix_path, matrix)
        meta_path.write_text(
            json.dumps(want, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        print(
            f"embedding_cache: could not write {cache_dir} ({exc}); "
            "embeddings held in memory for this run only"
        )
