"""The embedding cache must degrade, never crash, when it cannot be written.

Containers bind-mount the cache directory but pin their own copy of the code. A
container on an older loader computes a different chunking, so if it can write
it overwrites a newer, correct cache — and the two versions then ping-pong, each
start costing a full re-encode. Mounting the directory read-only stops that,
which only works if an unwritable cache leaves the app running.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from chatbot.rag.embedder.base import Embedder
from chatbot.rag.pipeline.embedding_cache import load_or_encode

TEXTS = ["alpha", "beta", "gamma"]


class StubEmbedder(Embedder):
    """Deterministic embeddings, counting how often encoding actually ran."""

    fingerprint = "stub:v1"

    def __init__(self):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    def encode_documents(self, texts):
        return self.encode(texts)

    def encode_queries(self, texts):
        return self.encode(texts)


def test_writes_and_then_reuses_the_cache(tmp_path):
    embedder = StubEmbedder()
    first = load_or_encode(embedder, TEXTS, tmp_path)
    second = load_or_encode(embedder, TEXTS, tmp_path)
    assert embedder.calls == 1  # second call served from disk
    assert np.array_equal(first, second)
    assert json.loads((tmp_path / "embeddings.meta.json").read_text())["count"] == 3


def _make_readonly(path):
    """Approximate a Docker ``:ro`` bind mount.

    A read-only *mount* refuses every write. Directory permissions alone do not:
    on Linux they gate creating and deleting entries, while overwriting an
    existing file needs write permission on the file itself — so the cache files
    must be locked down too, or an overwrite still succeeds.
    """
    for child in path.iterdir():
        child.chmod(0o400)
    path.chmod(0o500)


def _restore(path):
    path.chmod(0o700)
    for child in path.iterdir():
        child.chmod(0o600)


def test_readonly_dir_still_returns_embeddings(tmp_path):
    tmp_path.chmod(0o500)  # empty dir: nothing can be created in it
    try:
        matrix = load_or_encode(StubEmbedder(), TEXTS, tmp_path)
    finally:
        tmp_path.chmod(0o700)
    assert matrix.shape == (3, 2)
    assert not (tmp_path / "embeddings.npy").exists()


def test_readonly_dir_never_clobbers_a_newer_cache(tmp_path):
    # A newer cache written by the host…
    load_or_encode(StubEmbedder(), TEXTS, tmp_path)
    before = (tmp_path / "embeddings.npy").read_bytes()
    meta_before = (tmp_path / "embeddings.meta.json").read_text()

    # …survives a read-only reader that wants a *different* corpus.
    _make_readonly(tmp_path)
    try:
        stale = load_or_encode(StubEmbedder(), ["alpha", "beta"], tmp_path)
    finally:
        _restore(tmp_path)

    assert stale.shape == (2, 2)  # it still got usable embeddings
    assert (tmp_path / "embeddings.npy").read_bytes() == before
    assert (tmp_path / "embeddings.meta.json").read_text() == meta_before


@pytest.mark.parametrize("corrupt", ["not json", ""])
def test_unreadable_meta_falls_back_to_encoding(tmp_path, corrupt):
    load_or_encode(StubEmbedder(), TEXTS, tmp_path)
    (tmp_path / "embeddings.meta.json").write_text(corrupt)
    embedder = StubEmbedder()
    assert load_or_encode(embedder, TEXTS, tmp_path).shape == (3, 2)
    assert embedder.calls == 1
