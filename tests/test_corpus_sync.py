"""The boot-time corpus pull must download only what differs, and be safe midway.

It runs on every container start, so "already current" has to be decided by
content — the deploy rsync resets mtimes, and a timestamp compare would turn
each deploy into a full re-download.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from data.corpus_sync import pull as pull_mod
from data.corpus_sync.pull import is_current, parse_s3_uri, pull_corpus

BUCKET, PREFIX = "bkt", "model/data/"


def _etag(body: bytes, chunk: int | None = None) -> str:
    if chunk is None or len(body) <= chunk:
        return f'"{hashlib.md5(body).hexdigest()}"'
    parts = [body[i : i + chunk] for i in range(0, len(body), chunk)]
    digest = hashlib.md5(b"".join(hashlib.md5(p).digest() for p in parts)).hexdigest()
    return f'"{digest}-{len(parts)}"'


class FakeS3:
    """list_objects_v2 paginator + download_file over an in-memory bucket."""

    def __init__(self, objects: dict[str, bytes], chunk: int | None = None):
        self.objects = objects
        self.chunk = chunk
        self.downloads: list[str] = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix):
        assert Bucket == BUCKET
        yield {
            "Contents": [
                {"Key": k, "Size": len(v), "ETag": _etag(v, self.chunk)}
                for k, v in self.objects.items()
                if k.startswith(Prefix)
            ]
        }

    def download_file(self, bucket, key, filename):
        self.downloads.append(key)
        Path(filename).write_bytes(self.objects[key])


URI = f"s3://{BUCKET}/{PREFIX}"
OBJECTS = {
    PREFIX + "raw/a.json": b'{"a": 1}',
    PREFIX + "enriched/a.json": b'{"a": 2}',
    PREFIX + ".embeddings/embeddings.npy": b"\x00" * 100,
}


def test_first_pull_downloads_everything_then_nothing(tmp_path):
    s3 = FakeS3(OBJECTS)
    first = pull_corpus(s3, URI, tmp_path)
    assert sorted(first.downloaded) == [
        ".embeddings/embeddings.npy",
        "enriched/a.json",
        "raw/a.json",
    ]
    assert (tmp_path / "raw/a.json").read_bytes() == b'{"a": 1}'

    s3.downloads.clear()
    second = pull_corpus(s3, URI, tmp_path)
    assert s3.downloads == []
    assert len(second.unchanged) == 3


def test_changed_content_same_size_is_redownloaded(tmp_path):
    s3 = FakeS3(OBJECTS)
    pull_corpus(s3, URI, tmp_path)
    (tmp_path / "raw/a.json").write_bytes(b'{"a": 9}')  # same size, new content
    s3.downloads.clear()
    pull_corpus(s3, URI, tmp_path)
    assert s3.downloads == [PREFIX + "raw/a.json"]
    assert (tmp_path / "raw/a.json").read_bytes() == b'{"a": 1}'


def test_local_only_files_are_kept(tmp_path):
    (tmp_path / "enriched").mkdir()
    (tmp_path / "enriched/curated_on_box.json").write_text("{}")
    pull_corpus(FakeS3(OBJECTS), URI, tmp_path)
    assert (tmp_path / "enriched/curated_on_box.json").exists()


def test_multipart_etag_is_recognised(tmp_path, monkeypatch):
    monkeypatch.setattr(pull_mod, "_MULTIPART_CHUNK", 16)
    body = bytes(range(50))  # 4 parts of 16
    path = tmp_path / "x.npy"
    path.write_bytes(body)
    assert is_current(path, len(body), _etag(body, chunk=16))
    assert not is_current(path, len(body), _etag(body[::-1], chunk=16))


def test_failed_download_leaves_old_file_and_no_temp(tmp_path):
    s3 = FakeS3(OBJECTS)
    pull_corpus(s3, URI, tmp_path)
    (tmp_path / "raw/a.json").write_bytes(b"stale!!!")

    def boom(bucket, key, filename):
        Path(filename).write_bytes(b"{trunc")
        raise ConnectionError("dropped")

    s3.download_file = boom
    with pytest.raises(ConnectionError):
        pull_corpus(s3, URI, tmp_path)
    assert (tmp_path / "raw/a.json").read_bytes() == b"stale!!!"
    assert not list((tmp_path / "raw").glob(".pull-*"))


def test_empty_prefix_is_an_error(tmp_path):
    with pytest.raises(RuntimeError, match="nothing under"):
        pull_corpus(FakeS3({}), URI, tmp_path)


def test_key_escaping_dest_is_refused(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        pull_corpus(FakeS3({PREFIX + "../evil.json": b"x"}), URI, tmp_path / "store")


@pytest.mark.parametrize(
    "uri, expected",
    [
        ("s3://b/model/data/", ("b", "model/data/")),
        ("s3://b/model/data", ("b", "model/data/")),
        ("s3://b", ("b", "")),
    ],
)
def test_parse_s3_uri(uri, expected):
    assert parse_s3_uri(uri) == expected
