"""Content-compared, download-only sync of an S3 prefix into a directory."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# The part size `aws s3 cp/sync` uploads with by default (multipart_chunksize).
# A multipart ETag is md5(concat(part md5s))-N, so reproducing it locally needs
# the part size; anything uploaded with another one simply reads as changed and
# is re-downloaded once — wasteful, never wrong.
_MULTIPART_CHUNK = 8 * 1024 * 1024


@dataclass
class PullResult:
    downloaded: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """``s3://bucket/some/prefix`` -> (``bucket``, ``some/prefix/``)."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, prefix = uri[len("s3://"):].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def _local_etag(path: Path, parts: int) -> str:
    """The ETag S3 would give ``path`` if uploaded in ``parts`` parts (1 = plain md5)."""
    with path.open("rb") as f:
        if parts <= 1:
            return hashlib.file_digest(f, "md5").hexdigest()
        digests = b""
        while chunk := f.read(_MULTIPART_CHUNK):
            digests += hashlib.md5(chunk).digest()
    return f"{hashlib.md5(digests).hexdigest()}-{parts}"


def is_current(path: Path, size: int, etag: str) -> bool:
    """True when ``path`` already holds the object S3 describes."""
    if not path.is_file() or path.stat().st_size != size:
        return False
    etag = etag.strip('"')
    parts = int(etag.rpartition("-")[2]) if "-" in etag else 1
    return _local_etag(path, parts) == etag


def pull_corpus(client, uri: str, dest: Path) -> PullResult:
    """Download every object under ``uri`` into ``dest`` that differs locally.

    Downloads land in a temp file and are renamed into place, so a failure
    midway never leaves a truncated JSON or .npy behind. New files take the
    owner of ``dest``: the container runs as root, and a root-owned file in the
    bind-mounted store is one the next deploy's rsync (as the SSH user) cannot
    touch.
    """
    bucket, prefix = parse_s3_uri(uri)
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    owner = dest.stat()
    result = PullResult()

    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel or rel.endswith("/"):
                continue  # the prefix itself, or a console-made "folder"
            target = (dest / rel).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"key escapes {dest}: {obj['Key']!r}")

            if is_current(target, obj["Size"], obj["ETag"]):
                result.unchanged.append(rel)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".pull-")
            os.close(fd)
            try:
                client.download_file(bucket, obj["Key"], tmp)
                os.replace(tmp, target)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise
            if os.geteuid() == 0:
                os.chown(target, owner.st_uid, owner.st_gid)
            result.downloaded.append(rel)

    if not result.downloaded and not result.unchanged:
        raise RuntimeError(f"nothing under {uri} — wrong URI, or no read access to it")
    return result
