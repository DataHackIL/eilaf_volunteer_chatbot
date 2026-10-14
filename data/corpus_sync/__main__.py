"""CLI: ``python -m data.corpus_sync [S3_URI] [--dest DIR]``.

Both default from the environment the container already has:
``EILAF_CORPUS_S3_URI`` for the source and ``EILAF_STATIC_DIR`` (else the
repo's ``data/static``) for the destination — the same root the pipeline reads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3

from data.corpus_sync.pull import pull_corpus

_DEFAULT_STATIC = Path(__file__).resolve().parents[1] / "static"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m data.corpus_sync", description=__doc__)
    parser.add_argument("uri", nargs="?", default=os.environ.get("EILAF_CORPUS_S3_URI"))
    parser.add_argument(
        "--dest", type=Path, default=Path(os.environ.get("EILAF_STATIC_DIR") or _DEFAULT_STATIC)
    )
    args = parser.parse_args(argv)
    if not args.uri:
        parser.error("no S3 URI given and EILAF_CORPUS_S3_URI is unset")

    # Explicit region, as with the reports bucket: inside the container botocore
    # does not read it from instance metadata and falls back to us-east-1, which
    # against an eu-north-1 bucket fails as PermanentRedirect. The corpus shares
    # the reports bucket, so its region is the default.
    region = os.environ.get("EILAF_CORPUS_REGION") or os.environ.get("EILAF_REPORT_REGION") or None
    client = boto3.client("s3", region_name=region)

    result = pull_corpus(client, args.uri, args.dest)
    print(
        f"corpus_sync: {args.uri} -> {args.dest}: "
        f"{len(result.downloaded)} downloaded, {len(result.unchanged)} already current"
    )
    for rel in result.downloaded:
        print(f"corpus_sync:   + {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
