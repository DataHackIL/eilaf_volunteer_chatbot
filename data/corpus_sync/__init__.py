"""Pull the corpus store (raw/, enriched/, .embeddings/) down from S3.

The deployed container boots onto whatever ``data/static`` holds; this fills it
from the bucket first, so the box answers from the copy the dev machine pushed
with ``aws s3 sync`` rather than from whatever the last deploy rsync'd there.

    python -m data.corpus_sync s3://eilaf-dev-chatbot-logs/model/data/

It is a no-op when the local copy already matches: files are compared by
content (size, then the S3 ETag), not by timestamp, so a deploy rsync that
resets mtimes does not trigger a re-download on the next boot. Download-only and
additive — a local file absent from the bucket is left alone, matching the
deploy rsync, because boot-time curation writes files the bucket never saw.
"""

from data.corpus_sync.pull import PullResult, pull_corpus

__all__ = ["PullResult", "pull_corpus"]
