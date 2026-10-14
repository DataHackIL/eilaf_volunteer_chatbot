#!/usr/bin/env bash
# Container entrypoint: curate the corpus, then hand off to the CMD (the server).
#
# The corpus is bind-mounted from the host, not baked into the image, so a
# container can boot onto an empty or stale data/static/. Rebuilding it here —
# before uvicorn binds — means the webhook is never up while answering from a
# corpus that is still being written.
#
#   EILAF_CORPUS_S3_URI=s3://…   pull the corpus from S3 first (a no-op when
#                                the mounted copy already matches; see
#                                data/corpus_sync). Runs before curation.
#   SKIP_DATA_CURATION=1   skip the whole rebuild and start the server at once.
#
# Everything else is tunable through scripts/curate_data.sh's own environment
# variables (CURATE_SKIP_*, ENRICH_PROVIDER, ENRICH_LIMIT, …).
set -euo pipefail

if [ -n "${EILAF_CORPUS_S3_URI:-}" ]; then
    # Not fatal, for the same reason as curation below: an unreachable bucket
    # leaves the mounted corpus in place, and serving from it beats a
    # crash-loop. The server's own startup still fails loudly on an empty one.
    python -m data.corpus_sync || {
        echo "entrypoint: corpus pull from $EILAF_CORPUS_S3_URI failed —" \
             "starting on the corpus as mounted (see the log above)." >&2
    }
fi

if [ "${SKIP_DATA_CURATION:-0}" = "1" ]; then
    echo "entrypoint: SKIP_DATA_CURATION=1 — using the corpus as mounted."
else
    # Deliberately not fatal: a failed scrape or an exhausted LLM quota leaves
    # the previous corpus in place, and a bot answering from a slightly stale
    # corpus beats a container that crash-loops (re-scraping on every restart).
    # Set CURATE_STRICT=1 to make the failure loud *and* fatal instead.
    /app/scripts/curate_data.sh || {
        echo "entrypoint: data curation reported failures — starting the" \
             "server anyway on the existing corpus (see the log above)." >&2
    }
fi

exec "$@"
