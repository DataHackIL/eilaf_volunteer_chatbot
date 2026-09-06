#!/usr/bin/env bash
# Rebuild the corpus end to end: scrape → enrich → embed.
#
#   ./scripts/curate_data.sh
#
# The three stages are the manual steps documented in the README, chained in the
# only order they work in: scrapers write data/static/raw/, enrichment reads raw
# and writes data/static/enriched/, and the embed stage encodes enriched/ into
# data/static/enriched/.embeddings/ so the app's first query doesn't have to.
# Every stage is re-runnable and incremental, so running this repeatedly is
# cheap once the corpus is built.
#
# Knobs (all optional, all environment variables):
#   CURATE_SKIP_SCRAPE=1   skip a stage; the other two still run
#   CURATE_SKIP_ENRICH=1
#   CURATE_SKIP_EMBED=1
#   CURATE_SCRAPERS="..."  space-separated data.scraping modules to run
#                          (default: nevo kol_zchut gov_il — docx_source is
#                          excluded because it globs a private local folder)
#   ENRICH_PROVIDER=...    claude | gemini | cerebras (default: gemini, the
#                          working free back-end)
#   ENRICH_LIMIT=N         cap the number of segments sent to the LLM this run;
#                          unset means the full pending backlog, which can take
#                          hours. Nothing is lost between runs, so a cap is
#                          just a smaller bite.
#   ENRICH_ARGS="..."      extra flags passed straight to `python -m data.enrich`
#                          (e.g. --no-llm for a rules-only pass, no key needed)
#   PYTHON=...             interpreter to use (default: python)
#   CURATE_STRICT=1        exit non-zero on the FIRST stage failure instead of
#                          running the rest and reporting at the end
#
# Failures are non-fatal by default: a refused scrape or an exhausted LLM quota
# should still leave the embed stage free to publish whatever corpus exists.
# The script exits non-zero if anything failed, so a caller can still tell.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
CURATE_SCRAPERS="${CURATE_SCRAPERS:-nevo kol_zchut gov_il}"
ENRICH_PROVIDER="${ENRICH_PROVIDER:-gemini}"
FAILED=()

log() { printf '\n=== curate_data: %s ===\n' "$*"; }

# Resolved from EILAF_STATIC_DIR when set, else repo_root/data/static. Printed
# because every stage below writes there, and pointing it at the wrong mount is
# the failure that otherwise only shows up as a bot answering nothing.
log "static store: $("$PYTHON" -c \
    'from chatbot.rag.pipeline.pipeline import _STATIC_DIR; print(_STATIC_DIR)' \
    2>/dev/null || echo "<could not resolve — is the package importable?>")"

# Run a stage, recording (but not propagating) its failure — unless CURATE_STRICT.
run_stage() {
    local name="$1"
    shift
    if "$@"; then
        return 0
    fi
    echo "curate_data: STAGE FAILED: ${name}" >&2
    FAILED+=("$name")
    if [ "${CURATE_STRICT:-0}" = "1" ]; then
        echo "curate_data: CURATE_STRICT=1 — aborting." >&2
        exit 1
    fi
    return 1
}

if [ "${CURATE_SKIP_SCRAPE:-0}" = "1" ]; then
    log "scrape: skipped (CURATE_SKIP_SCRAPE=1)"
else
    for module in $CURATE_SCRAPERS; do
        log "scrape: data.scraping.${module}"
        # Each scraper module carries its own source-URL list in its __main__
        # block; add sources there, not here.
        run_stage "scrape:${module}" "$PYTHON" -m "data.scraping.${module}"
    done
fi

if [ "${CURATE_SKIP_ENRICH:-0}" = "1" ]; then
    log "enrich: skipped (CURATE_SKIP_ENRICH=1)"
else
    enrich_args=(--provider "$ENRICH_PROVIDER")
    [ -n "${ENRICH_LIMIT:-}" ] && enrich_args+=(--limit "$ENRICH_LIMIT")
    # Word-split on purpose: ENRICH_ARGS is a flag string, not one argument.
    # shellcheck disable=SC2206
    [ -n "${ENRICH_ARGS:-}" ] && enrich_args+=(${ENRICH_ARGS})
    log "enrich: ${enrich_args[*]}"
    run_stage "enrich" "$PYTHON" -m data.enrich "${enrich_args[@]}"
fi

if [ "${CURATE_SKIP_EMBED:-0}" = "1" ]; then
    log "embed: skipped (CURATE_SKIP_EMBED=1)"
else
    # No-op (sub-second) when neither the corpus nor the embedder changed;
    # a full re-encode — ~1h on CPU for the current corpus — when they did.
    log "embed: warming the embedding cache"
    run_stage "embed" "$PYTHON" -m chatbot.rag.pipeline.warm_cache
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo >&2
    echo "curate_data: FINISHED WITH FAILURES: ${FAILED[*]}" >&2
    exit 1
fi
log "done"
