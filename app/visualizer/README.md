# Streamlit visualizer

A developer tool for poking at the RAG pipeline: pick a language, pick an
answer engine, tune the retrieval knobs, ask a question.

**This lives only on the `dev/visualizer` branch.** `develop` and `main` ship
just the WhatsApp bot. Keep this branch current by merging `develop` *into* it;
never merge it back.

The visualizer (`app/visualizer/streamlit_app.py`) is the chat UI: pick a
language, pick an answer engine, ask a question. Either way of running it
serves on <http://localhost:8501>.

Build the corpus first (see the root README) — with an empty `data/static/enriched/`
the app runs but answers nothing.

## Secrets

The answer engines read their keys from the repo-root `.env`; see
"Answer-engine keys" in the root README. The default `echo` engine needs none.

## With uv

Run from the repo root (the script path is relative; the corpus dir is resolved
from the package itself, so it is found either way):

Streamlit is not a project dependency, so pull it in alongside the env:

```bash
UV_PROJECT_ENVIRONMENT="${ENVS_DIR:-.}/eilaf" uv run --with streamlit==1.59.2 streamlit run app/visualizer/streamlit_app.py
```

Or install it into the env once (`uv pip install streamlit==1.59.2`), activate
it (`source "${ENVS_DIR:-.}/eilaf/bin/activate"`) and just
`streamlit run app/visualizer/streamlit_app.py`.

## With Docker

`app/visualizer/docker-compose.yml` builds the image (context: repo root),
injects the repo-root `.env`, and publishes port 8501:

```bash
cd app/visualizer && docker compose up --build
```

The compose file bind-mounts `data/static/` (corpus + embedding cache)
**read-only** and keeps the e5 weights in a named volume, so the model downloads
once and survives restarts. The image installs the CPU build of torch regardless
of the committed CUDA lock, and adds Streamlit on top — see the comments in
`app/visualizer/Dockerfile`.

> **Why read-only.** The image pins its own copy of the code. A container built
> before a chunking change computes a different corpus, misses the cache, and —
> if it could write — would overwrite the host's newer one; the two versions
> then ping-pong, each start paying a full re-encode. This actually happened on
> 2026-08-19. Read-only means such a container recomputes in memory instead
> (slow start, nothing corrupted). **Rebuild the image after any change to
> chunking or the embedder**, or every container start pays that cost.

To build/run without compose:

```bash
docker build -f app/visualizer/Dockerfile -t eilaf-visualizer .   # from the repo root
docker run --rm -p 8501:8501 --env-file .env \
  -v "$PWD/data/static:/app/data/static:ro" eilaf-visualizer
```

## Cold-start times

The pipeline is built once per Streamlit session, on the first query — the page
itself renders immediately.

| | Time | Notes |
| --- | --- | --- |
| Server up (page served) | **~3 s** | identical for `uv` and Docker |
| First query, warm caches | **~7 s** | model load + BM25 fit ≈6 s, query ≈1 s (4790 chunks) |
| Later queries | **<1 s** | pipeline cached by `@st.cache_resource` |
| Docker image build | **~2 min** | ~35 s of that is the CPU-torch `uv sync`; longer on a first build (base images) |
| First-ever container boot | **+ model download** | ~2.2 GB of e5 weights into the `hf-cache` volume |
| Cold embedding cache | **~1 h on CPU** | only if `data/static/enriched/.embeddings` is stale or missing |

That last row is the one to watch: without a valid cache the whole corpus is
re-encoded before the first answer (≈0.8 s/chunk on CPU for
`multilingual-e5-large`, extrapolated from a 64-chunk sample; a GPU or
`multilingual-e5-base` cuts it sharply). The cache is fingerprinted by corpus
*and* embedder, so re-running enrichment or swapping the model invalidates it —
budget an hour the next time you rebuild the corpus.
