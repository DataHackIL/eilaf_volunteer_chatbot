# WhatsApp front-end image. Build context is the repo root:
#   docker build -f app/whatsapp/Dockerfile -t eilaf-whatsapp .
# (or via app/whatsapp/docker-compose.yml, which sets the context for you).
#
# Secrets are NEVER baked in — they are injected at run time with `--env-file .env`.
# Dev choice: the e5 model + embedding cache are NOT baked; the model downloads
# on first boot into the mounted HF cache volume, and data/static/ is bind-mounted
# in (see docker-compose.yml). Baking them for a release image is a later step.
FROM python:3.13-slim

# uv for fast, reproducible installs (pinned copy from the official image).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/models \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install dependencies first (their own layer) so edits to source don't bust the
# heavy torch/transformers install cache. Only the manifests are needed here.
COPY pyproject.toml uv.lock README.md ./
# The committed uv.lock pins a CUDA torch (the dev machine has a GPU), which would
# drag in ~7GB of unused CUDA libraries. Point torch at PyTorch's CPU index via a
# Docker-only override: a `[tool.uv.sources]` entry only remaps *direct* deps, and
# torch here is transitive (via sentence-transformers), so we also `uv add` it as
# a direct dependency. Re-locking then resolves torch==*+cpu (0 nvidia packages).
# This rewrites pyproject.toml + uv.lock *inside the image only*; the repo is
# untouched (its CUDA lock keeps working on the GPU dev machine).
RUN printf '\n[[tool.uv.index]]\nname = "pytorch-cpu"\nurl = "https://download.pytorch.org/whl/cpu"\nexplicit = true\n\n[tool.uv.sources]\ntorch = [{ index = "pytorch-cpu" }]\n' >> pyproject.toml \
    && uv add --no-sync torch
RUN uv sync --no-install-project --no-dev

# Copy only the source packages — NOT the manifests, which would revert the CPU
# override/lock above. (data/static is bind-mounted at runtime; see compose.)
# ``data`` carries the scrapers and the enrichment stage, which the entrypoint
# runs before the server; ``scripts`` carries the chain that drives them.
COPY app ./app
COPY chatbot ./chatbot
COPY data ./data
COPY scripts ./scripts
RUN uv sync --no-dev

EXPOSE 8000

# The entrypoint rebuilds the corpus (scrape → enrich → embed) and then execs
# the CMD below. Set SKIP_DATA_CURATION=1 to boot straight into the server on
# the corpus as mounted — which is what docker-compose.yml does by default.
ENTRYPOINT ["/app/app/whatsapp/entrypoint.sh"]

# Single uvicorn worker so the warm pipeline is loaded once and shared.
CMD ["python", "-m", "app.whatsapp.server"]
