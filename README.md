# eilaf_volunteer_chatbot

RAG-based volunteer chatbot for Eilaf, with scraping, embedding, and
speech-to-text units.

## Install / create the env

The project is installable and locked with [uv](https://docs.astral.sh/uv/).
Create the environment from `uv.lock` with a single command.

The env is placed under `$ENVS_DIR` if that variable is set, otherwise in the
project directory (`./eilaf`):

```bash
UV_PROJECT_ENVIRONMENT="${ENVS_DIR:-.}/eilaf" uv sync
```

`uv` reads the venv location from the `UV_PROJECT_ENVIRONMENT` environment
variable (it can't be a conditional path in `pyproject.toml`), and
`${ENVS_DIR:-.}` expands to `$ENVS_DIR` when set and `.` otherwise.

> **GPU note:** the lock pins the default PyPI `torch`. On a CUDA machine,
> add the appropriate `[tool.uv.sources]` / index for the `cuXXX` wheels
> before syncing so you don't overwrite an existing GPU stack.
