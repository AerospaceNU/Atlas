# ATLAS

Local AI agent deployment.

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (package & environment manager)
- [gitleaks](https://github.com/gitleaks/gitleaks) — `brew install gitleaks` (for the pre-commit secret scan)

## Setup

```bash
# 1. Clone
cd Atlas

# 2. Install uv if you don't have it (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies (creates .venv automatically, including dev tools)
uv sync

# 4. Configure environment
cp .env.example .env
# then edit .env and set your OPENROUTER_API_KEY

# 5. Install git hooks (lint, format, type-check, secret scan on commit)
uv run pre-commit install
```

## Run

```bash
uv run atlas
```

## Development

```bash
uv run pytest                       # run tests
uv run ruff check .                 # lint
uv run ruff format .                # format
uv run mypy src                     # type-check
uv run pre-commit run --all-files   # run all hooks on the whole repo
```
