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
# then edit .env: OPENROUTER_API_KEY (agent) and optional FIRMS_MAP_KEY
# FIRMS key: https://firms.modaps.eosdis.nasa.gov/api/map_key/ (free, emailed)

# 5. Install git hooks (lint, format, type-check, secret scan on commit)
uv run pre-commit install
```

## Run

```bash
uv run atlas
```

## Local tool-using agent interface

The initial agent interface operates on a caller-selected local artifact directory.
It has an explicit tool registry; a model may only use registered tools, and each
run is bounded by a maximum number of tool rounds. The built-in tools can list
and inspect local images, and generate a labeled contact sheet as a new local PNG.

```python
from pathlib import Path

from atlas.agent import Agent, LocalArtifactStore, default_registry
from atlas.agent.model import ModelConfig, OpenRouterModel

artifacts = LocalArtifactStore(Path(".atlas/my-session"))
model = OpenRouterModel(ModelConfig(model="openai/gpt-4.1-mini", api_key="..."))
agent = Agent(model, default_registry(), artifacts)
result = agent.run("List the local images and make a contact sheet.")
print(result.response)
print(result.steps)
```

`default_registry(allow_script_proposals=True)` also lets the agent *stage* a
Python helper under `proposals/` in the artifact directory, then use
`review_script_proposals` to create a static review report after the main
conversation. These drafts are not executed, loaded, or promoted automatically.
A reviewed implementation still needs to be added to application code and
registered explicitly. This keeps a self-improving workflow auditable without
granting the model arbitrary code execution or self-modification.

## Development

```bash
uv run pytest                       # run tests
uv run ruff check .                 # lint
uv run ruff format .                # format
uv run mypy src                     # type-check
uv run pre-commit run --all-files   # run all hooks on the whole repo
```
