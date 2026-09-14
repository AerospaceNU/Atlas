# Install Atlas on macOS

These steps assume `git` is already on your `PATH`.

Python 3.12 is pinned in `.python-version`. `uv` will download it if needed.

## 1. Install tools

[uv](https://docs.astral.sh/uv/getting-started/installation/) (skip if `uv --version` already works):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new terminal, or `source` the env file the installer prints (`~/.local/bin/env` on a typical install).

[gitleaks](https://github.com/gitleaks/gitleaks) (required for the pre-commit secret scan):

```bash
brew install gitleaks
```

Homebrew: https://brew.sh

## 2. Clone and install the project

```bash
git clone https://github.com/AerospaceNU/Atlas.git
cd Atlas

uv python install 3.12
uv sync

cp .env.example .env
```

Edit `.env`:

- `OPENROUTER_API_KEY` — required to run the agent. Create a key at https://openrouter.ai/keys
- `FIRMS_MAP_KEY` — optional; required only for FIRMS fire sources. Free key from https://firms.modaps.eosdis.nasa.gov/api/map_key/

## 3. Install git hooks

```bash
uv run pre-commit install
```

Hooks run Ruff, format, mypy (on `src/`), and gitleaks on each commit.

## 4. Check the install

```bash
uv run atlas
uv run pytest
```

`uv run atlas` prints `Hello from atlas!`.

## Troubleshooting

- **`uv: command not found`** — open a new terminal, or add `~/.local/bin` to `PATH`.
- **`gitleaks` hook fails** — confirm `gitleaks version` works in the same shell you use for `git commit`.
- **pre-commit cannot install hook environments (401 / missing setuptools)** — pip is using a private index. Bootstrap once from PyPI:

  ```bash
  PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= uv run pre-commit run --all-files
  ```
