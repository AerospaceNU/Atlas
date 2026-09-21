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

Rust, for the agent TUI. `atlas data` does not need it. [rustup](https://rustup.rs) installs `cargo` into `~/.cargo/bin`. Atlas looks there even when that directory is not on `PATH`:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

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

Put `atlas` on `PATH`, pointed at this checkout:

```bash
uv tool install --editable --force .
```

Source edits in this repo are picked up without reinstalling. Run that command again after dependency changes. `~/.local/bin` must be on `PATH`. A normal uv install already adds it.

## 3. Install git hooks

Install hooks once after cloning so lint, format, type-check, tests, and secret
scans run on every commit:

```bash
uv run pre-commit install
```

## 4. Check the install

```bash
atlas --help
atlas data --list
uv run pytest
```

`atlas` with no arguments opens the agent TUI in the current directory. The first launch compiles the Rust client under `tui/target/`, which can take a few minutes. Later launches reuse that binary and rebuild only after Rust changes.

`atlas data` lists satellites and downloads scenes. `atlas sentinel2 optical ...` is shorthand for `atlas data sentinel2 optical ...`.

The TUI reads `OPENROUTER_API_KEY` from the environment or from this repo's `.env`. The model comes from `<workspace>/.atlas/agent.toml` (`model`). When that key is missing, the default is `google/gemini-3.8-flash`. `ATLAS_MODEL` or `--model` overrides the file for one launch.

## Troubleshooting

- **`uv: command not found`** — open a new terminal, or add `~/.local/bin` to `PATH`.
- **`atlas: command not found`** — same `PATH` fix, then run `uv tool install --editable --force .` from the repo.
- **The TUI says cargo was not found** — install rustup from step 1 and run `atlas` again.
- **The first `atlas` prints `Building the Atlas TUI...` and waits** — that compile is once per Rust change. Leave it running.
- **`OPENROUTER_API_KEY is not set`** — fill in `.env`. The TUI loads it when it can find this checkout.
- **`gitleaks` hook fails** — confirm `gitleaks version` works in the same shell you use for `git commit`.
- **pre-commit cannot install hook environments (401 / missing setuptools)** — pip is using a private index. Bootstrap once from PyPI:

  ```bash
  PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL= uv run pre-commit run --all-files
  ```
