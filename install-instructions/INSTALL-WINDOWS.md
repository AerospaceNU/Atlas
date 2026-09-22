# Install Atlas on Windows

Use **PowerShell**. If you develop in WSL, follow [INSTALL-LINUX.md](INSTALL-LINUX.md) inside WSL instead.

These steps assume `git` is already on your `PATH`.

Python 3.12 is pinned in `.python-version`. `uv` will download it if needed.

## 1. Install tools

[uv](https://docs.astral.sh/uv/getting-started/installation/) (skip if `uv --version` already works):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Open a **new** PowerShell window so `PATH` picks up `%USERPROFILE%\.local\bin`.

[gitleaks](https://github.com/gitleaks/gitleaks) (required for the pre-commit secret scan):

```powershell
winget install --id Gitleaks.Gitleaks -e --source winget
```

Alternatives: `scoop install gitleaks` or `choco install gitleaks`.

Rust, for the agent TUI. `atlas data` does not need it. [rustup](https://rustup.rs) installs `cargo` into `%USERPROFILE%\.cargo\bin`. Atlas looks there even when that directory is not on `PATH`:

```powershell
winget install Rustlang.Rustup
```

Open a new PowerShell window after rustup finishes so it can install the stable toolchain.

## 2. Clone and install the project

```powershell
git clone https://github.com/AerospaceNU/Atlas.git
cd Atlas

uv python install 3.12
uv sync

Copy-Item .env.example .env
```

Edit `.env`:

- `OPENROUTER_API_KEY` — required to run the agent. Create a key at https://openrouter.ai/keys
- `FIRMS_MAP_KEY` — optional; required only for FIRMS fire sources. Free key from https://firms.modaps.eosdis.nasa.gov/api/map_key/

Put `atlas` on `PATH`, pointed at this checkout:

```powershell
uv tool install --editable --force .
```

Source edits in this repo are picked up without reinstalling. Run that command again after dependency changes. `%USERPROFILE%\.local\bin` must be on `PATH`. A normal uv install already adds it.

## 3. Install git hooks

Install hooks once after cloning so lint, format, type-check, tests, and secret
scans run on every commit:

```powershell
uv run pre-commit install
```

## 4. Check the install

```powershell
atlas --help
atlas data --list
uv run pytest
```

`atlas` with no arguments opens the agent TUI in the current directory. The first launch compiles the Rust client under `tui\target\`, which can take a few minutes. Later launches reuse that binary and rebuild only after Rust changes.

`atlas data` lists satellites and downloads scenes. `atlas sentinel2 optical ...` is shorthand for `atlas data sentinel2 optical ...`.

The TUI reads `OPENROUTER_API_KEY` from the environment or from this repo's `.env`. The first `atlas` launch writes `<workspace>/.atlas/agent.toml` when that file is missing, with `model` set to `google/gemini-3.8-flash`. Edit that file to change the model. `ATLAS_MODEL` or `--model` overrides it for one launch.

## Troubleshooting

- **`uv` is not recognized** — close PowerShell and open a new window. Confirm `%USERPROFILE%\.local\bin` is on your user `PATH`.
- **`atlas` is not recognized** — same `PATH` fix, then run `uv tool install --editable --force .` from the repo.
- **The TUI says cargo was not found** — install rustup from step 1, open a new window, and run `atlas` again.
- **The first `atlas` prints `Building the Atlas TUI...` and waits** — that compile is once per Rust change. Leave it running.
- **`OPENROUTER_API_KEY is not set`** — fill in `.env`. The TUI loads it when it can find this checkout.
- **`gitleaks` hook fails** — open a new terminal after installing, then run `gitleaks version`.
- **pre-commit cannot install hook environments (401 / missing setuptools)** — pip is using a private index. Bootstrap once from PyPI:

  ```powershell
  $env:PIP_INDEX_URL = "https://pypi.org/simple"
  $env:PIP_EXTRA_INDEX_URL = ""
  uv run pre-commit run --all-files
  ```
