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

## 3. Install git hooks

Install hooks once after cloning so lint, format, type-check, tests, and secret
scans run on every commit:

```powershell
uv run pre-commit install
```

## 4. Check the install

```powershell
uv run atlas
uv run pytest
```

`uv run atlas` prints `Hello from atlas!`.

## Troubleshooting

- **`uv` is not recognized** — close PowerShell and open a new window. Confirm `%USERPROFILE%\.local\bin` is on your user `PATH`.
- **`gitleaks` hook fails** — open a new terminal after installing, then run `gitleaks version`.
- **pre-commit cannot install hook environments (401 / missing setuptools)** — pip is using a private index. Bootstrap once from PyPI:

  ```powershell
  $env:PIP_INDEX_URL = "https://pypi.org/simple"
  $env:PIP_EXTRA_INDEX_URL = ""
  uv run pre-commit run --all-files
  ```
