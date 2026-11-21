# Install Atlas on Windows

The Atlas agent (the TUI and `atlas run`) needs a POSIX shell and process
groups. Its tools start `bash` (or `/bin/sh`) in its own session and kill the
process group on cancel. Native Windows has neither, so it cannot run the
agent. Use WSL2 and follow [INSTALL-LINUX.md](INSTALL-LINUX.md) inside WSL.

```powershell
wsl --install -d Ubuntu
```

Reboot if Windows asks, open **Ubuntu** from the Start menu, then follow
INSTALL-LINUX.md there. Keep the checkout and your workspaces on the WSL file
system (for example under `~`), not under `/mnt/c`.

`atlas` on native Windows refuses to open the TUI and prints:

> atlas: the Atlas agent needs a POSIX shell and process groups; native Windows is not
> supported. Install WSL2 (`wsl --install -d Ubuntu`) and run atlas inside it
> (see install-instructions/INSTALL-LINUX.md). Set ATLAS_ALLOW_NATIVE_WINDOWS=1
> to try anyway (unsupported).

`atlas run` exits 7 (not configured) with the same message.

`ATLAS_ALLOW_NATIVE_WINDOWS=1` bypasses that refusal. It is unsupported and
untested: bash tools and process-group kills will not work.

## Native Windows (data tooling only, not exercised in CI)

The rest of this page covers native PowerShell for the data tooling only. The
supported native commands are `atlas data` and `atlas indices`. Nothing here
is exercised in CI.

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

- `OPENROUTER_API_KEY` — used by the agent, which runs in WSL (see above). Create a key at https://openrouter.ai/keys
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
atlas indices
uv run pytest
```

`atlas data` lists satellites and downloads scenes. `atlas sentinel2 optical ...` is shorthand for `atlas data sentinel2 optical ...`.

The agent, its `agent.toml` and the `/key` commands run inside WSL; see
[INSTALL-LINUX.md](INSTALL-LINUX.md).

### Key files on native Windows

Key files request mode `0600` and the keys directory `0700`. Windows does not
enforce those Unix bits, and it has no directory fsync (that step is skipped), so
the file ACL is what restricts access to `%USERPROFILE%\.atlas\keys`. Windows
also has no `O_NOFOLLOW`, so a symlink at the key path is rejected by name.

## Troubleshooting

- **`uv` is not recognized** — close PowerShell and open a new window. Confirm `%USERPROFILE%\.local\bin` is on your user `PATH`.
- **`atlas` is not recognized** — same `PATH` fix, then run `uv tool install --editable --force .` from the repo.
- **`atlas` says the agent needs a POSIX shell** — expected on native Windows. Use WSL2 (top of this page).
- **`gitleaks` hook fails** — open a new terminal after installing, then run `gitleaks version`.
- **pre-commit cannot install hook environments (401 / missing setuptools)** — pip is using a private index. Bootstrap once from PyPI:

  ```powershell
  $env:PIP_INDEX_URL = "https://pypi.org/simple"
  $env:PIP_EXTRA_INDEX_URL = ""
  uv run pre-commit run --all-files
  ```
