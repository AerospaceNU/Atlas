from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class TuiLaunchError(Exception):
    """The Rust TUI binary could not be found or built."""


def _has_flag(argv: list[str], flag: str) -> bool:
    prefix = f"{flag}="
    return any(tok == flag or tok.startswith(prefix) for tok in argv)


def native_windows_message() -> str:
    """Return the guidance shown when Atlas is started on native Windows."""
    return (
        "the Atlas agent needs a POSIX shell and process groups; native Windows is not "
        "supported. Install WSL2 (`wsl --install -d Ubuntu`) and run atlas inside it "
        "(see install-instructions/INSTALL-LINUX.md). Set ATLAS_ALLOW_NATIVE_WINDOWS=1 "
        "to try anyway (unsupported)."
    )


def launch_tui(argv: list[str]) -> int:
    """Replace this process with ``atlas-tui``, forwarding repo root and Python executable."""
    if sys.platform == "win32" and os.environ.get("ATLAS_ALLOW_NATIVE_WINDOWS") != "1":
        print(f"atlas: {native_windows_message()}", file=sys.stderr)
        return 1
    try:
        binary = ensure_tui_binary()
    except TuiLaunchError as exc:
        print(f"atlas: {exc}", file=sys.stderr)
        return 1

    forwarded = list(argv)
    if not _has_flag(forwarded, "--repo"):
        repo = _repo_root()
        if repo is not None:
            forwarded = ["--repo", str(repo), *forwarded]
    if not _has_flag(forwarded, "--python"):
        python_exe = os.environ.get("ATLAS_PYTHON") or sys.executable
        forwarded = ["--python", python_exe, *forwarded]
    os.environ.setdefault("ATLAS_PYTHON", sys.executable)

    try:
        os.execv(str(binary), [str(binary), *forwarded])
    except OSError as exc:
        print(f"atlas: could not start {binary}: {exc}", file=sys.stderr)
        return 1


def ensure_tui_binary() -> Path:
    """Return the ``atlas-tui`` binary, building a release binary when non-test sources changed."""
    explicit = os.environ.get("ATLAS_TUI")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise TuiLaunchError(f"ATLAS_TUI is not a file: {path}")
        return path

    repo = _repo_root()
    release = (repo / "tui" / "target" / "release" / "atlas-tui") if repo is not None else None
    debug = (repo / "tui" / "target" / "debug" / "atlas-tui") if repo is not None else None
    on_path = _which("atlas-tui")
    source_mtime = _source_mtime(repo) if repo is not None else None

    for candidate in (release, debug):
        if candidate is not None and _is_fresh(candidate, source_mtime):
            return candidate

    cargo = _cargo_path()
    if repo is not None and cargo is not None and (repo / "tui" / "Cargo.toml").is_file():
        return _build(repo, cargo)

    for candidate in (release, debug, on_path):
        if candidate is not None and candidate.is_file():
            return candidate

    raise TuiLaunchError(
        "the TUI binary is not built and cargo was not found. "
        "Install Rust from https://rustup.rs, or set ATLAS_TUI to an atlas-tui binary."
    )


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    return Path(found) if found is not None else None


def _is_fresh(binary: Path, source_mtime: float | None) -> bool:
    if not binary.is_file():
        return False
    return True if source_mtime is None else binary.stat().st_mtime >= source_mtime


def _repo_root() -> Path | None:
    seen: set[Path] = set()
    for start in (Path(__file__).resolve(), Path.cwd()):
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            pyproject = candidate / "pyproject.toml"
            if not pyproject.is_file() or not (candidate / "tui" / "Cargo.toml").is_file():
                continue
            if 'name = "atlas"' in pyproject.read_text(encoding="utf-8"):
                return candidate
    return None


def _source_mtime(repo: Path) -> float | None:
    paths = [repo / "tui" / "Cargo.toml", repo / "tui" / "Cargo.lock"]
    src = repo / "tui" / "src"
    if src.is_dir():
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            if (
                path.name == "tests.rs"
                or path.name.endswith("_tests.rs")
                or "tests" in path.relative_to(src).parts
            ):
                continue
            paths.append(path)
    mtimes = [path.stat().st_mtime for path in paths if path.is_file()]
    return max(mtimes) if mtimes else None


def _cargo_path() -> Path | None:
    explicit = os.environ.get("ATLAS_CARGO")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    found = shutil.which("cargo")
    if found is not None:
        return Path(found)
    home = Path.home() / ".cargo" / "bin" / "cargo"
    return home if home.is_file() else None


def _build(repo: Path, cargo: Path) -> Path:
    print("Building the Atlas TUI...", file=sys.stderr)
    env = os.environ.copy()
    env["PATH"] = str(cargo.parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [str(cargo), "build", "--release", "--manifest-path", str(repo / "tui" / "Cargo.toml")],
        cwd=repo,
        env=env,
        check=False,
    )
    binary = repo / "tui" / "target" / "release" / "atlas-tui"
    if result.returncode != 0 or not binary.is_file():
        raise TuiLaunchError("cargo build of the TUI failed")
    return binary
