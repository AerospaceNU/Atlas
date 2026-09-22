from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


class TuiLaunchError(Exception):
    """The Rust TUI binary could not be found or built."""


def launch_tui(argv: list[str]) -> int:
    """Replace this process with the Rust TUI, passing ``argv`` through.

    ``os.execv`` does not return on success. The terminal stays attached to
    the TUI, which is what raw mode needs.
    """
    try:
        binary = ensure_tui_binary()
    except TuiLaunchError as exc:
        print(f"atlas: {exc}", file=sys.stderr)
        return 1
    try:
        os.execv(str(binary), [str(binary), *argv])
    except OSError as exc:
        print(f"atlas: could not start {binary}: {exc}", file=sys.stderr)
        return 1


def ensure_tui_binary() -> Path:
    """Return the ``atlas-tui`` binary, building a release one when sources changed.

    ``ATLAS_TUI`` wins and is not rebuilt. Otherwise a fresh binary already
    under ``tui/target`` is used. A missing or stale binary is built with
    ``cargo`` from ``PATH`` or ``~/.cargo/bin``, so the caller does not need
    Rust on ``PATH``.
    """
    explicit = os.environ.get("ATLAS_TUI")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise TuiLaunchError(f"ATLAS_TUI is not a file: {path}")
        return path

    repo = _repo_root()
    release = _target_binary(repo, "release")
    debug = _target_binary(repo, "debug")
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


def _target_binary(repo: Path | None, profile: str) -> Path | None:
    if repo is None:
        return None
    return repo / "tui" / "target" / profile / "atlas-tui"


def _which(name: str) -> Path | None:
    found = shutil.which(name)
    if found is None:
        return None
    return Path(found)


def _is_fresh(binary: Path, source_mtime: float | None) -> bool:
    if not binary.is_file():
        return False
    if source_mtime is None:
        return True
    return binary.stat().st_mtime >= source_mtime


def _repo_root() -> Path | None:
    starts = [Path.cwd(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            pyproject = candidate / "pyproject.toml"
            cargo = candidate / "tui" / "Cargo.toml"
            if not pyproject.is_file() or not cargo.is_file():
                continue
            text = pyproject.read_text(encoding="utf-8")
            if 'name = "atlas"' in text:
                return candidate
    return None


def _source_mtime(repo: Path) -> float | None:
    paths = [repo / "tui" / "Cargo.toml", repo / "tui" / "Cargo.lock"]
    src = repo / "tui" / "src"
    if src.is_dir():
        paths.extend(path for path in src.rglob("*") if path.is_file())
    mtimes = [path.stat().st_mtime for path in paths if path.is_file()]
    if not mtimes:
        return None
    return max(mtimes)


def _cargo_path() -> Path | None:
    explicit = os.environ.get("ATLAS_CARGO")
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    found = shutil.which("cargo")
    if found is not None:
        return Path(found)
    home = Path.home() / ".cargo" / "bin" / "cargo"
    if home.is_file():
        return home
    return None


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
