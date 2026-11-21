"""Locate tools inside a project's own virtual environment (POSIX layout).

Invariant: stat only, never PATH lookup, never ``resolve()`` (a venv python is a symlink
whose target loses pyvenv.cfg and site-packages).
"""

from __future__ import annotations

import os
from pathlib import Path

VENV_DIRS: tuple[str, ...] = (".venv", "venv")


def project_bin(root: Path, name: str) -> str | None:
    """Return ``<root>/{.venv,venv}/bin/<name>`` when it is an executable file."""
    for venv in VENV_DIRS:
        candidate = root / venv / "bin" / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def project_python(root: Path) -> str | None:
    """Return the project venv's python (``python`` then ``python3``), or None."""
    for venv in VENV_DIRS:
        for name in ("python", "python3"):
            candidate = root / venv / "bin" / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None
