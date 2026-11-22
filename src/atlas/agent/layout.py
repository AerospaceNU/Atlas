"""Project ``.atlas/`` and user ``~/.atlas/`` paths, layout, and credential storage."""

from __future__ import annotations

import logging
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Literal

from atlas.agent.redaction import redact_text

ATLAS_HOME_ENV = "ATLAS_HOME"
ATLAS_DIR_NAME = ".atlas"
AGENTS_DIR_NAME = ".agents"
SKILLS_DIR_NAME = "skills"
SKILL_DRAFTS_DIR_NAME = "skills-drafts"
AGENTS_MD_FILENAME = "AGENTS.md"
MANIFEST_FILENAME = "manifest.json"
AGENT_CONFIG_FILENAME = "agent.toml"
USER_CONFIG_FILENAME = "config.toml"
AGENTS_SKILLS_RELATIVE = f"{AGENTS_DIR_NAME}/{SKILLS_DIR_NAME}"
ATLAS_SKILLS_RELATIVE = f"{ATLAS_DIR_NAME}/{SKILLS_DIR_NAME}"
ATLAS_SKILL_DRAFTS_RELATIVE = f"{ATLAS_DIR_NAME}/{SKILL_DRAFTS_DIR_NAME}"
MANIFEST_TEMP_PREFIX = f".{MANIFEST_FILENAME}."
MANIFEST_TEMP_SUFFIX = ".tmp"
MANIFEST_TEMP_GLOB = f".{MANIFEST_FILENAME}.*.tmp"

_LIST_KINDS = frozenset({"sessions", "artifacts"})
_REMOVE_KINDS = frozenset({"sessions", "artifacts", "keys", "weights"})
OPENROUTER_API_KEY = "OPENROUTER_API_KEY"
EARTHDATA_TOKEN = "EARTHDATA_TOKEN"
FIRMS_MAP_KEY = "FIRMS_MAP_KEY"
KEY_VARIABLES = frozenset({OPENROUTER_API_KEY, EARTHDATA_TOKEN, FIRMS_MAP_KEY})
_KEY_FILES = {
    OPENROUTER_API_KEY: "openrouter_api_key",
    EARTHDATA_TOKEN: "earthdata_token",
    FIRMS_MAP_KEY: "firms_map_key",
}
_FIRMS_MAP_KEY_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_JWT_SEGMENT = r"[A-Za-z0-9_-]+={0,2}"
_EARTHDATA_TOKEN_RE = re.compile(rf"^{_JWT_SEGMENT}\.{_JWT_SEGMENT}\.{_JWT_SEGMENT}$")

_active_session_keys: dict[str, str] = {}
_active_secrets: list[str] | None = None

Kind = Literal["sessions", "artifacts", "keys", "weights"]


class AtlasPathError(ValueError):
    """A layout path left the chosen ``.atlas`` root or skipped confirmation."""


class CredentialKeyWarning(UserWarning):
    """A stored credential was kept, but its file mode looks wrong."""


class OpenRouterKeyWarning(CredentialKeyWarning):
    """The OpenRouter key was kept, but its shape or file mode looks wrong."""


def bind_session_credentials(store: dict[str, str]) -> None:
    global _active_session_keys
    _active_session_keys = store


def unbind_session_credentials() -> None:
    global _active_session_keys
    _active_session_keys = {}


def bind_session_secrets(secrets: list[str]) -> None:
    global _active_secrets
    _active_secrets = secrets


def unbind_session_secrets() -> None:
    global _active_secrets
    _active_secrets = None


def remember_secret(value: str) -> None:
    """Append ``value`` to the bound session secret list when active."""
    secrets = _active_secrets
    if secrets is None:
        return
    cleaned = value.strip()
    if cleaned and cleaned not in secrets:
        secrets.append(cleaned)


class _HttpxRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        redacted = redact_text(rendered, _active_secrets or ())
        if redacted != rendered:
            record.msg = redacted
            record.args = None
        return True


def quiet_http_client_logs() -> None:
    """Keep the httpx logger at WARNING and redact credential leaks from log records."""
    logger = logging.getLogger("httpx")
    if logger.level == logging.NOTSET or logger.level < logging.WARNING:
        logger.setLevel(logging.WARNING)
    if not any(isinstance(item, _HttpxRedactionFilter) for item in logger.filters):
        logger.addFilter(_HttpxRedactionFilter())


def session_credential(name: str) -> str | None:
    """Return the bound session value for ``name``, or None when unset/blank."""
    value = _active_session_keys.get(name)
    return value.strip() if value and value.strip() else None


def key_filename(name: str) -> str:
    """Return the user-key filename for an allowed variable name."""
    try:
        return _KEY_FILES[name]
    except KeyError:
        raise ValueError("unknown key variable") from None


def atlas_home_override() -> Path | None:
    override = os.environ.get(ATLAS_HOME_ENV, "").strip()
    return Path(override).expanduser() if override else None


def find_repo_root(start: Path | None = None, *, require_tui: bool = False) -> Path | None:
    """Walk upward from ``start`` (or cwd/package) for the Atlas checkout root."""
    starts = [start] if start is not None else [Path.cwd(), Path(__file__).resolve()]
    seen: set[Path] = set()
    for origin in starts:
        for candidate in (origin, *origin.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            pyproject = candidate / "pyproject.toml"
            if not pyproject.is_file():
                continue
            if require_tui and not (candidate / "tui" / "Cargo.toml").is_file():
                continue
            try:
                if 'name = "atlas"' in pyproject.read_text(encoding="utf-8"):
                    return candidate
            except OSError:
                continue
    return None


def project_atlas_root(workspace: Path) -> Path:
    return workspace / ATLAS_DIR_NAME


def user_atlas_root(home: Path | None = None) -> Path:
    override = atlas_home_override()
    if override is not None:
        return override
    base = Path.home() if home is None else Path(home)
    return base / ATLAS_DIR_NAME


def project_sessions_dir(workspace: Path) -> Path:
    return project_atlas_root(workspace) / "sessions"


def project_artifacts_dir(workspace: Path) -> Path:
    return project_atlas_root(workspace) / "artifacts"


def project_data_dir(workspace: Path | None = None) -> Path:
    return project_atlas_root(Path(".") if workspace is None else workspace) / "data"


def project_session_dir(workspace: Path, session_id: str) -> Path:
    return project_sessions_dir(workspace) / session_id


def project_artifact_dir(workspace: Path, session_id: str) -> Path:
    return project_artifacts_dir(workspace) / session_id


def project_skills_root(workspace: Path) -> Path:
    return project_atlas_root(workspace) / SKILLS_DIR_NAME


def project_skill_drafts_root(workspace: Path) -> Path:
    return project_atlas_root(workspace) / SKILL_DRAFTS_DIR_NAME


def project_agents_skills_dir(workspace: Path) -> Path:
    return workspace / AGENTS_DIR_NAME / SKILLS_DIR_NAME


def project_agents_md(workspace: Path) -> Path:
    return workspace / AGENTS_MD_FILENAME


def project_agent_config_path(workspace: Path) -> Path:
    return project_atlas_root(workspace) / AGENT_CONFIG_FILENAME


def pull_manifest_path(directory: Path) -> Path:
    return directory / MANIFEST_FILENAME


def user_skills_root(home: Path | None = None) -> Path:
    return user_atlas_root(home) / SKILLS_DIR_NAME


def user_skill_drafts_root(home: Path | None = None) -> Path:
    return user_atlas_root(home) / SKILL_DRAFTS_DIR_NAME


def user_agents_skills_dir(home: Path | None = None) -> Path:
    base = Path.home() if home is None else Path(home)
    return base / AGENTS_DIR_NAME / SKILLS_DIR_NAME


def user_config_path(home: Path | None = None) -> Path:
    return user_atlas_root(home) / USER_CONFIG_FILENAME


def user_keys_dir(home: Path | None = None) -> Path:
    return user_atlas_root(home) / "keys"


def user_weights_dir(home: Path | None = None) -> Path:
    return user_atlas_root(home) / "weights"


def user_defaults_dir(home: Path | None = None) -> Path:
    return user_atlas_root(home) / "defaults"


def user_key_path(home: Path | None = None, *, name: str = OPENROUTER_API_KEY) -> Path:
    return user_keys_dir(home) / key_filename(name)


_ATLAS_GITIGNORE = (
    "# Atlas local state: sessions, artifacts, checkpoints. Not for version control.\n*\n"
)


def ensure_atlas_gitignore(workspace: Path) -> None:
    """Write ``.atlas/.gitignore`` ignoring everything, only when missing."""
    target = project_atlas_root(workspace) / ".gitignore"
    try:
        if not target.exists():
            target.write_text(_ATLAS_GITIGNORE, encoding="utf-8")
    except OSError:
        return


def ensure_project_layout(workspace: Path) -> Path:
    """Create project session, artifact, and data directories if missing."""
    root = project_atlas_root(workspace)
    for directory in (
        project_sessions_dir(workspace),
        project_artifacts_dir(workspace),
        project_data_dir(workspace),
    ):
        directory.mkdir(parents=True, exist_ok=True)
    ensure_atlas_gitignore(workspace)
    return root


def ensure_user_layout(home: Path | None = None) -> Path:
    """Create user key, weight, and defaults directories if missing, tightening ``keys`` to 0700."""
    root = user_atlas_root(home)
    for directory in (user_keys_dir(home), user_weights_dir(home), user_defaults_dir(home)):
        directory.mkdir(parents=True, exist_ok=True)
    keys = root / "keys"
    if not keys.is_symlink() and keys.is_dir() and _is_inside(root, keys):
        _chmod(keys, 0o700)
    return root


def list_entries(atlas_root: Path, kind: str) -> list[str]:
    """List session or artifact names inside ``atlas_root``."""
    if kind not in _LIST_KINDS:
        raise AtlasPathError("only sessions and artifacts can be listed")
    directory = atlas_root / kind
    return sorted(path.name for path in directory.iterdir()) if directory.is_dir() else []


def remove_entry(atlas_root: Path, kind: str, name: str, *, confirm: bool) -> None:
    """Remove one named entry inside ``atlas_root`` (requiring ``confirm=True``)."""
    if not confirm:
        raise AtlasPathError("confirmation required")
    if kind not in _REMOVE_KINDS:
        raise AtlasPathError("unknown atlas entry kind")
    if kind in {"keys", "weights"} and not name.strip():
        raise AtlasPathError("keys and weights must be named explicitly")
    target = _resolve_inside(atlas_root / kind, name)
    if not target.exists():
        raise FileNotFoundError(name)
    if kind == "sessions":
        artifact = _resolve_inside(atlas_root / "artifacts", name)
        if artifact.exists():
            _delete(artifact)
    _delete(target)


def read_user_key(home: Path | None = None, *, name: str = OPENROUTER_API_KEY) -> str | None:
    """Return the stored user key for ``name``, ignoring symlinks and escaped dirs."""
    path = user_key_path(home, name=name)
    keys = path.parent
    if keys.is_symlink() or path.is_symlink() or not _is_inside(keys.parent, keys):
        return None
    if not path.is_file():
        return None
    try:
        fd = os.open(path, os.O_RDONLY | _no_follow_flag())
    except OSError:
        return None
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if os.name == "posix" and info.st_mode & 0o077:
            warn_cls = OpenRouterKeyWarning if name == OPENROUTER_API_KEY else CredentialKeyWarning
            label = "OpenRouter key" if name == OPENROUTER_API_KEY else name
            warnings.warn(f"{label} file permissions are too loose", warn_cls, stacklevel=2)
        text = stream.read().decode("utf-8").strip()
    return None if not text or "\n" in text or "\r" in text else text


def prepare_credential(key: str, *, name: str) -> str:
    """Strip and validate a credential value for ``name`` without echoing it in errors."""
    key_filename(name)
    cleaned = key.strip()
    if not cleaned or "\n" in cleaned or "\r" in cleaned:
        raise ValueError(f"{name} is empty or contains a newline")
    if name == FIRMS_MAP_KEY and _FIRMS_MAP_KEY_RE.fullmatch(cleaned) is None:
        raise ValueError(f"{name} must be 32 hexadecimal characters")
    if name == EARTHDATA_TOKEN and _EARTHDATA_TOKEN_RE.fullmatch(cleaned) is None:
        raise ValueError(f"{name} must be a JWT of three base64url segments")
    return cleaned


def prepare_openrouter_key(key: str, *, warn: bool = True) -> tuple[str, str]:
    """Strip an OpenRouter key and optionally warn when it lacks the ``sk-or-`` prefix."""
    cleaned = key.strip()
    if not cleaned or "\n" in cleaned or "\r" in cleaned:
        raise ValueError("key is empty or contains a newline")
    warning = ""
    if not cleaned.startswith("sk-or-"):
        warning = "OpenRouter key does not start with sk-or-"
        if warn:
            warnings.warn(warning, OpenRouterKeyWarning, stacklevel=2)
    return cleaned, warning


def write_user_key(key: str, home: Path | None = None, *, name: str = OPENROUTER_API_KEY) -> Path:
    """Persist a user key atomically under ``~/.atlas/keys`` with mode ``0600``."""
    cleaned = (
        prepare_openrouter_key(key)[0]
        if name == OPENROUTER_API_KEY
        else prepare_credential(key, name=name)
    )
    root = ensure_user_layout(home)
    keys = root / "keys"
    if keys.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    if not keys.is_dir() or not _is_inside(root, keys):
        raise AtlasPathError("key path escapes the user atlas root")
    _chmod(keys, 0o700)
    path = keys / key_filename(name)
    _atomic_write_private(keys, path, cleaned + "\n")
    return path


def delete_user_key(home: Path | None = None, *, name: str = OPENROUTER_API_KEY) -> bool:
    """Remove the user key file for ``name``, refusing symlinks or escaped dirs."""
    path = user_key_path(home, name=name)
    keys = path.parent
    if keys.is_symlink() or path.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    if keys.exists() and not _is_inside(keys.parent, keys):
        raise AtlasPathError("key path escapes the user atlas root")
    if not path.is_file():
        return False
    path.unlink()
    return True


def resolve_credential(
    name: str,
    *,
    session_key: str | None = None,
    env_key: str | None = None,
    home: Path | None = None,
) -> str | None:
    """Resolve credential ``name``: session > env > user file."""
    key_filename(name)
    if session_key is None:
        session_key = session_credential(name)
    if session_key and session_key.strip():
        return _remember(session_key.strip())
    if env_key is None:
        env_key = os.environ.get(name)
    if env_key and env_key.strip():
        return _remember(env_key.strip())
    return _remember(read_user_key(home, name=name))


def resolve_openrouter_key(
    *,
    session_key: str | None,
    env_key: str | None,
    home: Path | None = None,
) -> str | None:
    """Resolve the OpenRouter key from explicit session/env candidates or the user file."""
    if session_key and session_key.strip():
        return _remember(session_key.strip())
    if env_key and env_key.strip():
        return _remember(env_key.strip())
    return _remember(read_user_key(home))


def _remember(value: str | None) -> str | None:
    if value:
        remember_secret(value)
    return value


def _no_follow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", 0)
    return flag if isinstance(flag, int) else 0


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        if os.name == "posix":
            raise


def _atomic_write_private(directory: Path, path: Path, text: str) -> None:
    """Replace ``path`` via a mode-``0600`` temp file with file and directory fsync."""
    if path.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    if path.exists() and not path.is_file():
        raise AtlasPathError("key path is not a file")
    if path.is_file():
        _chmod(path, 0o600)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _no_follow_flag()
    temp = directory / f".{path.name}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
    fd = os.open(temp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(text.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    try:
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    if os.name == "posix":
        dir_flags = os.O_RDONLY | (
            getattr(os, "O_DIRECTORY", 0) if isinstance(getattr(os, "O_DIRECTORY", 0), int) else 0
        )
        dir_fd = os.open(directory, dir_flags)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    _chmod(path, 0o600)


def _is_inside(root: Path, candidate: Path) -> bool:
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    return candidate_resolved == root_resolved or root_resolved in candidate_resolved.parents


def _resolve_inside(parent: Path, name: str) -> Path:
    relative = Path(name)
    if (
        not name.strip()
        or relative.is_absolute()
        or ".." in relative.parts
        or relative.parts == (".",)
    ):
        raise AtlasPathError("path escapes the atlas root")
    parent_resolved = parent.resolve()
    candidate = (parent / relative).resolve()
    if candidate == parent_resolved or parent_resolved not in candidate.parents:
        raise AtlasPathError("path escapes the atlas root")
    return candidate


def _delete(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
