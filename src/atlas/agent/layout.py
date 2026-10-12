"""Project ``.atlas/`` and user ``~/.atlas/`` locations.

Project root (the workspace, the current directory):

- ``sessions/<id>/`` — one directory per session. ``meta.json`` records the
  id and model. No keys.
- ``artifacts/<id>/`` — files that belong to that session id
- ``data/`` — pulls from the data CLI (existing ``.atlas/data``)
- ``skills/`` and ``skills-drafts/`` — vetted skills and drafts that are not
  loaded until a person enables them
- ``agent.toml`` — workspace model and tool-call budget

``AGENTS.md`` stays at the workspace root. ``.agents/skills`` stays beside
``.atlas``, not inside it.

User root (the home directory):

- ``keys/`` — API keys, never written into the project tree. The directory
  is mode ``0700`` and each key file is mode ``0600``. Neither is read or
  written through a symlink. ``OPENROUTER_API_KEY``, ``EARTHDATA_TOKEN``, and
  ``FIRMS_MAP_KEY`` can be saved here for the user. A session value stays in
  memory and wins over the environment, which wins over the file.
- ``weights/`` — user-level weights (existing ``~/.atlas/weights``)
- ``defaults/`` — user defaults
- ``skills/``, ``skills-drafts/``, and ``config.toml`` — user skills and the
  ``home_skills`` flag

``ATLAS_HOME`` replaces this user root. It is the directory that would
otherwise be ``~/.atlas``, not the parent home. ``ATLAS_HOME_SKILLS`` still
only turns user skills on; the skills and ``config.toml`` it refers to are
read from that same root. On Windows the default user root is
``%USERPROFILE%/.atlas``.

The directory list other tickets should point at is ``docs/atlas-layout.md``.

Removing a session removes its artifacts. Removing entries from the project
root does not touch the user root. Absolute paths and ``..`` fail closed.
Keys and weights are deleted only when that kind is named.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import warnings
from pathlib import Path
from typing import Literal

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
SANDBOX_AUDIT_RELATIVE = f"{ATLAS_DIR_NAME}/sandbox_audit.log"
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
# The live session's credential map. ``serve`` binds the session dict here and
# unbinds it when the loop ends, so a session value dies with that process.
_active_session_keys: dict[str, str] = {}
# The live session's redaction list. Resolved env and file values are appended
# here so a later protocol line can hide them. Unbound outside a session.
_active_secrets: list[str] | None = None
# FIRMS area tutorial map key ``abcdef1234567890abcdef1234567890``.
# https://firms.modaps.eosdis.nasa.gov/api/area/
_FIRMS_MAP_KEY_RE = re.compile(r"^[0-9a-fA-F]{32}$")
# Earthdata Login user tokens are JWTs (three base64url segments).
# https://urs.earthdata.nasa.gov/documentation/for_users/user_token
_JWT_SEGMENT = r"[A-Za-z0-9_-]+={0,2}"
_EARTHDATA_TOKEN_RE = re.compile(rf"^{_JWT_SEGMENT}\.{_JWT_SEGMENT}\.{_JWT_SEGMENT}$")
# httpx logs the full request URL at INFO. FIRMS puts MAP_KEY in that path.
_FIRMS_PATH_KEY = re.compile(r"(https://firms\.modaps\.eosdis\.nasa\.gov/api/area/csv/)[^/\s\"']+")
_HTTPX_LOGGER = "httpx"

Kind = Literal["sessions", "artifacts", "keys", "weights"]


class AtlasPathError(ValueError):
    """A layout path left the chosen ``.atlas`` root or skipped confirmation."""


class CredentialKeyWarning(UserWarning):
    """A stored credential was kept, but its file mode looks wrong."""


class OpenRouterKeyWarning(CredentialKeyWarning):
    """The OpenRouter key was kept, but its shape or file mode looks wrong."""


def bind_session_credentials(store: dict[str, str]) -> None:
    """Point credential lookup at the live session map.

    ``store`` is the session object. Later writes to that dict are visible
    to data clients in this process. The map is not copied to disk.
    """
    global _active_session_keys
    _active_session_keys = store


def unbind_session_credentials() -> None:
    """Drop the live session map so later lookups no longer see it."""
    global _active_session_keys
    _active_session_keys = {}


def bind_session_secrets(secrets: list[str]) -> None:
    """Point credential redaction at the live session secret list.

    ``secrets`` is the session object. Resolved credentials are appended to
    it. The list is not copied to disk.
    """
    global _active_secrets
    _active_secrets = secrets


def unbind_session_secrets() -> None:
    """Drop the live secret list so later resolves are not remembered."""
    global _active_secrets
    _active_secrets = None


def remember_secret(value: str) -> None:
    """Append ``value`` to the bound session secret list.

    No list is bound outside a session, and then this does nothing. Blank
    values are ignored. A value already in the list is left where it is.
    """
    secrets = _active_secrets
    if secrets is None:
        return
    cleaned = value.strip()
    if cleaned and cleaned not in secrets:
        secrets.append(cleaned)


def quiet_http_client_logs() -> None:
    """Keep the httpx logger at WARNING and redact FIRMS map keys.

    httpx writes the request URL at INFO. A FIRMS area URL carries
    ``FIRMS_MAP_KEY`` in the path, so that line is a credential leak. The
    logger is raised to WARNING when it is unset or more verbose. A filter
    still redacts a FIRMS path key, and any bound session secret, if a record
    is emitted anyway.
    """
    logger = logging.getLogger(_HTTPX_LOGGER)
    if logger.level == logging.NOTSET or logger.level < logging.WARNING:
        logger.setLevel(logging.WARNING)
    if not any(isinstance(item, _HttpxRedactionFilter) for item in logger.filters):
        logger.addFilter(_HttpxRedactionFilter())


def session_credential(name: str) -> str | None:
    """Return the bound session value for ``name``, or None when it is blank."""
    value = _active_session_keys.get(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def key_filename(name: str) -> str:
    """Return the user-key filename for an allowed variable name.

    Raises:
        ValueError: if ``name`` is not one of the stored variables. The
        message does not echo ``name``, which may be a pasted secret.
    """
    try:
        return _KEY_FILES[name]
    except KeyError:
        raise ValueError("unknown key variable") from None


def atlas_home_override() -> Path | None:
    """Return ``ATLAS_HOME`` when it names the user root, else None.

    The value is the user ``.atlas`` directory itself. ``~`` is expanded. A
    blank value is treated as unset.
    """
    override = os.environ.get(ATLAS_HOME_ENV, "").strip()
    if not override:
        return None
    return Path(override).expanduser()


def project_atlas_root(workspace: Path) -> Path:
    """Return the project ``.atlas`` directory (it may not exist yet)."""
    return workspace / ATLAS_DIR_NAME


def user_atlas_root(home: Path | None = None) -> Path:
    """Return the user ``.atlas`` directory.

    ``ATLAS_HOME`` wins when it is set, including when ``home`` is passed.
    The variable is that directory, not the parent home. When it is unset,
    ``home`` is the parent and the result is ``<home>/.atlas``. ``None``
    uses the process home (``%USERPROFILE%`` on Windows).
    """
    override = atlas_home_override()
    if override is not None:
        return override
    base = Path.home() if home is None else Path(home)
    return base / ATLAS_DIR_NAME


def project_sessions_dir(workspace: Path) -> Path:
    """Return ``<workspace>/.atlas/sessions`` (it may not exist yet)."""
    return project_atlas_root(workspace) / "sessions"


def project_artifacts_dir(workspace: Path) -> Path:
    """Return ``<workspace>/.atlas/artifacts`` (it may not exist yet)."""
    return project_atlas_root(workspace) / "artifacts"


def project_data_dir(workspace: Path | None = None) -> Path:
    """Return the project data-pull directory.

    Args:
        workspace: Project folder. ``None`` uses ``.`` so the result stays the
            cwd-relative ``.atlas/data`` path the data CLI already defaults to.

    Returns:
        ``<workspace>/.atlas/data``.
    """
    base = Path(".") if workspace is None else workspace
    return project_atlas_root(base) / "data"


def project_session_dir(workspace: Path, session_id: str) -> Path:
    """Return one session directory under the project ``.atlas`` root."""
    return project_sessions_dir(workspace) / session_id


def project_artifact_dir(workspace: Path, session_id: str) -> Path:
    """Return one session's artifact directory under the project ``.atlas`` root."""
    return project_artifacts_dir(workspace) / session_id


def project_skills_root(workspace: Path) -> Path:
    """Return ``<workspace>/.atlas/skills`` without creating it."""
    return project_atlas_root(workspace) / SKILLS_DIR_NAME


def project_skill_drafts_root(workspace: Path) -> Path:
    """Return ``<workspace>/.atlas/skills-drafts`` without creating it.

    Drafts are not loaded from ``.atlas/skills``. A person promotes one later.
    """
    return project_atlas_root(workspace) / SKILL_DRAFTS_DIR_NAME


def project_agents_skills_dir(workspace: Path) -> Path:
    """Return ``<workspace>/.agents/skills`` without creating it.

    This directory sits beside ``.atlas``, not inside it.
    """
    return workspace / AGENTS_DIR_NAME / SKILLS_DIR_NAME


def project_agent_config_path(workspace: Path) -> Path:
    """Return ``<workspace>/.atlas/agent.toml`` without creating it."""
    return project_atlas_root(workspace) / AGENT_CONFIG_FILENAME


def project_agents_md(workspace: Path) -> Path:
    """Return ``<workspace>/AGENTS.md``.

    Project instructions stay at the workspace root. Nested copies are not
    read.
    """
    return workspace / AGENTS_MD_FILENAME


def pull_manifest_path(directory: Path) -> Path:
    """Return the data-pull manifest inside ``directory``."""
    return directory / MANIFEST_FILENAME


def user_skills_root(home: Path | None = None) -> Path:
    """Return the user skills directory under the user ``.atlas`` root.

    ``ATLAS_HOME`` selects that root the same way :func:`user_atlas_root` does,
    so ``ATLAS_HOME_SKILLS`` and ``config.toml`` see the same directory.
    """
    return user_atlas_root(home) / SKILLS_DIR_NAME


def user_skill_drafts_root(home: Path | None = None) -> Path:
    """Return the user skill drafts directory under the user ``.atlas`` root."""
    return user_atlas_root(home) / SKILL_DRAFTS_DIR_NAME


def user_agents_skills_dir(home: Path | None = None) -> Path:
    """Return ``<home>/.agents/skills`` without creating it.

    ``ATLAS_HOME`` does not move this directory. It stays under the home
    parent, beside the user ``.atlas`` root.
    """
    base = Path.home() if home is None else Path(home)
    return base / AGENTS_DIR_NAME / SKILLS_DIR_NAME


def user_config_path(home: Path | None = None) -> Path:
    """Return ``config.toml`` inside the user ``.atlas`` root.

    ``home_skills`` is read from this file when ``ATLAS_HOME_SKILLS`` is unset.
    """
    return user_atlas_root(home) / USER_CONFIG_FILENAME


def user_keys_dir(home: Path | None = None) -> Path:
    """Return ``<home>/.atlas/keys`` (it may not exist yet)."""
    return user_atlas_root(home) / "keys"


def user_weights_dir(home: Path | None = None) -> Path:
    """Return ``<home>/.atlas/weights`` (it may not exist yet)."""
    return user_atlas_root(home) / "weights"


def user_defaults_dir(home: Path | None = None) -> Path:
    """Return ``<home>/.atlas/defaults`` (it may not exist yet)."""
    return user_atlas_root(home) / "defaults"


def ensure_project_layout(workspace: Path) -> Path:
    """Create project session, artifact, and data directories if missing.

    Existing files, including ``.atlas/data``, are left in place.
    """
    root = project_atlas_root(workspace)
    for directory in (
        project_sessions_dir(workspace),
        project_artifacts_dir(workspace),
        project_data_dir(workspace),
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return root


def ensure_user_layout(home: Path | None = None) -> Path:
    """Create user key, weight, and default directories if missing.

    Existing ``~/.atlas/weights`` contents are left in place. An existing
    ``keys`` directory is tightened to mode ``0700``. On Windows that mode
    is requested but not enforced; the directory ACL is the real control.
    """
    root = user_atlas_root(home)
    for directory in (
        user_keys_dir(home),
        user_weights_dir(home),
        user_defaults_dir(home),
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _tighten_keys_dir(root)
    return root


def list_entries(atlas_root: Path, kind: str) -> list[str]:
    """List session or artifact names inside ``atlas_root``.

    Names are the directory entries only. Host paths are not returned.
    """
    if kind not in _LIST_KINDS:
        raise AtlasPathError("only sessions and artifacts can be listed")
    directory = atlas_root / kind
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir())


def remove_entry(atlas_root: Path, kind: str, name: str, *, confirm: bool) -> None:
    """Remove one named entry inside ``atlas_root``.

    ``confirm`` must be true. A session removal also removes the artifact
    directory of the same name. ``keys`` and ``weights`` are removed only
    when ``kind`` names them.
    """
    if not confirm:
        raise AtlasPathError("confirmation required")
    if kind not in _REMOVE_KINDS:
        raise AtlasPathError("unknown atlas entry kind")
    if kind in {"keys", "weights"} and not name.strip():
        raise AtlasPathError("keys and weights must be named explicitly")
    target = _resolve_inside(atlas_root / kind, name)
    if kind == "sessions":
        if not target.exists():
            raise FileNotFoundError(name)
        _delete_matching_artifact(atlas_root, name)
    if not target.exists():
        raise FileNotFoundError(name)
    _delete(target)


def read_user_key(home: Path | None = None, *, name: str = OPENROUTER_API_KEY) -> str | None:
    """Return the user key for ``name``, or None when the file is absent.

    The file is opened with ``O_NOFOLLOW`` where the OS provides it. A
    symlink, or a keys directory that resolves outside the user ``.atlas``
    root, is ignored. On POSIX, a file that is readable by group or others
    produces a warning that names ``name``. Windows does not report those
    Unix bits, so the warning is skipped there.

    Raises:
        ValueError: if ``name`` is not a stored variable. The message does
        not echo ``name``.
    """
    path = user_key_path(home, name=name)
    keys = path.parent
    if keys.is_symlink() or path.is_symlink() or not _is_inside(keys.parent, keys):
        return None
    if not path.is_file():
        return None
    flags = os.O_RDONLY | _no_follow_flag()
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if os.name == "posix" and info.st_mode & 0o077:
            if name == OPENROUTER_API_KEY:
                warnings.warn(
                    "OpenRouter key file permissions are too loose",
                    OpenRouterKeyWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    f"{name} file permissions are too loose",
                    CredentialKeyWarning,
                    stacklevel=2,
                )
        chunks: list[bytes] = []
        while True:
            block = os.read(fd, 4096)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    text = b"".join(chunks).decode("utf-8").strip()
    if not text or "\n" in text or "\r" in text:
        return None
    return text


def prepare_credential(key: str, *, name: str) -> str:
    """Strip ``key`` for ``name`` and reject an empty, broken, or mistyped value.

    ``FIRMS_MAP_KEY`` must be 32 hexadecimal characters, the shape of a FIRMS
    area API map key. ``EARTHDATA_TOKEN`` must be a JWT of three base64url
    segments, the shape of an Earthdata Login user token.

    Raises:
        ValueError: if ``name`` is unknown, or the value is empty, contains a
        newline or carriage return, or does not match that variable. The
        message names ``name`` and does not include the value.
    """
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
    """Strip ``key`` and return ``(cleaned, warning)``.

    A value that is empty after stripping, or that still contains a newline
    or carriage return, raises ``ValueError``. A key that does not start
    with ``sk-or-`` is kept. When ``warn`` is true that shape also emits
    :class:`OpenRouterKeyWarning`.
    """
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
    """Persist a user key under ``~/.atlas/keys`` and return the path.

    ``home`` is the directory that contains ``.atlas``, not the project
    workspace. ``name`` selects the file. The keys directory is mode
    ``0700``. The file is replaced atomically: a temporary file is created
    with ``O_EXCL`` and mode ``0600``, fsynced, then renamed into place. The
    keys directory is fsynced after that rename so the new directory entry
    itself is durable. An existing file is chmodded to ``0600`` before that
    rename and is never opened with ``O_TRUNC``.

    A symlink at the key path or the keys directory is refused, and so is a
    keys directory that resolves outside the user root. The project tree is
    not written.

    On Windows the Unix modes are not enforced. ``chmod`` is still called,
    but the file ACL decides who can read the key, and symlinks are rejected
    with ``is_symlink`` because ``O_NOFOLLOW`` is not available. Windows also
    has no directory ``fsync``, so that durability step is skipped.

    Raises:
        ValueError: if ``name`` is unknown, or ``key`` is empty, contains a
        newline, or does not match that variable. The message names the
        variable and does not include the value.
    """
    if name == OPENROUTER_API_KEY:
        cleaned, _warning = prepare_openrouter_key(key)
    else:
        cleaned = prepare_credential(key, name=name)
    root = ensure_user_layout(home)
    keys = _require_keys_dir(root)
    path = keys / key_filename(name)
    _atomic_write_private(keys, path, cleaned + "\n")
    return path


def delete_user_key(home: Path | None = None, *, name: str = OPENROUTER_API_KEY) -> bool:
    """Remove the user key file for ``name``.

    Returns true when a file was removed. A missing file returns false. A
    symlink is refused and left in place.

    Raises:
        ValueError: if ``name`` is not a stored variable.
        AtlasPathError: if the path is a symlink or escapes the user root.
    """
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


def user_key_path(home: Path | None = None, *, name: str = OPENROUTER_API_KEY) -> Path:
    """Path of the user key file for ``name``.

    Raises:
        ValueError: if ``name`` is not a stored variable.
    """
    return user_keys_dir(home) / key_filename(name)


def resolve_credential(
    name: str,
    *,
    session_key: str | None = None,
    env_key: str | None = None,
    home: Path | None = None,
) -> str | None:
    """Choose the credential for ``name``.

    A non-empty session value wins over the environment, which wins over the
    user file. When ``session_key`` is omitted, the bound session map is
    used. When ``env_key`` is omitted, ``os.environ`` is used. Missing
    everywhere returns None.

    Raises:
        ValueError: if ``name`` is not a stored variable.
    """
    key_filename(name)
    if session_key is None:
        session_key = session_credential(name)
    if session_key is not None and session_key.strip():
        return _remember_resolved(session_key.strip())
    if env_key is None:
        env_key = os.environ.get(name)
    if env_key is not None and env_key.strip():
        return _remember_resolved(env_key.strip())
    return _remember_resolved(read_user_key(home, name=name))


def resolve_openrouter_key(
    *,
    session_key: str | None,
    env_key: str | None,
    home: Path | None = None,
) -> str | None:
    """Choose the OpenRouter key.

    A non-empty session key wins over the environment and over the user file.
    The environment wins over the user file. Missing everywhere returns None.
    """
    if session_key is not None and session_key.strip():
        return _remember_resolved(session_key.strip())
    if env_key is not None and env_key.strip():
        return _remember_resolved(env_key.strip())
    return _remember_resolved(read_user_key(home))


def _remember_resolved(value: str | None) -> str | None:
    """Remember a resolved credential and return it unchanged."""
    if value:
        remember_secret(value)
    return value


class _HttpxRedactionFilter(logging.Filter):
    """Drop FIRMS map keys and bound session secrets from httpx log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        rendered = record.getMessage()
        redacted = _redact_http_log(rendered)
        if redacted != rendered:
            record.msg = redacted
            record.args = None
        return True


def _redact_http_log(message: str) -> str:
    """Replace a FIRMS path key and any bound session secret."""
    redacted = _FIRMS_PATH_KEY.sub(r"\1***", message)
    secrets = _active_secrets
    if secrets is None:
        return redacted
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "***")
    return redacted


def _tighten_keys_dir(root: Path) -> None:
    """Set an existing real keys directory to mode ``0700``."""
    keys = root / "keys"
    if keys.is_symlink() or not keys.is_dir() or not _is_inside(root, keys):
        return
    _chmod(keys, 0o700)


def _require_keys_dir(root: Path) -> Path:
    """Return the user keys directory, refusing a symlink or an escaped path."""
    keys = root / "keys"
    if keys.is_symlink():
        raise AtlasPathError("refusing to follow a key symlink")
    if not keys.is_dir() or not _is_inside(root, keys):
        raise AtlasPathError("key path escapes the user atlas root")
    _chmod(keys, 0o700)
    return keys


def _no_follow_flag() -> int:
    """Return ``O_NOFOLLOW`` when this OS has it, otherwise zero.

    Windows has no ``O_NOFOLLOW``. Callers still reject a symlink with
    ``Path.is_symlink`` before they open.
    """
    flag = getattr(os, "O_NOFOLLOW", 0)
    if isinstance(flag, int):
        return flag
    return 0


def _chmod(path: Path, mode: int) -> None:
    """Request ``mode``. On Windows a failure is ignored; POSIX re-raises.

    Windows does not apply Unix permission bits. The call is still made so
    a POSIX host, including a Windows drive mounted that way, is tightened.
    """
    try:
        os.chmod(path, mode)
    except OSError:
        if os.name == "posix":
            raise


def _atomic_write_private(directory: Path, path: Path, text: str) -> None:
    """Replace ``path`` via a mode-``0600`` temporary file in ``directory``.

    An existing regular file is chmodded to ``0600`` before the rename so a
    loose file is never truncated in place. The temporary file is opened
    with ``O_EXCL`` and fsynced before ``os.replace``. The directory is
    fsynced after the rename on POSIX so the new name survives a crash.
    """
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
        payload = memoryview(text.encode("utf-8"))
        while payload:
            written = os.write(fd, payload)
            if written <= 0:
                raise OSError("could not write the key file")
            payload = payload[written:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        temp.unlink(missing_ok=True)
        raise
    os.close(fd)
    try:
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    _fsync_directory(directory)
    _chmod(path, 0o600)


def _fsync_directory(directory: Path) -> None:
    """Fsync ``directory`` so a rename is durable across a crash.

    Opening a directory and calling ``fsync`` is POSIX. Windows does not
    support it, so the call is skipped there. The key file is still fsynced
    before ``os.replace``.
    """
    if os.name != "posix":
        return
    flags = os.O_RDONLY
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if isinstance(directory_flag, int):
        flags |= directory_flag
    fd = os.open(directory, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _is_inside(root: Path, candidate: Path) -> bool:
    """Return whether ``candidate`` resolves inside ``root``."""
    try:
        root_resolved = root.resolve()
        candidate_resolved = candidate.resolve()
    except OSError:
        return False
    return candidate_resolved == root_resolved or root_resolved in candidate_resolved.parents


def _delete_matching_artifact(atlas_root: Path, name: str) -> None:
    artifact = _resolve_inside(atlas_root / "artifacts", name)
    if artifact.exists():
        _delete(artifact)


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
        return
    path.unlink()
