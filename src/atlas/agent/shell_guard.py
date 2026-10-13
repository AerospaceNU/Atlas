"""Fail closed when a shell command leaves the workspace or reads a user key.

The file tools already resolve paths through :class:`~atlas.agent.artifacts.LocalArtifactStore`.
``bash_tool`` starts in that same workspace and, before it launches, every path
the command names is resolved the same way. Absolute paths and ``..`` that
leave the workspace are refused. A symlink is judged by its target, so a link
inside the workspace cannot open a host file or ``~/.atlas/keys``. The session
``proposals/`` directory is not a runnable path. On Linux the child is also
placed under Landlock, so a path the check misses still cannot open a host file.
"""

from __future__ import annotations

import contextlib
import re
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

from atlas.agent.artifacts import SandboxDenied
from atlas.agent.layout import user_keys_dir

_ESCAPE = "Shell path escapes the workspace"
_KEY = "User key directory is not readable"
_DRAFT = "Draft tool path is not executable"
_SUBSTITUTION = "Shell command substitution is not allowed"
_EXPANSION = "Shell parameter expansion is not allowed"
_GLOB = "Shell globbing is not allowed"
_HEREDOC = "Shell heredoc is not allowed"
_REJECTED = "Shell command is not allowed"
_UNAVAILABLE = "Shell sandbox is unavailable"

_SEPARATORS = frozenset({"&&", "||", "|", ";", "&", "\n"})
_PERSISTENT_CD = frozenset({"&&", "||", ";", "\n"})
_REDIRECT_OPS = ("&>>", "&>", ">>", ">&", ">|", "<>", "<<<", "<<", "<&", ">", "<")
_ALLOWED_DEVICES = frozenset(
    {
        "/dev/null",
        "/dev/zero",
        "/dev/tty",
        "/dev/full",
        "/dev/urandom",
        "/dev/random",
        "/dev/stdin",
        "/dev/stdout",
        "/dev/stderr",
    }
)
_PATH_IN_WORD = re.compile(
    r"(?:~/?[^\s'\"|;<>&()$]*|/[^\s'\"|;<>&()$]*|\.\./[^\s'\"|;<>&()$]*|"
    r"\./[^\s'\"|;<>&()$]*|(?:[\w.+@-]+/)+[\w.+@-]*)"
)

# python -I -c. argv[1] is the workspace, argv[2] is the original command.
# The script is stdlib-only so the child does not import Atlas before Landlock.
_LAUNCHER = r"""
import ctypes
import os
import sys

_UNAVAILABLE = "Shell sandbox is unavailable\n"
_EXECUTE = 1 << 0
_WRITE_FILE = 1 << 1
_READ_FILE = 1 << 2
_READ_DIR = 1 << 3
_REMOVE_DIR = 1 << 4
_REMOVE_FILE = 1 << 5
_MAKE_DIR = 1 << 7
_MAKE_REG = 1 << 8
_MAKE_SYM = 1 << 12
_REFER = 1 << 13
_TRUNCATE = 1 << 14
_IOCTL_DEV = 1 << 15
_ABI1 = (1 << 13) - 1
_READ_EXEC = _READ_FILE | _READ_DIR | _EXECUTE
_DEV = _READ_FILE | _WRITE_FILE
_SYS_LANDLOCK_CREATE = 444
_SYS_LANDLOCK_ADD = 445
_SYS_LANDLOCK_RESTRICT = 446
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_CREATE_VERSION = 1
_PR_SET_NO_NEW_PRIVS = 38
_DEVICES = (
    "/dev/null",
    "/dev/zero",
    "/dev/tty",
    "/dev/full",
    "/dev/urandom",
    "/dev/random",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
)
_RUNTIME = ("/usr", "/bin", "/lib", "/lib64")


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathAttr(ctypes.Structure):
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


def _fail() -> None:
    sys.stderr.write(_UNAVAILABLE)
    raise SystemExit(126)


def _syscall(libc, number, *args):
    rc = libc.syscall(number, *args)
    if rc < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return rc


def _handled(abi):
    access = _ABI1
    if abi >= 2:
        access |= _REFER
    if abi >= 3:
        access |= _TRUNCATE
    if abi >= 5:
        access |= _IOCTL_DEV
    return access


def _add(libc, ruleset, path, access):
    flags = os.O_PATH | os.O_CLOEXEC
    fd = os.open(path, flags)
    attr = _PathAttr(access, fd)
    try:
        _syscall(libc, _SYS_LANDLOCK_ADD, ruleset, _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(attr), 0)
    finally:
        os.close(fd)


def _confine(workspace):
    if sys.platform != "linux":
        return
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    try:
        abi = _syscall(libc, _SYS_LANDLOCK_CREATE, None, 0, _LANDLOCK_CREATE_VERSION)
    except OSError:
        _fail()
    if abi < 1:
        _fail()
    handled = _handled(abi)
    write = (
        _READ_FILE
        | _READ_DIR
        | _WRITE_FILE
        | _REMOVE_DIR
        | _REMOVE_FILE
        | _MAKE_DIR
        | _MAKE_REG
        | _MAKE_SYM
        | _REFER
        | _TRUNCATE
    ) & handled
    read_exec = _READ_EXEC & handled
    dev_access = _DEV & handled
    attr = _RulesetAttr(handled)
    try:
        ruleset = _syscall(
            libc, _SYS_LANDLOCK_CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0
        )
    except OSError:
        _fail()
    try:
        try:
            _add(libc, ruleset, workspace, write)
        except OSError:
            _fail()
        runtime = False
        for path in _RUNTIME:
            if not os.path.exists(path):
                continue
            try:
                _add(libc, ruleset, path, read_exec)
            except OSError:
                continue
            runtime = True
        if not runtime:
            _fail()
        for path in _DEVICES:
            if not os.path.exists(path):
                continue
            try:
                _add(libc, ruleset, path, dev_access)
            except OSError:
                continue
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            _fail()
        _syscall(libc, _SYS_LANDLOCK_RESTRICT, ruleset, 0)
    finally:
        os.close(ruleset)


def main():
    if len(sys.argv) != 3:
        _fail()
    workspace = sys.argv[1]
    command = sys.argv[2]
    try:
        _confine(workspace)
        bash = "/bin/bash" if os.path.exists("/bin/bash") else "bash"
        os.execvpe(bash, [bash, "-c", command], os.environ)
    except SystemExit:
        raise
    except Exception:
        _fail()


main()
"""


class _Word:
    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text


class _Redirect:
    __slots__ = ("target",)

    def __init__(self, target: str | None) -> None:
        self.target = target


def guard_shell_command(
    command: str,
    *,
    cwd: Path,
    workspace: Path,
    proposals: Path,
    home: Path | None,
) -> None:
    """Refuse a command that leaves ``workspace``, reads a key, or runs a draft.

    ``cwd`` is the directory the shell starts in. ``proposals`` is the session
    proposal directory (``<artifact root>/proposals``). ``home`` is the ``HOME``
    value the child will see, used only to expand ``~``.

    Raises:
        SandboxDenied: The command is not safe to run. Key denials do not
            include the path or the file contents.
    """
    if "\x00" in command:
        raise SandboxDenied(_REJECTED)
    root = workspace.resolve()
    start = cwd.resolve()
    if not _is_within(root, start):
        raise SandboxDenied(_ESCAPE)
    keys = _key_directories(home)
    _reject_cwd(start, keys, proposals)
    logical = start
    for segment, separator in _segments(_lex(command)):
        words, redirects = _parse(segment)
        for target in redirects:
            _check_reference(target, logical, root, keys, proposals, home)
        if not words:
            continue
        if words[0] == "cd":
            logical = _apply_cd(words, logical, root, keys, proposals, home, separator)
            continue
        for word in words:
            _check_word(word, logical, root, keys, proposals, home)


def sandboxed_shell_argv(workspace: Path, command: str) -> list[str]:
    """Return the argv of a child that confines itself to ``workspace`` then execs bash."""
    return [sys.executable, "-I", "-c", _LAUNCHER, str(workspace), command]


def sandbox_unavailable(returncode: int | None, stderr: str | None) -> bool:
    """Return whether the child refused to run because confinement failed."""
    return returncode == 126 and (stderr or "").strip() == _UNAVAILABLE


def _key_directories(home: Path | None) -> tuple[Path, ...]:
    found: list[Path] = [user_keys_dir()]
    if home is not None:
        found.append(home / ".atlas" / "keys")
    with contextlib.suppress(RuntimeError):
        found.append(Path.home() / ".atlas" / "keys")
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        marker = str(path)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(path)
    return tuple(unique)


def _reject_cwd(cwd: Path, keys: Sequence[Path], proposals: Path) -> None:
    if _is_key_path(cwd, keys):
        raise SandboxDenied(_KEY)
    if _under_proposals(cwd, proposals):
        raise SandboxDenied(_DRAFT)


def _apply_cd(
    words: list[str],
    cwd: Path,
    workspace: Path,
    keys: Sequence[Path],
    proposals: Path,
    home: Path | None,
    separator: str | None,
) -> Path:
    args = words[1:]
    if not args or any(arg.startswith("-") for arg in args):
        raise SandboxDenied(_ESCAPE)
    target = args[-1]
    _check_reference(target, cwd, workspace, keys, proposals, home)
    resolved = _resolve(target, cwd, home)
    if _is_key_path(resolved, keys):
        raise SandboxDenied(_KEY)
    if not resolved.is_dir():
        return cwd
    if separator in _PERSISTENT_CD or separator is None:
        return resolved
    return cwd


def _check_word(
    word: str,
    cwd: Path,
    workspace: Path,
    keys: Sequence[Path],
    proposals: Path,
    home: Path | None,
) -> None:
    if "*" in word or "?" in word:
        raise SandboxDenied(_GLOB)
    expanded = _expand_tilde(word, home)
    candidates: list[str] = []
    if _looks_like_path(expanded) or _names_entry(cwd, expanded):
        candidates.append(expanded)
    for match in _PATH_IN_WORD.finditer(expanded):
        text = match.group(0)
        if text and text not in candidates:
            candidates.append(text)
    for candidate in candidates:
        _check_reference(candidate, cwd, workspace, keys, proposals, home)


def _check_reference(
    raw: str,
    cwd: Path,
    workspace: Path,
    keys: Sequence[Path],
    proposals: Path,
    home: Path | None,
) -> None:
    expanded = _expand_tilde(raw, home)
    if _looks_like_key(expanded):
        raise SandboxDenied(_KEY)
    path = Path(expanded)
    if not path.is_absolute():
        path = cwd / path
    if _is_key_path(path, keys):
        raise SandboxDenied(_KEY)
    resolved = _resolve_existing(path)
    if _is_key_path(resolved, keys) or _looks_like_key(str(resolved)):
        raise SandboxDenied(_KEY)
    if _is_allowed_device(resolved):
        return
    if not _is_within(workspace, resolved):
        raise SandboxDenied(_ESCAPE)
    if _under_proposals(path, proposals) or _under_proposals(resolved, proposals):
        raise SandboxDenied(_DRAFT)
    _reject_hardlink(resolved)


def _resolve(raw: str, cwd: Path, home: Path | None) -> Path:
    expanded = _expand_tilde(raw, home)
    path = Path(expanded)
    if not path.is_absolute():
        path = cwd / path
    return _resolve_existing(path)


def _resolve_existing(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        raise SandboxDenied(_ESCAPE) from None


def _expand_tilde(token: str, home: Path | None) -> str:
    if token == "~" or token.startswith("~/"):
        if home is None:
            raise SandboxDenied(_ESCAPE)
        suffix = token[1:]
        return str(home) + suffix
    if token.startswith("~"):
        raise SandboxDenied(_ESCAPE)
    return token


def _looks_like_path(token: str) -> bool:
    if token.startswith("/") or token.startswith("~") or "/" in token or "\\" in token:
        return True
    return ".." in Path(token).parts


def _looks_like_key(token: str) -> bool:
    parts = Path(token).parts
    return any(
        parts[index] == ".atlas" and parts[index + 1] == "keys" for index in range(len(parts) - 1)
    )


def _names_entry(cwd: Path, token: str) -> bool:
    if not token or "/" in token or token.startswith("~"):
        return False
    candidate = cwd / token
    try:
        return candidate.is_symlink() or candidate.exists()
    except OSError:
        return False


def _is_key_path(path: Path, keys: Sequence[Path]) -> bool:
    if _looks_like_key(str(path)):
        return True
    return any(_is_within(root, path) for root in keys)


def _under_proposals(path: Path, proposals: Path) -> bool:
    lexical = path if path.is_absolute() else path.absolute()
    root = proposals if proposals.is_absolute() else proposals.absolute()
    try:
        lexical.relative_to(root)
        return True
    except ValueError:
        return _is_within(proposals, path)


def _is_within(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (ValueError, OSError, RuntimeError):
        return False
    return True


def _is_allowed_device(path: Path) -> bool:
    try:
        return path.resolve().as_posix() in _ALLOWED_DEVICES
    except (OSError, RuntimeError):
        return False


def _reject_hardlink(path: Path) -> None:
    try:
        info = path.stat()
    except OSError:
        return
    if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
        raise SandboxDenied("Shell path is a hardlinked file that may lead outside the workspace")


def _lex(command: str) -> list[_Word | _Redirect | str]:
    tokens: list[_Word | _Redirect | str] = []
    index = 0
    length = len(command)
    while index < length:
        char = command[index]
        if char in " \t":
            index += 1
            continue
        if char == "\n":
            tokens.append("\n")
            index += 1
            continue
        if command.startswith("&&", index):
            tokens.append("&&")
            index += 2
            continue
        if command.startswith("||", index):
            tokens.append("||")
            index += 2
            continue
        if command.startswith("&>>", index) or command.startswith("&>", index):
            token, index = _read_redirect(command, index)
            tokens.append(token)
            continue
        if _redirect_at(command, index):
            index = _skip_fd(command, index)
            token, index = _read_redirect(command, index)
            tokens.append(token)
            continue
        if char in "|&;()":
            tokens.append(char)
            index += 1
            continue
        word, index = _read_word(command, index)
        tokens.append(_Word(word))
    return tokens


def _redirect_at(command: str, index: int) -> bool:
    cursor = _skip_fd(command, index)
    return cursor < len(command) and command[cursor] in "<>"


def _skip_fd(command: str, index: int) -> int:
    cursor = index
    while cursor < len(command) and command[cursor].isdigit():
        cursor += 1
    if cursor != index and cursor < len(command) and command[cursor] in "<>":
        return cursor
    return index


def _read_redirect(command: str, index: int) -> tuple[_Redirect, int]:
    op = ""
    for candidate in _REDIRECT_OPS:
        if command.startswith(candidate, index):
            op = candidate
            index += len(candidate)
            break
    if not op:
        raise SandboxDenied(_REJECTED)
    if op in {"<<", "<<<"}:
        raise SandboxDenied(_HEREDOC)
    while index < len(command) and command[index] in " \t":
        index += 1
    if index >= len(command) or command[index] in "\n|&;()<>":
        raise SandboxDenied(_REJECTED)
    target, index = _read_word(command, index)
    if not target:
        raise SandboxDenied(_REJECTED)
    if target.isdigit() and "&" in op:
        return _Redirect(None), index
    return _Redirect(target), index


def _read_word(command: str, index: int) -> tuple[str, int]:
    parts: list[str] = []
    length = len(command)
    while index < length:
        char = command[index]
        if char in " \t\n|&;()<>":
            break
        if char == "\\":
            if index + 1 >= length:
                raise SandboxDenied(_REJECTED)
            parts.append(command[index + 1])
            index += 2
            continue
        if char == "'":
            end = command.find("'", index + 1)
            if end < 0:
                raise SandboxDenied(_REJECTED)
            parts.append(command[index + 1 : end])
            index = end + 1
            continue
        if char == '"':
            index += 1
            while index < length and command[index] != '"':
                if command[index] == "\\" and index + 1 < length:
                    parts.append(command[index + 1])
                    index += 2
                    continue
                if command[index] == "`" or command.startswith("$(", index):
                    raise SandboxDenied(_SUBSTITUTION)
                if command[index] == "$":
                    raise SandboxDenied(_EXPANSION)
                parts.append(command[index])
                index += 1
            if index >= length or command[index] != '"':
                raise SandboxDenied(_REJECTED)
            index += 1
            continue
        if char == "`" or command.startswith("$(", index):
            raise SandboxDenied(_SUBSTITUTION)
        if command.startswith("<(", index) or command.startswith(">(", index):
            raise SandboxDenied(_SUBSTITUTION)
        if char == "$":
            raise SandboxDenied(_EXPANSION)
        parts.append(char)
        index += 1
    return "".join(parts), index


def _segments(
    tokens: Sequence[_Word | _Redirect | str],
) -> list[tuple[list[_Word | _Redirect | str], str | None]]:
    segments: list[tuple[list[_Word | _Redirect | str], str | None]] = []
    current: list[_Word | _Redirect | str] = []
    for token in tokens:
        if isinstance(token, str) and token in _SEPARATORS:
            segments.append((current, token))
            current = []
            continue
        current.append(token)
    segments.append((current, None))
    return segments


def _parse(segment: Sequence[_Word | _Redirect | str]) -> tuple[list[str], list[str]]:
    words: list[str] = []
    redirects: list[str] = []
    for token in segment:
        if isinstance(token, _Word):
            words.append(token.text)
        elif isinstance(token, _Redirect) and token.target is not None:
            redirects.append(token.target)
    return words, redirects
