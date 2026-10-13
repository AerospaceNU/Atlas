"""Best-effort command check, with the OS sandbox as the real boundary.

The text scan refuses obvious paths, ``..``, command substitution, parameter
expansion, and a few interpreter-eval shapes. It does not see paths an
interpreter builds at runtime, and on Windows it is the only layer. On Linux
the child is placed under Landlock (kernel 5.13 or newer). The artifact root
and ``proposals/`` are masked out of that view with a user namespace, because
a Landlock rule cannot exclude a child of a granted directory. Hiding a
directory that already sits inside the workspace requires that namespace. If
it is unavailable the shell does not start. On macOS ``sandbox-exec`` is the
boundary; if it is
missing or cannot be applied, ``bash_tool`` does not run. The network is not
restricted on any platform.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from atlas.agent.artifacts import SandboxDenied
from atlas.agent.layout import atlas_home_override, user_keys_dir

_ESCAPE = "Shell path escapes the workspace"
_KEY = "User key directory is not readable"
_DRAFT = "Draft tool path is not executable"
_SUBSTITUTION = "Shell command substitution is not allowed"
_EXPANSION = "Shell parameter expansion is not allowed"
_GLOB = "Shell globbing is not allowed"
_HEREDOC = "Shell heredoc is not allowed"
_REJECTED = "Shell command is not allowed"
_EVAL = "Interpreter eval is not allowed; this check is best-effort"
_EXEC_OPEN = "Interpreter exec of a file is not allowed"
_UNAVAILABLE = "Shell sandbox is unavailable"
_KEY_INSIDE = "Shell cannot start because the user key directory is inside the workspace"
_ATLAS_HOME_INSIDE = "Shell cannot start because ATLAS_HOME is inside the workspace"
_KERNEL = "Landlock requires Linux 5.13 or newer"
MACOS_SANDBOX_DISABLED = "bash_tool is disabled because sandbox-exec is unavailable"
WINDOWS_BASH_WARNING = (
    "Warning: bash_tool has no OS confinement on Windows and is off unless it is opted in. "
    "The command check is best-effort only. The network is not restricted."
)

# Best-effort only. Combined short flags (-Ic), an env prefix, and a bare
# "-" (program on stdin) are included. A normal script path is not.
_INTERPRETER_EVAL = re.compile(
    r"(?:^|&&|\|\||[|;&(\n])\s*"
    r"(?:(?:[^\s|;&()]*/)?env\b"
    r"(?:\s+(?:-[A-Za-z0-9]+|[A-Za-z_][A-Za-z0-9_]*=\S+))*\s+)?"
    r"(?:python3?|perl|node)\b"
    r"(?:\s+\S+)*?"
    r"(?:\s+-[A-Za-z0-9]*[ce][A-Za-z0-9]*|\s+-)(?!\S)"
)
SANDBOX_EXEC_READY = "ATLAS_SANDBOX_READY"
SANDBOX_EXEC_FAILED = f"{_UNAVAILABLE}: sandbox-exec could not be applied"
_EXEC_OPEN_CALL = re.compile(r"\bexec\s*\(\s*open\s*\(")
_SANDBOX_PATH_CHARS = set('"\\\n()')

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

# python -I -c. argv[1] is the workspace, argv[2] the command, argv[3] the
# private TMPDIR, argv[4] an empty mask directory, and the rest are paths to
# hide. The script is stdlib-only so the child does not import Atlas before
# Landlock. Landlock needs Linux 5.13 or newer. The network is not restricted.
# A path-beneath rule covers every child, so the workspace root is granted
# only after a user-namespace bind mount hides those paths. There is no
# host-side rename fallback. ATLAS_SHELL_NO_USERNS=1 forces the unavailable
# path when a directory inside the workspace still has to be hidden.
_LAUNCHER = r"""
import ctypes
import os
import stat
import sys

_UNAVAILABLE = "Shell sandbox is unavailable"
_EXECUTE = 1 << 0
_WRITE_FILE = 1 << 1
_READ_FILE = 1 << 2
_READ_DIR = 1 << 3
_REMOVE_DIR = 1 << 4
_REMOVE_FILE = 1 << 5
_MAKE_CHAR = 1 << 6
_MAKE_DIR = 1 << 7
_MAKE_REG = 1 << 8
_MAKE_SOCK = 1 << 9
_MAKE_FIFO = 1 << 10
_MAKE_BLOCK = 1 << 11
_MAKE_SYM = 1 << 12
_REFER = 1 << 13
_TRUNCATE = 1 << 14
_IOCTL_DEV = 1 << 15
_ABI1 = (1 << 13) - 1
_READ_EXEC = _READ_FILE | _READ_DIR | _EXECUTE
_READ_ONLY = _READ_FILE | _READ_DIR
_DEV = _READ_FILE | _WRITE_FILE | _IOCTL_DEV
_FILE_ONLY = _EXECUTE | _WRITE_FILE | _READ_FILE | _TRUNCATE | _IOCTL_DEV
_SYS_LANDLOCK_CREATE = 444
_SYS_LANDLOCK_ADD = 445
_SYS_LANDLOCK_RESTRICT = 446
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_CREATE_VERSION = 1
_PR_SET_NO_NEW_PRIVS = 38
_CLONE_NEWUSER = 0x10000000
_CLONE_NEWNS = 0x00020000
_MS_BIND = 4096
_MS_REC = 16384
_MS_PRIVATE = 1 << 18
_DEVICES = (
    "/dev/null",
    "/dev/zero",
    "/dev/tty",
    "/dev/full",
    "/dev/urandom",
    "/dev/random",
)
_RUNTIME = ("/usr", "/bin", "/lib", "/lib64")
_CAPSET = {"x86_64": 126, "aarch64": 91, "armv7l": 91, "i386": 126, "i686": 126}


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int)]


class _CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def _fail(detail=""):
    sys.stderr.write(_UNAVAILABLE)
    if detail:
        sys.stderr.write(": " + detail)
    sys.stderr.write("\n")
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


def _add(libc, ruleset, path, access, handled):
    flags = os.O_PATH | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if isinstance(nofollow, int):
        flags |= nofollow
    fd = os.open(path, flags)
    try:
        mode = os.fstat(fd).st_mode
        if stat.S_ISLNK(mode):
            return False
        if stat.S_ISREG(mode):
            access &= _FILE_ONLY
        elif stat.S_ISCHR(mode):
            access &= _READ_FILE | _WRITE_FILE | _IOCTL_DEV
        elif not stat.S_ISDIR(mode):
            return False
        access &= handled
        if access == 0:
            return False
        attr = _PathAttr(access, fd)
        _syscall(
            libc,
            _SYS_LANDLOCK_ADD,
            ruleset,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )
        return True
    finally:
        os.close(fd)


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    libc.unshare.argtypes = [ctypes.c_int]
    libc.unshare.restype = ctypes.c_int
    libc.mount.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    libc.mount.restype = ctypes.c_int
    return libc


_NO_USERNS = "needs user namespaces or an artifact root outside the workspace"


def _apply(libc, workspace, tmpdir):
    try:
        abi = _syscall(libc, _SYS_LANDLOCK_CREATE, None, 0, _LANDLOCK_CREATE_VERSION)
    except OSError:
        _fail("Landlock requires Linux 5.13 or newer")
    if abi < 1:
        _fail("Landlock requires Linux 5.13 or newer")
    handled = _handled(abi)
    attr = _RulesetAttr(handled)
    try:
        ruleset = _syscall(
            libc, _SYS_LANDLOCK_CREATE, ctypes.byref(attr), ctypes.sizeof(attr), 0
        )
    except OSError:
        _fail("Landlock could not be applied")
    try:
        try:
            if not _add(libc, ruleset, workspace, handled, handled):
                _fail("Landlock could not be applied")
        except OSError:
            _fail("Landlock could not be applied")
        try:
            _add(libc, ruleset, tmpdir, handled, handled)
        except OSError:
            _fail("Landlock could not be applied")
        runtime_paths = list(_RUNTIME)
        runtime_paths.append(os.path.dirname(os.path.realpath(sys.executable)))
        for prefix in (getattr(sys, "prefix", ""), getattr(sys, "base_prefix", "")):
            if prefix:
                runtime_paths.append(prefix)
        seen = set()
        runtime = False
        for path in runtime_paths:
            if not path or path in seen or path == os.sep or not os.path.exists(path):
                continue
            seen.add(path)
            try:
                if _add(libc, ruleset, path, _READ_EXEC, handled):
                    runtime = True
            except OSError:
                continue
        if not runtime:
            _fail("Landlock could not be applied")
        if os.path.isdir("/etc"):
            try:
                _add(libc, ruleset, "/etc", _READ_ONLY, handled)
            except OSError:
                _fail("Landlock could not be applied")
        for path in _DEVICES:
            if not os.path.exists(path):
                continue
            try:
                _add(libc, ruleset, path, _DEV, handled)
            except OSError:
                continue
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            _fail("Landlock could not be applied")
        try:
            _syscall(libc, _SYS_LANDLOCK_RESTRICT, ruleset, 0)
        except OSError:
            _fail("Landlock could not be applied")
    finally:
        os.close(ruleset)


def _mask(libc, excluded, mask_dir):
    if libc.mount(b"", b"/", None, _MS_REC | _MS_PRIVATE, None) != 0:
        return False
    source = os.fsencode(mask_dir)
    for path in excluded:
        if not os.path.isdir(path) or os.path.islink(path):
            return False
        if libc.mount(source, os.fsencode(path), None, _MS_BIND, None) != 0:
            return False
    return True


def _drop_caps(libc):
    number = _CAPSET.get(os.uname().machine, 126)
    header = _CapHeader(0x20080522, 0)
    data = (_CapData * 2)()
    try:
        _syscall(libc, number, ctypes.byref(header), data)
    except OSError:
        _fail("Landlock could not be applied")


def _exec_bash(command):
    bash = "/bin/bash" if os.path.exists("/bin/bash") else "bash"
    os.execvpe(bash, [bash, "-c", command], os.environ)


def _confine(libc, workspace, command, tmpdir):
    os.environ["TMPDIR"] = tmpdir
    os.environ["TMP"] = tmpdir
    os.environ["TEMP"] = tmpdir
    _apply(libc, workspace, tmpdir)
    _exec_bash(command)


def _reap(pid):
    try:
        os.kill(pid, 9)
    except OSError:
        pass
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass


def _prepare(libc, workspace, command, tmpdir, mask_dir, excluded):
    notify_r, notify_w = os.pipe()
    go_r, go_w = os.pipe()
    done_r, done_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(notify_r)
        os.close(go_w)
        os.close(done_r)
        try:
            if libc.unshare(_CLONE_NEWUSER | _CLONE_NEWNS) != 0:
                os.write(notify_w, b"C")
                os.close(notify_w)
                os.close(go_r)
                os.close(done_w)
                os._exit(0)
            os.write(notify_w, b"R")
            os.close(notify_w)
            os.read(go_r, 1)
            os.close(go_r)
            if _mask(libc, excluded, mask_dir):
                os.write(done_w, b"K")
                os.close(done_w)
                _drop_caps(libc)
                _confine(libc, workspace, command, tmpdir)
            else:
                os.write(done_w, b"P")
                os.close(done_w)
                os._exit(0)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 126
            os._exit(code)
        except Exception:
            try:
                _fail("Landlock could not be applied")
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else 126
                os._exit(code)
        os._exit(126)
    os.close(notify_w)
    os.close(go_r)
    os.close(done_w)
    try:
        kind = os.read(notify_r, 1)
        if kind == b"C":
            os.waitpid(pid, 0)
            _fail(_NO_USERNS)
        if kind != b"R":
            _reap(pid)
            _fail("Landlock could not be applied")
        if not _write_id_maps(pid):
            _reap(pid)
            _fail(_NO_USERNS)
        os.write(go_w, b"G")
        outcome = os.read(done_r, 1)
        if outcome == b"P":
            os.waitpid(pid, 0)
            _fail(_NO_USERNS)
        if outcome != b"K":
            _reap(pid)
            _fail("Landlock could not be applied")
        _pid, status = os.waitpid(pid, 0)
        return status
    finally:
        os.close(notify_r)
        os.close(go_w)
        os.close(done_r)


def _write_id_maps(pid):
    uid = os.getuid()
    gid = os.getgid()
    try:
        with open(f"/proc/{pid}/setgroups", "w", encoding="ascii") as handle:
            handle.write("deny\n")
        with open(f"/proc/{pid}/uid_map", "w", encoding="ascii") as handle:
            handle.write(f"0 {uid} 1\n")
        with open(f"/proc/{pid}/gid_map", "w", encoding="ascii") as handle:
            handle.write(f"0 {gid} 1\n")
    except OSError:
        return False
    return True


def _forward(status):
    if os.WIFEXITED(status):
        os._exit(os.WEXITSTATUS(status))
    if os.WIFSIGNALED(status):
        os._exit(128 + os.WTERMSIG(status))
    os._exit(126)


def _run(workspace, command, tmpdir, mask_dir, excluded):
    libc = _libc()
    if not excluded:
        _confine(libc, workspace, command, tmpdir)
        return
    if os.environ.get("ATLAS_SHELL_NO_USERNS") == "1":
        _fail(_NO_USERNS)
    _forward(_prepare(libc, workspace, command, tmpdir, mask_dir, excluded))


def main():
    if len(sys.argv) < 5:
        _fail("Landlock could not be applied")
    workspace = sys.argv[1]
    command = sys.argv[2]
    tmpdir = sys.argv[3]
    mask_dir = sys.argv[4]
    excluded = [path for path in sys.argv[5:] if path]
    try:
        _run(workspace, command, tmpdir, mask_dir, excluded)
    except SystemExit:
        raise
    except Exception:
        _fail("Landlock could not be applied")


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


def bash_tool_trust(platform: str | None = None) -> Literal["default", "opt_in"]:
    """Return ``opt_in`` on Windows and ``default`` everywhere else.

    Windows has no OS confinement, so the shell stays out of the default tool
    list. The return value is a trust level (``default`` or ``opt_in``).
    """
    plat = sys.platform if platform is None else platform
    if plat == "win32":
        return "opt_in"
    return "default"


def kernel_release_supports_landlock(release: str) -> bool:
    """Return whether ``release`` is Linux 5.13 or newer.

    Landlock shipped in 5.13. A ``+`` or ``-`` suffix on the release is ignored.
    """
    head = release.split("-", 1)[0].rstrip("+")
    parts: list[int] = []
    for piece in head.split("."):
        digits = ""
        for char in piece:
            if char.isdigit():
                digits += char
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3]) >= (5, 13, 0)


def require_landlock_kernel(release: str | None = None) -> None:
    """Refuse to start on a Linux kernel older than 5.13.

    Non-Linux platforms return without checking. A missing Landlock syscall
    still fails closed in the child.
    """
    if sys.platform != "linux":
        return
    current = platform_release() if release is None else release
    if not kernel_release_supports_landlock(current):
        raise SandboxDenied(f"{_UNAVAILABLE}: {_KERNEL}")


def platform_release() -> str:
    """Return the kernel release string."""
    return os.uname().release


def refuse_shell_start(workspace: Path) -> None:
    """Refuse when the key directory or ``ATLAS_HOME`` resolves inside ``workspace``.

    Landlock cannot revoke a child of a directory it grants, and a key that
    lives in the workspace would be readable. The denial names neither the
    path nor the key.

    Raises:
        SandboxDenied: The shell must not start.
    """
    root = workspace.resolve()
    override = atlas_home_override()
    if override is not None and _contains(root, override):
        raise SandboxDenied(_ATLAS_HOME_INSIDE)
    home = Path(os.environ["HOME"]) if os.environ.get("HOME", "").strip() else None
    for key_dir in _key_directories(home):
        if _contains(root, key_dir):
            raise SandboxDenied(_KEY_INSIDE)


def shell_excluded_paths(workspace: Path, artifact_root: Path, proposals: Path) -> tuple[Path, ...]:
    """Return artifact and proposal paths that must stay out of the shell view.

    A path equal to ``workspace`` is omitted, because masking the workspace
    would hide the tree the shell is allowed to use. A proposal directory
    inside an excluded artifact root is omitted too; masking the artifact root
    already hides it. A path that does not exist yet is omitted: there is
    nothing to hide, and creating it would require a user namespace for an
    empty directory. Hiding a directory that does exist requires a user
    namespace. Without one the shell does not start.
    """
    root = workspace.resolve()
    chosen: list[Path] = []
    for path in (artifact_root.resolve(), proposals.resolve()):
        if _paths_equal(path, root) or not _contains(root, path):
            continue
        if not path.is_symlink() and not path.exists():
            continue
        if any(_contains(earlier, path) for earlier in chosen):
            continue
        chosen = [earlier for earlier in chosen if not _contains(path, earlier)]
        chosen.append(path)
    return tuple(chosen)


def prepare_mount_points(paths: Sequence[Path]) -> list[Path]:
    """Create missing mask targets and return the ones this call created.

    An existing symlink is not a mount point. Masking it would follow the
    link, so the shell fails closed instead.
    """
    created: list[Path] = []
    for path in paths:
        if path.is_symlink():
            raise SandboxDenied(f"{_UNAVAILABLE}: Landlock could not be applied")
        if path.exists():
            if not path.is_dir():
                raise SandboxDenied(f"{_UNAVAILABLE}: Landlock could not be applied")
            continue
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def cleanup_mount_points(created: Sequence[Path]) -> None:
    """Remove mask targets this process created, when they are still empty."""
    for path in reversed(created):
        try:
            path.rmdir()
        except OSError:
            continue


def sandboxed_shell_argv(
    workspace: Path,
    command: str,
    *,
    tmpdir: Path,
    mask_dir: Path | None = None,
    excluded: Sequence[Path] = (),
    key_dirs: Sequence[Path] = (),
    platform: str | None = None,
    sandbox_exec: str | None = None,
) -> list[str]:
    """Return argv that confines ``command`` to ``workspace`` and then runs bash.

    Linux uses the Landlock launcher. macOS uses ``sandbox-exec``. Windows
    runs bash with no OS confinement; the caller warns and keeps the tool
    opt-in. ``sandbox_exec`` overrides the lookup, which tests use.

    Raises:
        SandboxDenied: macOS has no ``sandbox-exec``, or a profile path is
            not safe to embed.
    """
    plat = sys.platform if platform is None else platform
    if plat == "linux":
        mask = "" if mask_dir is None else str(mask_dir)
        hidden = [str(path) for path in excluded]
        return [
            sys.executable,
            "-I",
            "-c",
            _LAUNCHER,
            str(workspace),
            command,
            str(tmpdir),
            mask,
            *hidden,
        ]
    if plat == "darwin":
        binary = sandbox_exec if sandbox_exec is not None else shutil.which("sandbox-exec")
        if not binary:
            raise SandboxDenied(MACOS_SANDBOX_DISABLED)
        profile = macos_sandbox_profile(
            workspace,
            excluded=excluded,
            key_dirs=key_dirs,
            tmpdir=tmpdir,
        )
        bash = "/bin/bash" if Path("/bin/bash").exists() else "bash"
        # The marker is printed only after sandbox-exec has started bash. A
        # launch or profile failure has no marker and is not a command result.
        marked = f"printf '%s\\n' '{SANDBOX_EXEC_READY}'; {command}"
        return [binary, "-p", profile, bash, "-c", marked]
    if plat == "win32":
        # which() follows the real OS. A caller can name Windows while the
        # process is elsewhere, and then which() looks for Win32 path rules.
        windows_bash = "bash"
        if sys.platform == "win32":
            windows_bash = shutil.which("bash") or "bash"
        return [windows_bash, "-c", command]
    bash = "/bin/bash" if Path("/bin/bash").exists() else "bash"
    return [bash, "-c", command]


def macos_sandbox_profile(
    workspace: Path,
    *,
    excluded: Sequence[Path],
    key_dirs: Sequence[Path],
    tmpdir: Path,
) -> str:
    """Return a ``sandbox-exec`` profile that keeps the shell in ``workspace``.

    The profile allows the toolchain and system read paths a command needs,
    denies the key directories and the excluded trees, and allows the network.
    Network access is not a confinement boundary.
    """
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow signal (target self))",
        "(allow sysctl-read)",
        "(allow mach-lookup)",
        "(allow network*)",
        f'(allow file-read* file-write* file-ioctl file-exec* (subpath "{_profile_path(workspace)}"))',
        f'(allow file-read* file-write* file-ioctl (subpath "{_profile_path(tmpdir)}"))',
    ]
    system_paths = (
        "/usr",
        "/bin",
        "/sbin",
        "/opt",
        "/opt/homebrew",
        "/usr/local",
        "/System",
        "/Library",
        "/dev",
        "/etc",
        "/private/etc",
    )
    for system_path in system_paths:
        lines.append(f'(allow file-read* file-exec* (subpath "{system_path}"))')
    for path in (*key_dirs, *excluded):
        lines.append(
            f'(deny file-read* file-write* file-ioctl file-exec* (subpath "{_profile_path(path)}"))'
        )
    return "\n".join(lines) + "\n"


def guard_shell_command(
    command: str,
    *,
    cwd: Path,
    workspace: Path,
    proposals: Path,
    home: Path | None,
) -> None:
    """Refuse a command the text scan can see leaving the workspace.

    This check is best-effort. It does not see a path an interpreter builds
    while the command runs. On Windows it is the only layer. ``cwd`` is the
    directory the shell starts in. ``proposals`` is the session proposal
    directory. ``home`` is the ``HOME`` value the child will see, used only
    to expand ``~``. Interpreter eval covers ``-c``/``-e`` including combined
    short flags, an ``env`` prefix, and a program read from stdin.

    Raises:
        SandboxDenied: The command is not safe to run. Key denials do not
            include the path or the file contents.
    """
    if "\x00" in command:
        raise SandboxDenied(_REJECTED)
    if _EXEC_OPEN_CALL.search(command):
        raise SandboxDenied(_EXEC_OPEN)
    if _INTERPRETER_EVAL.search(command):
        raise SandboxDenied(_EVAL)
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


def sandbox_unavailable(returncode: int | None, stderr: str | None) -> bool:
    """Return whether the child refused to run because confinement failed."""
    text = (stderr or "").strip()
    return returncode == 126 and (text == _UNAVAILABLE or text.startswith(f"{_UNAVAILABLE}:"))


def _profile_path(path: Path) -> str:
    text = path.resolve().as_posix()
    if any(char in text for char in _SANDBOX_PATH_CHARS):
        raise SandboxDenied(f"{_UNAVAILABLE}: sandbox profile path is not embeddable")
    return text


def _casefold_enabled() -> bool:
    """Return whether path checks should ignore case.

    Linux can hold both casings as different files. macOS and Windows
    filesystems usually cannot, and ``Path.relative_to`` stays case-sensitive
    there too.
    """
    return sys.platform in {"darwin", "win32"}


def _paths_equal(left: Path, right: Path) -> bool:
    if _casefold_enabled():
        return os.path.normcase(str(left)).casefold() == os.path.normcase(str(right)).casefold()
    return left == right


def _within_key(base: Path, candidate: Path) -> bool:
    if _casefold_enabled():
        base = Path(os.path.normcase(str(base)).casefold())
        candidate = Path(os.path.normcase(str(candidate)).casefold())
    try:
        candidate.relative_to(base)
    except ValueError:
        return False
    return True


def _contains(root: Path, path: Path) -> bool:
    """Return whether ``path`` resolves inside ``root``, including equality."""
    try:
        candidate = path.resolve()
        base = root.resolve()
    except (OSError, RuntimeError):
        return False
    return _within_key(base, candidate)


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
    if _within_key(root, lexical):
        return True
    return _is_within(proposals, path)


def _is_within(root: Path, path: Path) -> bool:
    return _contains(root, path)


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
