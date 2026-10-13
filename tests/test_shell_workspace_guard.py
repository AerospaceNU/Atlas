"""The shell stays in the workspace and cannot read user keys or run drafts."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.layout import write_user_key
from atlas.agent.shell_guard import (
    _LAUNCHER,
    MACOS_SANDBOX_DISABLED,
    SANDBOX_EXEC_FAILED,
    SANDBOX_EXEC_READY,
    WINDOWS_BASH_WARNING,
    _is_within,
    bash_tool_trust,
    guard_shell_command,
    kernel_release_supports_landlock,
    macos_sandbox_profile,
    sandboxed_shell_argv,
)
from atlas.agent.tools.BashTool import BashTool

_SECRET = "sk-or-shell-guard-secret-9f3c2a"


def _deny(command: str, store: LocalArtifactStore, **extra: object) -> str:
    arguments: dict[str, object] = {"command": command, **extra}
    with pytest.raises(SandboxDenied) as excinfo:
        BashTool().run(arguments, store)
    return str(excinfo.value)


def test_shell_cwd_is_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("at workspace", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "artifacts", read_root=workspace)

    result = BashTool().run({"command": "pwd"}, store)

    assert "exit_code: 0" in result.text
    assert str(workspace.resolve()) in result.text
    assert "at workspace" in BashTool().run({"command": "cat notes.txt"}, store).text


def test_shell_rejects_parent_and_absolute_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("outside-secret", encoding="utf-8")
    store = LocalArtifactStore(workspace)

    parent = _deny("cat ../secret.txt", store)
    absolute = _deny(f"cat {outside}", store)
    climbed = _deny("cd .. && cat secret.txt", store)

    assert "escapes" in parent
    assert "escapes" in absolute
    assert "escapes" in climbed
    assert "outside-secret" not in parent
    assert "outside-secret" not in absolute
    assert "outside-secret" not in climbed
    assert outside.read_text(encoding="utf-8") == "outside-secret"
    missing = tmp_path / "not-created"
    with pytest.raises(SandboxDenied):
        BashTool().run({"command": f"touch {missing}"}, store)
    assert not missing.exists()


def test_shell_allows_a_parent_segment_that_stays_inside_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "notes.txt").write_text("inside", encoding="utf-8")
    store = LocalArtifactStore(workspace)

    result = BashTool().run(
        {"command": "cat ../notes.txt", "working_directory": "sub"},
        store,
    )

    assert "exit_code: 0" in result.text
    assert "inside" in result.text


def test_shell_rejects_a_symlink_that_leaves_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("symlink-secret", encoding="utf-8")
    (workspace / "link").symlink_to(outside)
    nested = tmp_path / "nested.txt"
    nested.write_text("nested-secret", encoding="utf-8")
    (workspace / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    store = LocalArtifactStore(workspace)

    message = _deny("cat link", store)
    nested_message = _deny("cat dirlink/nested.txt", store)

    assert "escapes" in message
    assert "escapes" in nested_message
    assert "symlink-secret" not in message
    assert "nested-secret" not in nested_message


def test_shell_denies_the_user_key_file_without_returning_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ATLAS_HOME", raising=False)
    key_path = write_user_key(_SECRET, home)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "keylink").symlink_to(key_path)
    store = LocalArtifactStore(workspace)

    commands = (
        f"cat {key_path}",
        "cat ~/.atlas/keys/openrouter_api_key",
        "cat keylink",
        "cat $HOME/.atlas/keys/openrouter_api_key",
    )
    for command in commands:
        message = _deny(command, store)
        assert _SECRET not in message
        assert key_path.read_text(encoding="utf-8").strip() == _SECRET

    # The key directory stays denied when the workspace is the home directory,
    # so a relative path that resolves there is not treated as an ordinary file.
    home_store = LocalArtifactStore(home)
    message = _deny("cat .atlas/keys/openrouter_api_key", home_store)
    assert _SECRET not in message
    assert key_path.read_text(encoding="utf-8").strip() == _SECRET


def test_shell_rejects_a_hardlink_that_may_leave_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("hardlink-secret", encoding="utf-8")
    os.link(outside, workspace / "notes.txt")
    store = LocalArtifactStore(workspace)

    message = _deny("cat notes.txt", store)

    assert "hardlinked" in message
    assert "hardlink-secret" not in message
    assert outside.read_text(encoding="utf-8") == "hardlink-secret"


def test_shell_does_not_execute_a_drafted_tool(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / ".atlas" / "artifacts" / "session"
    proposals = artifact / "proposals"
    proposals.mkdir(parents=True)
    marker = workspace / "draft-ran.txt"
    script = proposals / "helper.py"
    script.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('RAN_DRAFT_TOOL\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    shell_script = proposals / "helper.sh"
    shell_script.write_text(
        f"#!/bin/sh\necho RAN_DRAFT_TOOL > {marker}\n",
        encoding="utf-8",
    )
    shell_script.chmod(0o755)
    store = LocalArtifactStore(artifact, read_root=workspace)
    relative = script.relative_to(workspace)
    shell_relative = shell_script.relative_to(workspace)

    for command in (
        f"python3 {relative}",
        f"python3 {script}",
        f"bash {shell_relative}",
        f"{shell_script}",
    ):
        message = _deny(command, store)
        assert "executable" in message
        assert "RAN_DRAFT_TOOL" not in message
        assert not marker.exists()

    artifact_store = LocalArtifactStore(artifact)
    for command in (
        "python3 proposals/helper.py",
        "./proposals/helper.sh",
        "sh proposals/helper.sh",
    ):
        message = _deny(command, artifact_store)
        assert "executable" in message
        assert not marker.exists()

    # A one-liner is refused by the best-effort scan. It is not the boundary.
    message = _deny(
        "python3 -c \"open('proposals/helper.py').read()\"",
        artifact_store,
    )
    assert "eval" in message
    assert not marker.exists()

    message = _deny(
        "echo hi",
        store,
        working_directory=os.fspath(proposals.relative_to(workspace)),
    )
    assert "executable" in message
    assert not marker.exists()


def _landlock_available() -> bool:
    if sys.platform != "linux":
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    rc = libc.syscall(444, None, 0, 1)
    return rc >= 1


def test_shell_refuses_to_start_when_the_key_directory_is_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ATLAS_HOME", raising=False)
    key_path = write_user_key(_SECRET, home)
    store = LocalArtifactStore(home)

    message = _deny("echo hi", store)

    assert "key directory is inside the workspace" in message
    assert _SECRET not in message
    assert str(key_path) not in message
    assert key_path.read_text(encoding="utf-8").strip() == _SECRET


def test_shell_refuses_to_start_when_atlas_home_is_inside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    atlas_home = workspace / "atlas-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ATLAS_HOME", str(atlas_home))
    key_path = write_user_key(_SECRET, home)
    store = LocalArtifactStore(workspace)

    message = _deny("echo hi", store)

    assert "ATLAS_HOME is inside the workspace" in message
    assert _SECRET not in message
    assert key_path.read_text(encoding="utf-8").strip() == _SECRET


@pytest.mark.skipif(not _landlock_available(), reason="Landlock is unavailable")
def test_linux_interpreter_cannot_read_a_key_outside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("ATLAS_HOME", raising=False)
    key_path = write_user_key(_SECRET, home)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "readkey.py").write_text(
        "import os\n"
        "s = chr(47)\n"
        "print(open(os.environ['HOME'] + s + '.atlas' + s + 'keys' + s + "
        "'openrouter_api_key').read())\n",
        encoding="utf-8",
    )
    store = LocalArtifactStore(workspace)

    result = BashTool().run({"command": "python3 readkey.py"}, store)

    assert _SECRET not in result.text
    assert "pyvenv.cfg" not in result.text
    assert "Permission" in result.text
    assert key_path.read_text(encoding="utf-8").strip() == _SECRET


@pytest.mark.skipif(not _landlock_available(), reason="Landlock is unavailable")
def test_linux_landlock_denies_an_interpreter_built_outside_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("landlock-secret\n", encoding="utf-8")
    (workspace / "readout.py").write_text(
        f"s = chr(47)\nparts = {[*outside.parts]!r}\nprint(open(s.join(parts)).read())\n",
        encoding="utf-8",
    )
    store = LocalArtifactStore(workspace)

    result = BashTool().run({"command": "python3 readout.py"}, store)

    assert "landlock-secret" not in result.text
    assert "pyvenv.cfg" not in result.text
    assert "Permission denied" in result.text
    assert outside.read_text(encoding="utf-8") == "landlock-secret\n"


@pytest.mark.skipif(not _landlock_available(), reason="Landlock is unavailable")
@pytest.mark.parametrize("force_park", [False, True])
def test_linux_runtime_symlink_cannot_read_or_write_outside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force_park: bool
) -> None:
    if force_park:
        monkeypatch.setenv("ATLAS_SHELL_NO_USERNS", "1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("toctou-secret\n", encoding="utf-8")
    (workspace / "plant.py").write_text(
        f"import sys\nimport os\nos.symlink({str(outside)!r}, sys.argv[1])\nprint('planted')\n",
        encoding="utf-8",
    )
    store = LocalArtifactStore(workspace)

    read = BashTool().run({"command": "python3 plant.py link && cat link"}, store)
    write = BashTool().run({"command": "python3 plant.py wlink; echo PWNED > wlink"}, store)

    assert "planted" in read.text
    assert "Permission denied" in read.text
    assert "toctou-secret" not in read.text
    assert (workspace / "link").is_symlink()
    assert os.readlink(workspace / "link") == str(outside)
    assert "Permission denied" in write.text
    assert (workspace / "wlink").is_symlink()
    assert "toctou-secret" not in write.text
    assert "PWNED" not in outside.read_text(encoding="utf-8")
    assert outside.read_text(encoding="utf-8") == "toctou-secret\n"


@pytest.mark.skipif(not _landlock_available(), reason="Landlock is unavailable")
def test_linux_cannot_run_proposals_hidden_from_the_text_scan(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    artifact = workspace / ".atlas" / "artifacts" / "session"
    proposals = artifact / "proposals"
    proposals.mkdir(parents=True)
    marker = workspace / "draft-ran.txt"
    (proposals / "helper.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('RAN_DRAFT_TOOL\\n', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (proposals / "helper.sh").write_text("echo RAN_DRAFT_TOOL\n", encoding="utf-8")
    (workspace / "run_draft.py").write_text(
        "exec(open('.atlas/' + 'artifacts/session/propo' + 'sals/helper.py').read())\n",
        encoding="utf-8",
    )
    store = LocalArtifactStore(artifact, read_root=workspace)

    executed = BashTool().run({"command": "python3 run_draft.py"}, store)
    # The proposal tree is under the artifact root, which the shell cannot see.
    # A relative proposals path from the workspace does not exist; run the copy
    # from the artifact root by using a store whose workspace is the artifact.
    artifact_store = LocalArtifactStore(artifact)
    (artifact / "copy_run.py").write_text(
        "import shutil\n"
        "from pathlib import Path\n"
        "shutil.copytree('proposals', 'copied')\n"
        "print(Path('copied/helper.sh').read_text(encoding='utf-8'))\n",
        encoding="utf-8",
    )
    copied = BashTool().run({"command": "python3 copy_run.py"}, artifact_store)

    assert "RAN_DRAFT_TOOL" not in executed.text
    assert "RAN_DRAFT_TOOL" not in copied.text
    assert not marker.exists()
    assert (proposals / "helper.py").is_file()


def test_shell_rejects_command_substitution_backticks_and_parameters(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = LocalArtifactStore(workspace)

    substitution = _deny("echo $(pwd)", store)
    backticks = _deny("echo `pwd`", store)
    parameter = _deny("echo $HOME", store)
    quoted = _deny('echo "$HOME"', store)

    assert "substitution" in substitution
    assert "substitution" in backticks
    assert "expansion" in parameter
    assert "expansion" in quoted


def test_scanner_flags_interpreter_eval_and_exec_open(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    proposals = workspace / "proposals"
    store = LocalArtifactStore(workspace)

    python_c = _deny("python3 -c 'print(1)'", store)
    combined = _deny("python3 -Ic 'print(1)'", store)
    env_prefix = _deny("env python3 -c 'print(1)'", store)
    env_flag = _deny("env -i python3 -c 'print(1)'", store)
    stdin_prog = _deny("printf code | python3 -", store)
    perl_e = _deny("perl -ne 'print 1'", store)
    node_e = _deny("node -e 'console.log(1)'", store)
    exec_open = _deny("python3 -c 'exec(open(\"x.py\").read())'", store)

    assert "best-effort" in python_c
    assert "best-effort" in combined
    assert "best-effort" in env_prefix
    assert "best-effort" in env_flag
    assert "best-effort" in stdin_prog
    assert "best-effort" in perl_e
    assert "best-effort" in node_e
    assert "exec" in exec_open
    for command in ("python3 ok.py", "python3 -m mod", "env python3 ok.py", "echo python3 -c"):
        guard_shell_command(
            command,
            cwd=workspace,
            workspace=workspace,
            proposals=proposals,
            home=None,
        )


@pytest.mark.skipif(sys.platform != "linux", reason="Landlock is a Linux syscall")
def test_landlock_syscall_failure_exits_without_running_the_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    ran = workspace / "ran.txt"
    broken = _LAUNCHER.replace("_SYS_LANDLOCK_CREATE = 444", "_SYS_LANDLOCK_CREATE = 999", 1)
    tmpdir = workspace / "tmp"
    tmpdir.mkdir()
    mask = tmp_path / "mask"
    mask.mkdir()
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            broken,
            str(workspace),
            "echo RAN > ran.txt",
            str(tmpdir),
            str(mask),
        ],
        cwd=workspace,
        env={"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "")},
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert proc.returncode == 126
    assert proc.stderr.strip().startswith("Shell sandbox is unavailable")
    assert "Traceback" not in proc.stderr
    assert not ran.exists()


def test_kernel_release_gate_requires_linux_5_13() -> None:
    assert kernel_release_supports_landlock("5.13")
    assert kernel_release_supports_landlock("5.13.0-generic")
    assert kernel_release_supports_landlock("6.12.94+")
    assert not kernel_release_supports_landlock("5.12.9")
    assert not kernel_release_supports_landlock("4.19.0")


def test_darwin_profile_denies_keys_and_confines_the_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    key_dir = tmp_path / "home" / ".atlas" / "keys"
    proposals = workspace / "proposals"
    profile = macos_sandbox_profile(
        workspace,
        excluded=(proposals,),
        key_dirs=(key_dir,),
        tmpdir=workspace / ".shell-tmp",
    )
    argv = sandboxed_shell_argv(
        workspace,
        "echo hi",
        tmpdir=workspace / ".shell-tmp",
        excluded=(proposals,),
        key_dirs=(key_dir,),
        platform="darwin",
        sandbox_exec="/usr/bin/sandbox-exec",
    )

    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[1] == "-p"
    assert SANDBOX_EXEC_READY in argv[-1]
    assert argv[-1].endswith("; echo hi")
    assert "(allow network*)" in profile
    assert f'(subpath "{workspace.resolve().as_posix()}")' in profile
    assert f'(subpath "{key_dir.resolve().as_posix()}")' in profile
    assert profile.index("(deny file-read*") > profile.index("(allow file-read* file-write*")
    assert "(deny default)" in argv[2]


def test_darwin_disables_bash_when_sandbox_exec_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = LocalArtifactStore(workspace)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("atlas.agent.tools.BashTool.shutil.which", lambda _name: None)

    message = _deny("echo hi", store)

    assert message == MACOS_SANDBOX_DISABLED


def _sandbox_exec_which(binary: str):
    real_which = shutil.which

    def which(name: str) -> str | None:
        if name == "sandbox-exec":
            return binary
        return real_which(name)

    return which


@pytest.mark.skipif(sys.platform == "win32", reason="the fake sandbox-exec is a shell script")
def test_darwin_sandbox_exec_failure_is_a_sandbox_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake = tmp_path / "sandbox-exec"
    fake.write_text("#!/bin/sh\necho 'sandbox-exec: denied' >&2\nexit 64\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(sys, "platform", "darwin")
    which = _sandbox_exec_which(str(fake))
    monkeypatch.setattr("atlas.agent.tools.BashTool.shutil.which", which)
    monkeypatch.setattr("atlas.agent.shell_guard.shutil.which", which)
    store = LocalArtifactStore(workspace)

    message = _deny("echo hi", store)

    assert message == SANDBOX_EXEC_FAILED
    assert "exit_code" not in message


@pytest.mark.skipif(sys.platform == "win32", reason="the fake sandbox-exec is a shell script")
def test_darwin_sandbox_exec_launch_failure_is_a_sandbox_denial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(sys, "platform", "darwin")
    which = _sandbox_exec_which(str(tmp_path / "missing-sandbox-exec"))
    monkeypatch.setattr("atlas.agent.tools.BashTool.shutil.which", which)
    monkeypatch.setattr("atlas.agent.shell_guard.shutil.which", which)

    message = _deny("echo hi", LocalArtifactStore(workspace))

    assert message == SANDBOX_EXEC_FAILED


@pytest.mark.skipif(not Path("/bin/bash").exists(), reason="bash is required to exec the profile")
def test_darwin_command_failure_stays_a_command_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake = tmp_path / "sandbox-exec"
    fake.write_text('#!/bin/sh\nshift 2\nexec "$@"\n', encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(sys, "platform", "darwin")
    which = _sandbox_exec_which(str(fake))
    monkeypatch.setattr("atlas.agent.tools.BashTool.shutil.which", which)
    monkeypatch.setattr("atlas.agent.shell_guard.shutil.which", which)
    store = LocalArtifactStore(workspace)

    result = BashTool().run({"command": "echo hi && false"}, store)

    assert "exit_code: 1" in result.text
    assert "hi" in result.text
    assert SANDBOX_EXEC_READY not in result.text


def test_windows_bash_tool_is_opt_in_and_warns(tmp_path: Path) -> None:
    assert bash_tool_trust("win32") == "opt_in"
    assert bash_tool_trust("linux") == "default"
    assert bash_tool_trust("darwin") == "default"
    assert "no OS confinement" in WINDOWS_BASH_WARNING
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    argv = sandboxed_shell_argv(
        workspace,
        "echo hi",
        tmpdir=workspace / ".shell-tmp",
        platform="win32",
    )

    assert argv[0] == "bash"
    assert argv[1:] == ["-c", "echo hi"]
    assert "sandbox-exec" not in argv[0]


@pytest.mark.skipif(not _landlock_available(), reason="Landlock is unavailable")
def test_linux_without_user_namespaces_can_create_at_the_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ATLAS_SHELL_NO_USERNS", "1")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    proposals = workspace / "proposals"
    proposals.mkdir()
    secret = proposals / "helper.py"
    secret.write_text("SECRET_DRAFT\n", encoding="utf-8")
    (workspace / "peek.py").write_text(
        "from pathlib import Path\n"
        "try:\n"
        "    print(Path('propo' + 'sals/helper.py').read_text(encoding='utf-8'))\n"
        "except OSError as exc:\n"
        "    print(type(exc).__name__)\n",
        encoding="utf-8",
    )
    store = LocalArtifactStore(workspace)

    result = BashTool().run(
        {"command": "echo hi > out.txt && mkdir d && ln -s out.txt linkname && python3 peek.py"},
        store,
    )

    assert "exit_code: 0" in result.text
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "hi\n"
    assert (workspace / "d").is_dir()
    assert (workspace / "linkname").is_symlink()
    assert "SECRET_DRAFT" not in result.text
    assert secret.read_text(encoding="utf-8") == "SECRET_DRAFT\n"
    assert not (workspace / "proposals.shell-displaced").exists()


@pytest.mark.skipif(not _landlock_available(), reason="Landlock is unavailable")
def test_linux_unshare_failure_can_create_at_the_workspace_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "atlas.agent.shell_guard._LAUNCHER",
        _LAUNCHER.replace(
            "if libc.unshare(_CLONE_NEWUSER | _CLONE_NEWNS) != 0:",
            "if True:",
            1,
        ),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    proposals = workspace / "proposals"
    proposals.mkdir()
    secret = proposals / "helper.py"
    secret.write_text("SECRET_DRAFT\n", encoding="utf-8")
    (workspace / "peek.py").write_text(
        "from pathlib import Path\n"
        "try:\n"
        "    print(Path('propo' + 'sals/helper.py').read_text(encoding='utf-8'))\n"
        "except OSError as exc:\n"
        "    print(type(exc).__name__)\n",
        encoding="utf-8",
    )
    store = LocalArtifactStore(workspace)

    result = BashTool().run({"command": "echo hi > out.txt && python3 peek.py"}, store)

    assert "exit_code: 0" in result.text
    assert (workspace / "out.txt").read_text(encoding="utf-8") == "hi\n"
    assert "SECRET_DRAFT" not in result.text
    assert "FileNotFoundError" in result.text
    assert secret.read_text(encoding="utf-8") == "SECRET_DRAFT\n"


@pytest.mark.skipif(sys.platform != "linux", reason="the launcher parks on Linux")
def test_park_inside_the_workspace_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    proposals = workspace / "proposals"
    proposals.mkdir()
    secret = proposals / "helper.py"
    secret.write_text("SECRET_DRAFT\n", encoding="utf-8")
    tmpdir = workspace / "tmp"
    tmpdir.mkdir()
    mask = workspace / "mask"
    mask.mkdir()
    ran = workspace / "out.txt"
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            _LAUNCHER,
            str(workspace),
            "echo hi > out.txt",
            str(tmpdir),
            str(mask),
            str(proposals),
        ],
        cwd=workspace,
        env={
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", ""),
            "ATLAS_SHELL_NO_USERNS": "1",
        },
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert proc.returncode == 126
    assert (
        proc.stderr.strip()
        == "Shell sandbox is unavailable: needs user namespaces or an artifact root outside the workspace"
    )
    assert "Traceback" not in proc.stderr
    assert not ran.exists()
    assert secret.read_text(encoding="utf-8") == "SECRET_DRAFT\n"


def test_containment_folds_case_only_on_case_insensitive_platforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path("/var/Workspace")
    inside = Path("/var/workspace/notes.txt")
    monkeypatch.setattr(sys, "platform", "linux")
    assert not _is_within(root, inside)
    monkeypatch.setattr(sys, "platform", "darwin")
    assert _is_within(root, inside)
    monkeypatch.setattr(sys, "platform", "win32")
    assert _is_within(root, inside)
