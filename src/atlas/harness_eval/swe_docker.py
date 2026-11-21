"""Host side of a SWE-bench rollout: one task container per run.

Each task runs inside its official SWE-bench instance image, which holds the
repository at the base commit and an environment that can run its tests. The
agent is not installed in that image. A Linux Atlas runtime lives in a Docker
volume (:data:`RUNTIME_VOLUME`) and the package source is bind-mounted
read-only, so every image runs the working tree's Atlas without a rebuild.

The container holds the issue text and the harness tree, never the task's
tests or reference patch. It sits on a Docker network with no route out, and
its only way to the internet is an allowlisting proxy
(:mod:`atlas.harness_eval.egress_proxy`) that reaches the model provider and
nothing else. A package index or a code host would hand the agent the upstream
fix, which is what an unrestricted rollout did on its first run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import tomllib
from dataclasses import dataclass
from pathlib import Path

RUNTIME_VOLUME = "atlas-swe-rt"
RUNTIME_MOUNT = "/opt/atlas-rt"
RUNTIME_PYTHON = f"{RUNTIME_MOUNT}/venv/bin/python"
RUNTIME_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm-slim"
SRC_MOUNT = "/opt/atlas-src"
HARNESS_MOUNT = "/harness"
RUN_MOUNT = "/atlas-run"
WORKSPACE = "/testbed"
# The published instance images are amd64 only; an arm64 host emulates them.
PLATFORM = "linux/amd64"
# Rollout containers reach these hosts, through the proxy, and no others.
ALLOWED_HOSTS: tuple[str, ...] = ("openrouter.ai",)
ENV_ALLOWED_HOSTS = "ATLAS_SWE_ALLOWED_HOSTS"
NETWORK = "atlas-swe-net"
PROXY_NAME = "atlas-swe-proxy"
PROXY_PORT = 8080
_PROXY_LABEL = "atlas.swe.allow"
_PROXY_LOCK = threading.Lock()
_ROLLOUT = (RUNTIME_PYTHON, "-m", "atlas.harness_eval.rollout")
# A tool call or model request in flight when the deadline passes may still finish.
_DEADLINE_SLACK_SECONDS = 1200


class DockerError(RuntimeError):
    """A docker command failed; the message carries its output."""


@dataclass(frozen=True)
class RolloutSpec:
    """Everything one container rollout needs."""

    instance_id: str
    run_dir: Path
    harness_dir: Path
    prompt: str
    model: str
    api_key: str
    max_tool_calls: int | None = None
    deadline_seconds: float = 2700.0
    reasoning_effort: str | None = None
    provider_order: tuple[str, ...] = ()
    tool_profile: str = "full"
    temperature: float | None = None
    top_p: float | None = None


def rollout_run_args(spec: RolloutSpec, step: list[str]) -> list[str]:
    """Return the in-container ``rollout run`` command for ``spec``."""
    run_args = [
        *_ROLLOUT,
        "run",
        *step,
        "--harness",
        HARNESS_MOUNT,
        "--prompt-file",
        f"{RUN_MOUNT}/task.md",
        "--model",
        spec.model,
        "--deadline",
        str(spec.deadline_seconds),
        "--api-key-stdin",
    ]
    if spec.max_tool_calls is not None:
        run_args += ["--max-tool-calls", str(spec.max_tool_calls)]
    if spec.reasoning_effort is not None:
        run_args += ["--reasoning-effort", spec.reasoning_effort]
    if spec.provider_order:
        run_args += ["--provider", ",".join(spec.provider_order)]
    if spec.tool_profile != "full":
        run_args += ["--tool-profile", spec.tool_profile]
    if spec.temperature is not None:
        run_args += ["--temperature", str(spec.temperature)]
    if spec.top_p is not None:
        run_args += ["--top-p", str(spec.top_p)]
    return run_args


def instance_image(instance_id: str) -> str:
    """Return the published image for a SWE-bench instance."""
    key = instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.x86_64.{key}:latest"


def src_dir() -> Path:
    """Return the directory that holds the ``atlas`` package being run."""
    return Path(__file__).resolve().parents[2]


def allowed_hosts() -> tuple[str, ...]:
    """Return the hosts a rollout may reach; the env var replaces the default."""
    raw = os.environ.get(ENV_ALLOWED_HOSTS)
    if raw is None:
        return ALLOWED_HOSTS
    return tuple(host.strip().lower() for host in raw.split(",") if host.strip())


def docker(
    *args: str,
    timeout: float = 120,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run one docker command and return it; raise :class:`DockerError` on failure."""
    proc = subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        env=env,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-800:]
        raise DockerError(f"docker {args[0]} failed (exit {proc.returncode}): {detail}")
    return proc


def daemon_ready() -> bool:
    """Return whether the docker CLI exists and its daemon answers."""
    if shutil.which("docker") is None:
        return False
    try:
        return docker("info", "--format", "{{.ServerVersion}}", check=False).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def _runtime_mounts(source: Path) -> list[str]:
    return [
        "-v",
        f"{RUNTIME_VOLUME}:{RUNTIME_MOUNT}:ro",
        "-v",
        f"{source}:{SRC_MOUNT}:ro",
    ]


def build_runtime(project_root: Path | None = None) -> None:
    """Build the Linux Atlas runtime into :data:`RUNTIME_VOLUME`.

    The volume holds a standalone CPython 3.12 and a virtualenv with the
    dependencies ``pyproject.toml`` declares. A ``.pth`` file points the
    environment at :data:`SRC_MOUNT`, where the package source is mounted at
    run time. Rebuild after the declared dependencies change.
    """
    root = project_root or src_dir().parent
    with (root / "pyproject.toml").open("rb") as handle:
        deps = tomllib.load(handle)["project"]["dependencies"]
    site = f"{RUNTIME_MOUNT}/venv/lib/python3.12/site-packages"
    script = " && ".join(
        [
            f"uv venv --clear --python 3.12 {RUNTIME_MOUNT}/venv",
            f"uv pip install --python {RUNTIME_PYTHON} {shlex.join(deps)}",
            f"echo {SRC_MOUNT} > {site}/atlas_src.pth",
        ]
    )
    docker(
        "run",
        "--rm",
        "--platform",
        PLATFORM,
        "-v",
        f"{RUNTIME_VOLUME}:{RUNTIME_MOUNT}",
        "-e",
        f"UV_PYTHON_INSTALL_DIR={RUNTIME_MOUNT}/python",
        "-e",
        "UV_PYTHON_PREFERENCE=only-managed",
        "-e",
        "UV_LINK_MODE=copy",
        RUNTIME_IMAGE,
        "sh",
        "-c",
        script,
        timeout=1800,
    )


def runtime_error() -> str:
    """Return why the runtime cannot import the rollout module, or ``""``."""
    try:
        proc = docker(
            "run",
            "--rm",
            "--platform",
            PLATFORM,
            *_runtime_mounts(src_dir()),
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            RUNTIME_IMAGE,
            RUNTIME_PYTHON,
            "-c",
            "import atlas.harness_eval.rollout",
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "timed out importing atlas in the runtime container"
    if proc.returncode == 0:
        return ""
    return (proc.stderr or proc.stdout).strip()[-400:] or "runtime import failed"


def ensure_image(image: str) -> None:
    """Pull ``image`` unless it is already present."""
    if docker("image", "inspect", image, check=False).returncode == 0:
        return
    docker("pull", "--platform", PLATFORM, image, timeout=3600)


def ensure_proxy() -> None:
    """Start the egress proxy and its isolated network unless both are up.

    The network is created ``--internal``, so a container on it has no route
    out. The proxy joins that network and the default bridge. A proxy started
    with a different allowlist is replaced.
    """
    allow = allowed_hosts()
    if not allow:
        raise DockerError(f"{ENV_ALLOWED_HOSTS} is empty; a rollout must reach its model")
    wanted = ",".join(allow)
    with _PROXY_LOCK:
        if docker("network", "inspect", NETWORK, check=False).returncode != 0:
            docker("network", "create", "--internal", NETWORK)
        state = docker(
            "inspect",
            "-f",
            '{{.State.Running}} {{index .Config.Labels "' + _PROXY_LABEL + '"}}',
            PROXY_NAME,
            check=False,
        )
        if state.returncode == 0 and state.stdout.strip() == f"true {wanted}":
            return
        docker("rm", "-f", PROXY_NAME, check=False)
        allow_flags = [flag for host in allow for flag in ("--allow", host)]
        docker(
            "run",
            "-d",
            "--name",
            PROXY_NAME,
            "--platform",
            PLATFORM,
            "--label",
            f"{_PROXY_LABEL}={wanted}",
            "--restart",
            "unless-stopped",
            "-v",
            f"{src_dir()}:{SRC_MOUNT}:ro",
            RUNTIME_IMAGE,
            "python",
            f"{SRC_MOUNT}/atlas/harness_eval/egress_proxy.py",
            "--port",
            str(PROXY_PORT),
            *allow_flags,
            timeout=600,
        )
        docker("network", "connect", NETWORK, PROXY_NAME)


def _proxy_env() -> list[str]:
    """Return ``-e`` flags that route a container's HTTP clients through the proxy."""
    url = f"http://{PROXY_NAME}:{PROXY_PORT}"
    flags: list[str] = []
    for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        flags += ["-e", f"{key}={url}"]
    return flags


def _container_name(run_dir: Path) -> str:
    digest = hashlib.sha1(str(run_dir.resolve()).encode()).hexdigest()[:12]
    return f"atlas-swe-{digest}"


def _exec(
    name: str,
    command: list[str],
    *,
    timeout: float,
    stdin: str | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run ``command`` in a container, optionally feeding it ``stdin``."""
    args = ["docker", "exec", "-e", "PYTHONDONTWRITEBYTECODE=1"]
    if stdin is not None:
        args.append("-i")
    proc = subprocess.run(
        [*args, name, *command],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        input=stdin,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()[-800:]
        raise DockerError(f"docker exec failed (exit {proc.returncode}): {detail}")
    return proc


def _collect_error(run_dir: Path, col: subprocess.CompletedProcess[str] | None) -> str | None:
    """Return why patch collection failed, or None when ``collect.json`` says ok."""
    path = run_dir / "collect.json"
    rc = "timed out" if col is None else f"rc {col.returncode}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return f"collect did not run or wrote no collect.json ({rc})"
    if isinstance(data, dict) and data.get("status") == "error":
        return str(data.get("error") or "unknown error")
    return None


def run_rollout(spec: RolloutSpec) -> dict[str, object]:
    """Run one rollout in its task container and return ``meta.json``.

    The run directory is mounted into the container and receives
    ``trace.json``, ``trajectory.md``, ``meta.json`` and ``patch.diff``. The
    container is always removed. A run that overruns its deadline is killed
    and still has its patch collected; its meta reports ``status: timeout``.

    Raises:
        DockerError: The container could not be started or prepared.
    """
    run_dir = spec.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.md").write_text(spec.prompt, encoding="utf-8")
    image = instance_image(spec.instance_id)
    ensure_image(image)
    ensure_proxy()
    name = _container_name(run_dir)
    docker("rm", "-f", name, check=False, timeout=600)
    docker(
        "run",
        "-d",
        "--name",
        name,
        "--platform",
        PLATFORM,
        "--network",
        NETWORK,
        *_proxy_env(),
        *_runtime_mounts(src_dir()),
        "-v",
        f"{spec.harness_dir.resolve()}:{HARNESS_MOUNT}:ro",
        "-v",
        f"{run_dir}:{RUN_MOUNT}",
        "-w",
        WORKSPACE,
        "--entrypoint",
        "sleep",
        image,
        "infinity",
        timeout=600,
    )
    step = ["--workspace", WORKSPACE, "--out", RUN_MOUNT]
    try:
        _exec(name, [*_ROLLOUT, "prepare", *step], timeout=600)
        run_args = rollout_run_args(spec, step)
        # Activate the task environment so bash children run the repository's
        # interpreter; Atlas itself runs from the runtime volume by absolute path.
        shell = "source /opt/miniconda3/bin/activate testbed 2>/dev/null; exec " + shlex.join(
            run_args
        )
        timed_out = False
        try:
            # The key goes in on stdin: an environment variable would stay readable
            # in /proc for every command the agent runs.
            proc = _exec(
                name,
                ["bash", "-c", shell],
                timeout=spec.deadline_seconds + _DEADLINE_SLACK_SECONDS,
                stdin=spec.api_key + "\n",
                check=False,
            )
            (run_dir / "rollout.log").write_text(
                (proc.stdout + "\n" + proc.stderr).strip() + "\n", encoding="utf-8"
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            _exec(name, ["pkill", "-f", "atlas.harness_eval.rollout"], timeout=60, check=False)
        try:
            col: subprocess.CompletedProcess[str] | None = _exec(
                name, [*_ROLLOUT, "collect", *step], timeout=600, check=False
            )
        except subprocess.TimeoutExpired:
            col = None
    finally:
        docker("rm", "-f", name, check=False, timeout=600)
    meta_path = run_dir / "meta.json"
    if timed_out or not meta_path.is_file():
        meta: dict[str, object] = {
            "status": "timeout" if timed_out else "error",
            "error": "rollout exceeded its wall-clock limit"
            if timed_out
            else "rollout exited without writing meta.json; see rollout.log",
        }
    else:
        loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = loaded if isinstance(loaded, dict) else {"status": "error", "error": "bad meta.json"}
    collect_error = _collect_error(run_dir, col)
    if collect_error is not None:
        if timed_out:
            meta["collect_error"] = collect_error
        else:
            meta["status"] = "error"
            meta["error"] = f"patch collection failed: {collect_error}"
    if timed_out or not meta_path.is_file() or collect_error is not None:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
