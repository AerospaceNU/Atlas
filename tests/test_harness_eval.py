from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from atlas.agent.contracts import ChatMessage, ModelResponse, ToolCall, ToolDefinition
from atlas.agent.trace import read_trace
from atlas.harness_eval.headless import render_trajectory, run_headless
from atlas.harness_eval.surface import ATLAS_SURFACE, diff_summary, is_allowed, validate
from atlas.harness_eval.swe_adapter import SWEAdapter


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[tuple[list[ChatMessage], list[ToolDefinition]]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        self.requests.append((list(messages), list(tools)))
        return self.responses.pop(0)


def test_headless_writes_trace_and_returns(tmp_path: Path) -> None:
    (tmp_path / "notes.txt").write_text("the coast is clear", encoding="utf-8")
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="read_file", arguments={"path": "notes.txt"})
                ]
            ),
            ModelResponse(content="The notes say the coast is clear."),
        ]
    )
    out = run_headless(
        workspace=tmp_path,
        prompt="What do the notes say?",
        model_id="",
        api_key="",
        trace_path=tmp_path / "trace.json",
        model=model,
    )
    assert out.run.response == "The notes say the coast is clear."
    assert out.trace_path.is_file()
    assert read_trace(out.trace_path).run.response == "The notes say the coast is clear."
    assert "[ACTION] read_file" in render_trajectory(out.run)
    assert "[FINAL]" in render_trajectory(out.run)


def test_headless_rejects_empty_prompt(tmp_path: Path) -> None:
    try:
        run_headless(
            workspace=tmp_path, prompt="   ", model_id="", api_key="", model=ScriptedModel([])
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_surface_accepts_and_rejects() -> None:
    assert is_allowed(ATLAS_SURFACE, "systemprompt.md")
    assert is_allowed(ATLAS_SURFACE, "skills/demo/SKILL.md")
    assert is_allowed(ATLAS_SURFACE, "scripts/helper.py")
    assert not is_allowed(ATLAS_SURFACE, "etc/passwd")
    assert not is_allowed(ATLAS_SURFACE, "../evil.py")
    tree = {"systemprompt.md": "hi\n", "skills/demo/SKILL.md": "---\nname: demo\n---\nbody\n"}
    validate(ATLAS_SURFACE, tree)
    before = {"systemprompt.md": "a"}
    after = {"systemprompt.md": "b", "guardrails.md": "new"}
    diff = diff_summary(ATLAS_SURFACE, before, after)
    assert diff["modified"] == ["systemprompt.md"]
    assert diff["added"] == ["guardrails.md"]


def test_substrate_materialises(tmp_path: Path) -> None:
    from atlas.harness_eval.substrate import AtlasSubstrate

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "systemprompt.md").write_text("base\n", encoding="utf-8")
    (tree / "guardrails.md").write_text("guard\n", encoding="utf-8")
    spec = type("Spec", (), {"install_mode": "workspace_agents_md"})()
    install = AtlasSubstrate().materialise(tree, spec, {})
    try:
        assert (install.staging_dir / "systemprompt.md").is_file()
        assert install.fingerprint
    finally:
        import shutil

        shutil.rmtree(install.staging_dir, ignore_errors=True)


def _make_lite(tmp_path: Path) -> tuple[Path, Path]:
    jsonl = tmp_path / "lite.jsonl"
    records = [
        {
            "instance_id": "repo__pkg-1",
            "repo": "repo/pkg",
            "problem_statement": "Add a feature for things.",
            "base_commit": "abc123",
        },
        {
            "instance_id": "repo__pkg-2",
            "repo": "repo/pkg",
            "problem_statement": "Fix the crash.",
            "base_commit": "def456",
        },
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    data_root = tmp_path / "swe_data"
    for r in records:
        task_dir = data_root / r["instance_id"]
        repo = task_dir / "repo"
        repo.mkdir(parents=True)
        (repo / "README.md").write_text("# repo\n", encoding="utf-8")
        # Minimal git so collect_patch works
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    return jsonl, data_root


def test_swe_adapter_stream_hidden_hints(tmp_path: Path) -> None:
    jsonl, data_root = _make_lite(tmp_path)
    adapter = SWEAdapter(jsonl=jsonl, data_root=data_root, runs_root=tmp_path / "runs")
    assert adapter.tasks() == ["repo__pkg-1", "repo__pkg-2"]
    assert adapter.domain_of("repo__pkg-1") == "repo/pkg"
    assert "Add a feature" in adapter.task_prompt("repo__pkg-1")
    # Hints must never appear (adapter only reads problem_statement).
    assert adapter.task_files("repo__pkg-1") == data_root / "repo__pkg-1"
    base = adapter.base_harness()
    assert (base / "systemprompt.md").is_file()


def test_swe_adapter_execute_offline(tmp_path: Path) -> None:
    jsonl, data_root = _make_lite(tmp_path)
    adapter = SWEAdapter(jsonl=jsonl, data_root=data_root, runs_root=tmp_path / "runs")
    model = ScriptedModel([ModelResponse(content="no files changed")])
    run_dir = adapter.execute("repo__pkg-1", None, "run-1", model=model)
    assert (run_dir / "trace.json").is_file()
    assert (run_dir / "patch.diff").is_file()
    traj = adapter.trajectory(run_dir)
    assert traj
    # Score without pytest: no patch => unresolved.
    assert adapter.score(run_dir, "repo__pkg-1") == {"resolved": False, "reason": "no patch"}


def test_harness2_bench_file_is_present() -> None:
    """Bridge file must exist for setup_harness2_swe.sh to copy (no harness2 import)."""
    from atlas.harness_eval import harness2_bench

    assert hasattr(harness2_bench, "SWEBenchAdapter")
    assert hasattr(harness2_bench, "_bench_spec")


def test_reasoning_effort_reaches_container_command_and_meta(tmp_path: Path) -> None:
    from atlas.harness_eval.rollout import _meta, _parser
    from atlas.harness_eval.swe_docker import RolloutSpec, rollout_run_args

    spec = RolloutSpec(
        instance_id="a__b-1",
        run_dir=tmp_path,
        harness_dir=tmp_path,
        prompt="p",
        model="m",
        api_key="k",
    )
    assert "--reasoning-effort" not in rollout_run_args(spec, [])
    cmd = rollout_run_args(replace(spec, reasoning_effort="high"), [])
    assert cmd[cmd.index("--reasoning-effort") + 1] == "high"
    assert "--provider" not in cmd
    pinned = rollout_run_args(replace(spec, provider_order=("a", "b")), [])
    assert pinned[pinned.index("--provider") + 1] == "a,b"
    argv = ["run", "--workspace", "w", "--out", "o", "--prompt-file", "p", "--model", "m"]
    assert _parser().parse_args(argv).reasoning_effort is None
    assert _parser().parse_args([*argv, "--reasoning-effort", "low"]).reasoning_effort == "low"
    assert _parser().parse_args([*argv, "--provider", "a, b"]).provider == "a, b"

    (tmp_path / "n.txt").write_text("x", encoding="utf-8")
    result = run_headless(
        workspace=tmp_path,
        prompt="go",
        model_id="m",
        api_key="k",
        model=ScriptedModel([ModelResponse(content="done")]),
    )
    assert _meta(result, "low")["reasoning_effort"] == "low"


def test_coding_profile_drops_satellite_model_and_web_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ATLAS_HEURISTIC_MODELS", "1")
    from atlas.harness_eval.headless import NON_CODING_TOOLS, profile_registry

    full = profile_registry("full")
    coding = profile_registry("coding")
    assert set(full.names) >= NON_CODING_TOOLS
    assert not NON_CODING_TOOLS & set(coding.names)
    assert {"bash", "read_file", "edit_file", "run_tests", "list_tools"} <= set(coding.names)
    listed = {d.name for d in coding.definitions}
    assert listed == set(coding.names)
    try:
        profile_registry("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown profile accepted")


def test_coding_profile_reaches_agent_and_children(tmp_path: Path) -> None:
    from atlas.agent.artifacts import LocalArtifactStore
    from atlas.harness_eval.headless import _build_agent
    from atlas.harness_eval.substrate import InstalledHarness

    agent = _build_agent(
        store=LocalArtifactStore(tmp_path, artifact_root=tmp_path),
        model=ScriptedModel([]),
        harness=InstalledHarness(),
        max_tool_calls=None,
        read_only=False,
        approval="allow",
        tool_profile="coding",
    )
    assert "web_fetch" not in agent.tools.names
    assert "spawn_agent" in agent.tools.names


def test_tool_profile_and_sampling_reach_command_and_meta(tmp_path: Path) -> None:
    from atlas.harness_eval.rollout import _meta, _parser
    from atlas.harness_eval.swe_docker import RolloutSpec, rollout_run_args

    spec = RolloutSpec(
        instance_id="i", run_dir=tmp_path, harness_dir=tmp_path, prompt="p", model="m", api_key="k"
    )
    cmd = rollout_run_args(spec, [])
    assert not {"--tool-profile", "--temperature", "--top-p"} & set(cmd)
    cmd = rollout_run_args(replace(spec, tool_profile="coding", temperature=0.5, top_p=0.9), [])
    assert cmd[cmd.index("--tool-profile") + 1] == "coding"
    assert cmd[cmd.index("--temperature") + 1] == "0.5"
    assert cmd[cmd.index("--top-p") + 1] == "0.9"
    argv = ["run", "--workspace", "w", "--out", "o", "--prompt-file", "p", "--model", "m"]
    args = _parser().parse_args([*argv, "--tool-profile", "coding", "--temperature", "0.3"])
    assert (args.tool_profile, args.temperature, args.top_p) == ("coding", 0.3, None)
    result = run_headless(
        workspace=tmp_path,
        prompt="go",
        model_id="m",
        api_key="k",
        model=ScriptedModel([ModelResponse(content="done")]),
        tool_profile="coding",
    )
    meta = _meta(result, None, (), "coding", 0.3, None)
    assert (meta["tool_profile"], meta["temperature"], meta["top_p"]) == ("coding", 0.3, None)
