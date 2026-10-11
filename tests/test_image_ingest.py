"""Local image artifacts are read in the sandbox and sent to the model."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import BaseModel

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import (
    ChatMessage,
    ModelResponse,
    Role,
    Tool,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
)
from atlas.agent.images import (
    MAX_IMAGE_BYTES,
    MAX_REQUEST_IMAGE_BYTES,
    MAX_SOURCE_BYTES,
    apply_image_budget,
    ingest_image,
    ingest_images,
)
from atlas.agent.model import ModelConfig, OpenRouterModel
from atlas.agent.runtime import Agent
from atlas.agent.session import AgentSession


class _Opened:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    def __enter__(self) -> FakeResponse:
        return self.response

    def __exit__(self, *_exc: object) -> None:
        return None


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.status_code = 200
        self.elapsed_seconds = 0.0
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self.payload

    def iter_lines(self) -> list[str]:
        import json

        return [f"data: {json.dumps(self.payload)}", "data: [DONE]"]


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def stream(self, _method: str, _url: str, **kwargs: Any) -> _Opened:
        self.calls.append(kwargs)
        return _Opened(
            FakeResponse({"choices": [{"message": {"content": "seen", "tool_calls": []}}]})
        )


class ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests: list[list[ChatMessage]] = []

    def complete(self, messages: list[ChatMessage], tools: list[ToolDefinition]) -> ModelResponse:
        del tools
        self.requests.append(list(messages))
        return self.responses.pop(0)


class EmptyInput(BaseModel):
    pass


class EmitArtifacts(Tool):
    name = "emit_artifacts"
    description = "Return prepared artifact paths."

    def __init__(self, artifacts: list[str]) -> None:
        self._artifacts = artifacts

    @property
    def input_model(self) -> type[BaseModel]:
        return EmptyInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        del arguments, store
        return ToolResult(text="wrote artifacts", artifacts=list(self._artifacts))


def _save(path: Path, *, fmt: str, color: str = "red") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 2), color).save(path, format=fmt)


def _registry(artifacts: list[str]) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EmitArtifacts(artifacts))
    return registry


def test_ingest_png_jpeg_webp_and_gif(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    specs = [
        ("scenes/coast.png", "PNG", "image/png"),
        ("coast.jpg", "JPEG", "image/jpeg"),
        ("coast.jpeg", "JPEG", "image/jpeg"),
        ("coast.webp", "WEBP", "image/webp"),
        ("coast.gif", "GIF", "image/gif"),
    ]
    for name, fmt, _media in specs:
        _save(tmp_path / name, fmt=fmt)

    images = ingest_images(store, [name for name, _fmt, _media in specs])

    assert [image.path for image in images] == [name for name, _fmt, _media in specs]
    assert [image.media_type for image in images] == [media for _name, _fmt, media in specs]
    for image, (name, _fmt, _media) in zip(images, specs, strict=True):
        assert base64.b64decode(image.data) == (tmp_path / name).read_bytes()
        assert "\n" not in image.data


def test_ingest_rejects_an_escape_and_a_hardlink(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = LocalArtifactStore(workspace)
    outside = tmp_path / "secret.png"
    _save(outside, fmt="PNG")
    os.link(outside, workspace / "linked.png")

    with pytest.raises(SandboxDenied, match="escapes"):
        ingest_image(store, "../secret.png")
    with pytest.raises(SandboxDenied, match="hardlinked"):
        ingest_image(store, "linked.png")


def test_ingest_rejects_missing_unsupported_and_unreadable_files(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(tmp_path / "notes.bmp", fmt="BMP")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    (tmp_path / "no-suffix").write_bytes(b"not an image")

    with pytest.raises(FileNotFoundError, match=r"missing\.png"):
        ingest_image(store, "missing.png")
    with pytest.raises(ValueError, match="Unsupported image format BMP"):
        ingest_image(store, "notes.bmp")
    with pytest.raises(ValueError, match="not a readable image"):
        ingest_image(store, "no-suffix")
    with pytest.raises(ValueError, match="not a readable image"):
        ingest_image(store, "broken.png")


def test_media_type_comes_from_the_pillow_format(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    path = tmp_path / "coast.png"
    Image.new("RGB", (4, 2), "blue").save(path, format="JPEG")

    image = ingest_image(store, "coast.png")

    assert image.media_type == "image/jpeg"
    assert base64.b64decode(image.data) == path.read_bytes()


def test_ingest_reads_a_workspace_image_and_prefers_the_write_root(tmp_path: Path) -> None:
    write_root = tmp_path / "artifacts"
    read_root = tmp_path / "project"
    store = LocalArtifactStore(write_root, read_root=read_root)
    _save(read_root / "inputs" / "scene.png", fmt="PNG", color="blue")
    _save(write_root / "scene.png", fmt="PNG", color="red")
    _save(read_root / "scene.png", fmt="PNG", color="green")

    workspace = ingest_image(store, "inputs/scene.png")
    preferred = ingest_image(store, "scene.png")

    assert workspace.path == "inputs/scene.png"
    assert not workspace.path.startswith("/")
    assert base64.b64decode(workspace.data) == (read_root / "inputs" / "scene.png").read_bytes()
    assert base64.b64decode(preferred.data) == (write_root / "scene.png").read_bytes()


def test_source_over_the_read_limit_is_rejected_without_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    big = tmp_path / "big.png"
    _save(big, fmt="PNG")
    monkeypatch.setattr("atlas.agent.images.MAX_SOURCE_BYTES", big.stat().st_size - 1)
    original = Path.read_bytes

    def _guarded(self: Path) -> bytes:
        if self.name == "big.png":
            raise AssertionError("oversized image was read")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", _guarded)

    with pytest.raises(SandboxDenied, match=f"big.png exceeds the {big.stat().st_size - 1} byte"):
        ingest_image(store, "big.png")


def test_image_at_the_byte_limit_is_ingested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    path = tmp_path / "limit.png"
    _save(path, fmt="PNG")
    raw = path.read_bytes()
    monkeypatch.setattr("atlas.agent.images.MAX_IMAGE_BYTES", len(base64.b64encode(raw)))

    image = ingest_image(store, "limit.png")

    assert image.data == base64.b64encode(raw).decode("ascii")
    assert len(image.data) <= len(base64.b64encode(raw))


def test_image_over_the_per_image_cap_is_downscaled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    path = tmp_path / "wide.png"
    Image.frombytes(
        "RGB",
        (128, 128),
        bytes((index * 17) % 256 for index in range(128 * 128 * 3)),
    ).save(path)
    cap = 200
    assert path.stat().st_size > cap
    monkeypatch.setattr("atlas.agent.images.MAX_IMAGE_BYTES", cap)

    image = ingest_image(store, "wide.png")

    assert len(image.data) <= cap
    with Image.open(io.BytesIO(base64.b64decode(image.data))) as reduced:
        assert reduced.size[0] < 128
    assert MAX_IMAGE_BYTES == 5 * 1024 * 1024
    assert MAX_REQUEST_IMAGE_BYTES == 15 * 1024 * 1024
    assert MAX_SOURCE_BYTES > MAX_IMAGE_BYTES


def test_base64_over_the_cap_is_downscaled_when_raw_bytes_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    path = tmp_path / "wide.png"
    Image.frombytes(
        "RGB",
        (128, 128),
        bytes((index * 17) % 256 for index in range(128 * 128 * 3)),
    ).save(path)
    raw = path.read_bytes()
    cap = len(raw)
    assert len(base64.b64encode(raw)) > cap
    monkeypatch.setattr("atlas.agent.images.MAX_IMAGE_BYTES", cap)

    image = ingest_image(store, "wide.png")

    assert len(image.data) <= cap
    with Image.open(io.BytesIO(base64.b64decode(image.data))) as reduced:
        assert reduced.size[0] < 128


def test_model_sends_image_artifacts_as_data_urls(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(tmp_path / "coast.png", fmt="PNG", color="blue")
    image = ingest_image(store, "coast.png")
    client = FakeClient()
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), client)

    model.complete(
        [ChatMessage(role=Role.USER, content="Describe the coast.", images=[image])],
        [],
    )

    content = client.calls[0]["json"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe the coast."}
    assert content[1] == {"type": "text", "text": "Image artifact: coast.png"}
    url = content[2]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert (
        base64.b64decode(url.removeprefix("data:image/png;base64,"))
        == (tmp_path / "coast.png").read_bytes()
    )


def test_model_keeps_text_content_when_there_are_no_images() -> None:
    client = FakeClient()
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), client)

    model.complete([ChatMessage(role=Role.USER, content="look")], [])

    assert client.calls[0]["json"]["messages"][0]["content"] == "look"


def test_agent_sends_tool_images_on_a_following_user_message(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(tmp_path / "scenes" / "chip.png", fmt="PNG")
    (tmp_path / "notes.txt").write_text("plain", encoding="utf-8")
    (tmp_path / "heat.npy").write_bytes(b"npy")
    (tmp_path / "broken.png").write_bytes(b"nope")
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="emit_artifacts", arguments={})]),
            ModelResponse(content="I can see the chip."),
        ]
    )
    agent = Agent(
        model,
        _registry(["scenes/chip.png", "notes.txt", "heat.npy", "broken.png"]),
        store,
    )

    result = agent.run("What was written?")

    assert result.response == "I can see the chip."
    assert result.steps[0].error_kind == "tool_failure"
    assert result.steps[0].error is not None
    assert "broken.png" in result.steps[0].error
    assert "not a readable image" in result.steps[0].error
    assert "notes.txt" not in result.steps[0].error
    tool_message = model.requests[1][-2]
    follow = model.requests[1][-1]
    assert tool_message.role == Role.TOOL
    assert tool_message.images == []
    assert follow.role == Role.USER
    assert [image.path for image in follow.images] == ["scenes/chip.png"]
    assert follow.images[0].media_type == "image/png"
    assert follow.content is not None
    assert "broken.png" in follow.content
    assert (
        base64.b64decode(follow.images[0].data) == (tmp_path / "scenes" / "chip.png").read_bytes()
    )


def test_agent_records_an_over_cap_tool_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    path = tmp_path / "huge.png"
    _save(path, fmt="PNG")
    monkeypatch.setattr("atlas.agent.images.MAX_IMAGE_BYTES", 1)
    model = ScriptedModel(
        [
            ModelResponse(tool_calls=[ToolCall(id="call-1", name="emit_artifacts", arguments={})]),
            ModelResponse(content="continued"),
        ]
    )
    agent = Agent(model, _registry(["huge.png"]), store)

    result = agent.run("look")

    assert result.response == "continued"
    assert result.steps[0].result is not None
    assert result.steps[0].error_kind == "tool_failure"
    assert result.steps[0].error is not None
    assert "exceeds" in result.steps[0].error
    assert model.requests[1][-2].role == Role.TOOL
    assert model.requests[1][-2].images == []
    assert model.requests[1][-1].role == Role.USER
    assert model.requests[1][-1].images == []
    assert model.requests[1][-1].content is not None
    assert "exceeds" in model.requests[1][-1].content


def test_run_ingests_user_images_before_calling_the_model(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(tmp_path / "before.png", fmt="PNG", color="green")
    model = ScriptedModel([ModelResponse(content="green tile")])
    agent = Agent(model, ToolRegistry(), store)

    result = agent.run("What color is this?", images=["before.png"])

    assert result.response == "green tile"
    user = model.requests[0][1]
    assert user.role == Role.USER
    assert user.content == "What color is this?"
    assert user.images[0].path == "before.png"
    assert user.images[0].media_type == "image/png"


def test_run_does_not_call_the_model_when_the_image_escapes(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="should not run")])
    agent = Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path))

    with pytest.raises(SandboxDenied, match="escapes"):
        agent.run("look", images=["../outside.png"])

    assert model.requests == []


def test_session_turn_sends_user_images_and_keeps_history(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(tmp_path / "frame.png", fmt="PNG")
    model = ScriptedModel([ModelResponse(content="a frame")])
    session = AgentSession(Agent(model, ToolRegistry(), store))

    result = session.turn("Describe the frame.", images=["frame.png"])

    assert result.response == "a frame"
    assert session.messages[-2].images[0].path == "frame.png"
    assert session.messages[-1].content == "a frame"


def test_tool_messages_do_not_carry_image_parts(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(tmp_path / "one.png", fmt="PNG", color="red")
    _save(tmp_path / "two.png", fmt="PNG", color="blue")
    image = ingest_image(store, "one.png")
    client = FakeClient()
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), client)

    model.complete(
        [
            ChatMessage(role=Role.TOOL, tool_call_id="call-1", content="{}", images=[image]),
            ChatMessage(role=Role.USER, content="see", images=[image]),
        ],
        [],
    )

    sent = client.calls[0]["json"]["messages"]
    assert sent[0]["content"] == "{}"
    assert sent[1]["content"][2]["image_url"]["url"].startswith("data:image/png;base64,")


def test_newer_images_replace_older_ones_in_the_request_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    paths = []
    for name, color in (("old.png", "red"), ("mid.png", "green"), ("new.png", "blue")):
        _save(tmp_path / name, fmt="PNG", color=color)
        paths.append(name)
    images = ingest_images(store, paths)
    monkeypatch.setattr(
        "atlas.agent.images.MAX_REQUEST_IMAGE_BYTES",
        len(images[2].data) + len(images[1].data),
    )
    messages = [
        ChatMessage(role=Role.USER, content="old", images=[images[0]]),
        ChatMessage(role=Role.USER, content="mid", images=[images[1]]),
        ChatMessage(role=Role.USER, content="new", images=[images[2]]),
    ]
    client = FakeClient()
    model = OpenRouterModel(ModelConfig(model="example/model", api_key="test"), client)

    model.complete(messages, [])

    assert messages[0].images == []
    assert messages[0].content == "old\n[image old.png omitted]"
    assert [image.path for image in messages[1].images] == ["mid.png"]
    assert [image.path for image in messages[2].images] == ["new.png"]
    sent = client.calls[0]["json"]["messages"]
    assert sent[0]["content"] == "old\n[image old.png omitted]"
    assert "image_url" not in str(sent[0])
    assert "image_url" in str(sent[1]["content"])
    assert "image_url" in str(sent[2]["content"])
    model.complete(messages, [])
    resent = client.calls[1]["json"]["messages"]
    assert resent[0]["content"] == "old\n[image old.png omitted]"
    assert "image_url" not in str(resent[0])


def test_evicted_image_without_text_leaves_only_the_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(tmp_path / "old.png", fmt="PNG", color="red")
    _save(tmp_path / "new.png", fmt="PNG", color="blue")
    images = ingest_images(store, ["old.png", "new.png"])
    monkeypatch.setattr("atlas.agent.images.MAX_REQUEST_IMAGE_BYTES", len(images[1].data))
    messages = [ChatMessage(role=Role.USER, content=None, images=images)]

    apply_image_budget(messages)

    assert [image.path for image in messages[0].images] == ["new.png"]
    assert messages[0].content == "[image old.png omitted]"


def test_two_tool_images_follow_the_tool_messages(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(tmp_path / "one.png", fmt="PNG", color="red")
    _save(tmp_path / "two.png", fmt="PNG", color="blue")
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(id="call-1", name="emit_artifacts", arguments={}),
                    ToolCall(id="call-2", name="emit_artifacts", arguments={}),
                ]
            ),
            ModelResponse(content="both"),
        ]
    )

    class EmitSequence(Tool):
        name = "emit_artifacts"
        description = "Return the next artifact batch."

        def __init__(self) -> None:
            self.batches = [["one.png"], ["two.png"]]

        @property
        def input_model(self) -> type[BaseModel]:
            return EmptyInput

        def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
            del arguments, store
            return ToolResult(text="wrote", artifacts=self.batches.pop(0))

    registry = ToolRegistry()
    registry.register(EmitSequence())
    result = Agent(model, registry, store).run("look")

    assert result.response == "both"
    history = model.requests[1]
    assert [message.role for message in history[-3:]] == [Role.TOOL, Role.TOOL, Role.USER]
    assert history[-3].images == []
    assert history[-2].images == []
    assert [image.path for image in history[-1].images] == ["one.png", "two.png"]


def test_session_turn_leaves_history_alone_when_ingest_fails(tmp_path: Path) -> None:
    model = ScriptedModel([ModelResponse(content="nope")])
    session = AgentSession(Agent(model, ToolRegistry(), LocalArtifactStore(tmp_path)))
    before = list(session.messages)

    with pytest.raises(FileNotFoundError, match=r"gone\.png"):
        session.turn("look", images=["gone.png"])

    assert session.messages == before
    assert model.requests == []
    assert session._analysis_open is False
