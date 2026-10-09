from __future__ import annotations

import builtins
import io
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import build_registry, default_registry
from atlas.agent.tools.rsicc_tool import RSICCTool


@dataclass
class _FakeResponse:
    text: str | None


@dataclass
class _FakeModels:
    reply: str | None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def generate_content(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self.reply)


class _FakeGemini:
    """Stands in for ``google.genai.Client`` and records every request."""

    def __init__(self, reply: str | None) -> None:
        self.api_keys: list[str | None] = []
        self.models = _FakeModels(reply)

    def __call__(self, *, api_key: str | None = None) -> _FakeGemini:
        self.api_keys.append(api_key)
        return self


class _FakeInlineData:
    def __init__(self, data: bytes, mime_type: str) -> None:
        self.data = data
        self.mime_type = mime_type


class _FakePart:
    def __init__(self, data: bytes, mime_type: str) -> None:
        self.inline_data = _FakeInlineData(data, mime_type)

    @classmethod
    def from_bytes(cls, *, data: bytes, mime_type: str) -> _FakePart:
        return cls(data, mime_type)


class _FakeAutomaticFunctionCallingConfig:
    def __init__(self, *, disable: bool) -> None:
        self.disable = disable


class _FakeGenerateContentConfig:
    def __init__(self, *, automatic_function_calling: _FakeAutomaticFunctionCallingConfig) -> None:
        self.automatic_function_calling = automatic_function_calling


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch, client: _FakeGemini) -> None:
    """Expose a Gemini stand-in so tests run without the google-genai package."""
    genai_types = ModuleType("google.genai.types")
    genai_types.Part = _FakePart  # type: ignore[attr-defined]
    genai_types.GenerateContentConfig = _FakeGenerateContentConfig  # type: ignore[attr-defined]
    genai_types.AutomaticFunctionCallingConfig = _FakeAutomaticFunctionCallingConfig  # type: ignore[attr-defined]

    genai = ModuleType("google.genai")
    genai.Client = client  # type: ignore[attr-defined]
    genai.types = genai_types  # type: ignore[attr-defined]

    google = sys.modules.get("google")
    if google is None:
        google = ModuleType("google")
        google.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", genai, raising=False)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", genai_types)


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> _FakeGemini:
    fake = _FakeGemini(reply="  The lake in the before image is dry in the after image.\n")
    _install_fake_genai(monkeypatch, fake)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return fake


def _save(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path.name


def _rgb(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _sent_images(gemini: _FakeGemini) -> list[Any]:
    contents = gemini.models.calls[-1]["contents"]
    return [part for part in contents if getattr(part, "inline_data", None) is not None]


def _decode(part: Any) -> Image.Image:
    assert part.inline_data is not None
    assert part.inline_data.data is not None
    return Image.open(io.BytesIO(part.inline_data.data))


def test_rsicc_tool_is_opt_in() -> None:
    names = [definition.name for definition in default_registry().definitions]
    opted_in = [
        definition.name
        for definition in build_registry(("atlas.agent.tools",), include_opt_in=True).definitions
    ]

    assert RSICCTool.trust == "opt_in"
    assert "rsicc_tool" not in names
    assert "rsicc_tool" in opted_in


def test_returns_the_gemini_reply(tmp_path: Path, gemini: _FakeGemini) -> None:
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(40, 30), tmp_path / "before.png")
    after = _save(_rgb(40, 30), tmp_path / "after.png")

    result = RSICCTool().run({"image1": before, "image2": after}, store)

    assert result.text == "The lake in the before image is dry in the after image."
    assert result.artifacts == []
    assert gemini.api_keys == ["test-key"]


def test_sends_labelled_before_and_after_images_in_order(
    tmp_path: Path, gemini: _FakeGemini
) -> None:
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(40, 30), tmp_path / "before.png")
    after = _save(_rgb(20, 10), tmp_path / "after.png")

    RSICCTool().run({"image1": before, "image2": after}, store)

    [call] = gemini.models.calls
    contents = call["contents"]
    assert contents[0] == "Before image:"
    assert contents[2] == "After image:"
    assert isinstance(contents[4], str)
    assert _decode(contents[1]).size == (40, 30)
    assert _decode(contents[3]).size == (20, 10)


def test_disables_automatic_function_calling(tmp_path: Path, gemini: _FakeGemini) -> None:
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(8, 8), tmp_path / "before.png")
    after = _save(_rgb(8, 8), tmp_path / "after.png")

    RSICCTool().run({"image1": before, "image2": after}, store)

    config = gemini.models.calls[0]["config"]
    assert config.automatic_function_calling.disable is True


def test_reads_images_from_subdirectories(tmp_path: Path, gemini: _FakeGemini) -> None:
    store = LocalArtifactStore(tmp_path)
    _save(_rgb(8, 8), tmp_path / "scenes" / "2020.png")
    _save(_rgb(8, 8), tmp_path / "scenes" / "2024.png")

    RSICCTool().run({"image1": "scenes/2020.png", "image2": "scenes/2024.png"}, store)

    assert len(_sent_images(gemini)) == 2


def test_missing_image_raises_without_calling_gemini(tmp_path: Path, gemini: _FakeGemini) -> None:
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(8, 8), tmp_path / "before.png")

    with pytest.raises(FileNotFoundError, match=r"missing\.png"):
        RSICCTool().run({"image1": before, "image2": "missing.png"}, store)

    assert gemini.models.calls == []


def test_path_outside_the_workspace_is_rejected(tmp_path: Path, gemini: _FakeGemini) -> None:
    store = LocalArtifactStore(tmp_path / "workspace")
    _save(_rgb(8, 8), tmp_path / "outside.png")
    inside = _save(_rgb(8, 8), tmp_path / "workspace" / "inside.png")

    with pytest.raises(ValueError, match="escapes the artifact root"):
        RSICCTool().run({"image1": "../outside.png", "image2": inside}, store)

    assert gemini.models.calls == []


def test_missing_api_key_raises(
    tmp_path: Path, gemini: _FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY")
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(8, 8), tmp_path / "before.png")
    after = _save(_rgb(8, 8), tmp_path / "after.png")

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        RSICCTool().run({"image1": before, "image2": after}, store)

    assert gemini.models.calls == []


def test_missing_sdk_raises_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(8, 8), tmp_path / "before.png")
    after = _save(_rgb(8, 8), tmp_path / "after.png")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    for name in ("google", "google.genai", "google.genai.types"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    real_import = builtins.__import__

    def _blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "google" or name.startswith("google."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)

    with pytest.raises(RuntimeError, match="uv sync --extra gemini"):
        RSICCTool().run({"image1": before, "image2": after}, store)


def test_oversized_image_is_rejected_without_reading(
    tmp_path: Path, gemini: _FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)
    big = tmp_path / "big.png"
    big.write_bytes(b"not-read")
    os.truncate(big, RSICCTool.max_bytes + 1)
    small = _save(_rgb(8, 8), tmp_path / "small.png")
    original = Path.read_bytes

    def _guarded(self: Path) -> bytes:
        if self.name == "big.png":
            raise AssertionError("oversized image was read")
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", _guarded)

    with pytest.raises(SandboxDenied, match=f"big.png exceeds the {RSICCTool.max_bytes} byte"):
        RSICCTool().run({"image1": "big.png", "image2": small}, store)

    assert gemini.models.calls == []


def test_image_at_the_byte_limit_is_sent(
    tmp_path: Path, gemini: _FakeGemini, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"image bytes"
    monkeypatch.setattr(RSICCTool, "max_bytes", len(payload))
    store = LocalArtifactStore(tmp_path)
    (tmp_path / "limit.png").write_bytes(payload)

    RSICCTool().run({"image1": "limit.png", "image2": "limit.png"}, store)

    parts = _sent_images(gemini)
    assert len(parts) == 2
    assert parts[0].inline_data.data == payload


@pytest.mark.parametrize("reply", [None, ""])
def test_empty_gemini_reply_raises(tmp_path: Path, gemini: _FakeGemini, reply: str | None) -> None:
    gemini.models.reply = reply
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(8, 8), tmp_path / "before.png")
    after = _save(_rgb(8, 8), tmp_path / "after.png")

    with pytest.raises(RuntimeError, match="no text"):
        RSICCTool().run({"image1": before, "image2": after}, store)


@pytest.mark.parametrize("arguments", [{"image1": "a.png"}, {"image1": "a.png", "image2": 3}])
def test_invalid_arguments_are_rejected(
    tmp_path: Path, gemini: _FakeGemini, arguments: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        RSICCTool().run(arguments, LocalArtifactStore(tmp_path))


@pytest.mark.parametrize(
    ("filename", "mime_type"),
    [
        ("image.png", "image/png"),
        ("image.jpg", "image/jpeg"),
        ("image.jpeg", "image/jpeg"),
        ("image.webp", "image/webp"),
        ("image.heic", "image/heic"),
        ("image.heif", "image/heif"),
        ("IMAGE.PNG", "image/png"),
    ],
)
def test_supported_values_are_sent_fine(
    tmp_path: Path, gemini: _FakeGemini, filename: str, mime_type: str
) -> None:
    store = LocalArtifactStore(tmp_path)

    (tmp_path / filename).write_bytes(b"image bytes")

    RSICCTool().run({"image1": filename, "image2": filename}, store)

    parts = _sent_images(gemini)
    assert len(parts) == 2
    for part in parts:
        assert part.inline_data is not None
        assert part.inline_data.mime_type == mime_type
        assert part.inline_data.data == b"image bytes"


@pytest.mark.parametrize("filename", ["palette.gif", "rgb.bmp", "rgb.tif", "gray.tiff"])
def test_unsupported_formats_are_rejected_without_calling_gemini(
    tmp_path: Path, gemini: _FakeGemini, filename: str
) -> None:
    store = LocalArtifactStore(tmp_path)
    (tmp_path / filename).write_bytes(b"image bytes")

    with pytest.raises(ValueError, match="Unsupported image type"):
        RSICCTool().run({"image1": filename, "image2": filename}, store)

    assert gemini.models.calls == []


@pytest.mark.evals
def test_live_gemini_describes_an_obvious_change(tmp_path: Path) -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set")

    blank = np.full((128, 128, 3), 255, dtype=np.uint8)
    square = blank.copy()
    square[32:96, 32:96] = (220, 30, 30)
    store = LocalArtifactStore(tmp_path)
    before = _save(Image.fromarray(blank), tmp_path / "before.png")
    after = _save(Image.fromarray(square), tmp_path / "after.png")

    result = RSICCTool().run({"image1": before, "image2": after}, store)

    assert result.text.strip()
    assert "red" in result.text.lower() or "square" in result.text.lower()
