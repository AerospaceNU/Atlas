from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from google.genai import types
from PIL import Image
from pydantic import ValidationError

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import default_registry
from atlas.agent.tools.rsicc_tool import RSICCTool

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


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


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> _FakeGemini:
    fake = _FakeGemini(reply="  The lake in the before image is dry in the after image.\n")
    monkeypatch.setattr("google.genai.Client", fake)
    monkeypatch.setenv("GEMMA_API_KEY", "test-key")
    return fake


def _save(image: Image.Image, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path.name


def _rgb(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _sent_images(gemini: _FakeGemini) -> list[types.Part]:
    contents = gemini.models.calls[-1]["contents"]
    return [part for part in contents if isinstance(part, types.Part)]


def _decode(part: types.Part) -> Image.Image:
    assert part.inline_data is not None
    assert part.inline_data.data is not None
    return Image.open(io.BytesIO(part.inline_data.data))


def test_rsicc_tool_is_discovered() -> None:
    names = [definition.name for definition in default_registry().definitions]

    assert "rsicc_tool" in names


def test_returns_the_gemini_reply(tmp_path: Path, gemini: _FakeGemini) -> None:
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(40, 30), tmp_path / "before.png")
    after = _save(_rgb(40, 30), tmp_path / "after.png")

    result = default_registry().execute("rsicc_tool", {"image1": before, "image2": after}, store)

    assert result.text == "The lake in the before image is dry in the after image."
    assert result.artifacts == []
    assert gemini.api_keys == ["test-key"]


def test_sends_labelled_before_and_after_images_in_order(
    tmp_path: Path, gemini: _FakeGemini
) -> None:
    store = LocalArtifactStore(tmp_path)
    # Different sizes tell the two images apart once they are re-encoded.
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
    monkeypatch.delenv("GEMMA_API_KEY")
    store = LocalArtifactStore(tmp_path)
    before = _save(_rgb(8, 8), tmp_path / "before.png")
    after = _save(_rgb(8, 8), tmp_path / "after.png")

    with pytest.raises(RuntimeError, match="GEMMA_API_KEY"):
        RSICCTool().run({"image1": before, "image2": after}, store)

    assert gemini.models.calls == []


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


def _gray8() -> Image.Image:
    return _rgb(16, 12).convert("L")


@pytest.mark.parametrize(
    ("filename", "make"),
    [
        ("rgba.png", lambda: _rgb(16, 12).convert("RGBA")),
        ("rgb.jpg", lambda: _rgb(16, 12)),
        ("rgb.jpeg", lambda: _rgb(16, 12)),
        ("palette.gif", lambda: _rgb(16, 12).convert("P")),
        ("rgb.bmp", lambda: _rgb(16, 12)),
        ("rgb.webp", lambda: _rgb(16, 12)),
        ("rgb.tif", lambda: _rgb(16, 12)),
        ("gray.tiff", _gray8),
        ("UPPER.PNG", lambda: _rgb(16, 12)),
    ],
)
def test_supported_formats_reach_gemini_as_png(
    tmp_path: Path, gemini: _FakeGemini, filename: str, make: Any
) -> None:
    store = LocalArtifactStore(tmp_path)
    name = _save(make(), tmp_path / filename)

    RSICCTool().run({"image1": name, "image2": name}, store)

    for part in _sent_images(gemini):
        assert part.inline_data is not None
        assert part.inline_data.mime_type == "image/png"
        assert part.inline_data.data is not None
        assert part.inline_data.data.startswith(_PNG_SIGNATURE)
        assert _decode(part).size == (16, 12)


@pytest.mark.xfail(strict=True, reason="PNG cannot hold float data; convert to 8-bit first")
def test_float32_geotiff_is_supported(tmp_path: Path, gemini: _FakeGemini) -> None:
    reflectance = np.random.default_rng(0).random((12, 16), dtype=np.float32)
    store = LocalArtifactStore(tmp_path)
    name = _save(Image.fromarray(reflectance), tmp_path / "reflectance.tif")

    RSICCTool().run({"image1": name, "image2": name}, store)


@pytest.mark.xfail(strict=True, reason="PNG cannot hold CMYK; convert to RGB first")
def test_cmyk_jpeg_is_supported(tmp_path: Path, gemini: _FakeGemini) -> None:
    store = LocalArtifactStore(tmp_path)
    name = _save(_rgb(16, 12).convert("CMYK"), tmp_path / "print.jpg")

    RSICCTool().run({"image1": name, "image2": name}, store)


@pytest.mark.xfail(strict=True, reason="16-bit data is passed through unstretched and looks black")
def test_16bit_tiff_is_stretched_to_visible_8bit(tmp_path: Path, gemini: _FakeGemini) -> None:
    # Typical surface reflectance counts: 0-4000 out of a possible 65535.
    counts = np.random.default_rng(0).integers(0, 4000, (12, 16), dtype=np.uint16)
    store = LocalArtifactStore(tmp_path)
    name = _save(Image.fromarray(counts), tmp_path / "counts.tif")

    RSICCTool().run({"image1": name, "image2": name}, store)

    sent = _decode(_sent_images(gemini)[0])
    assert sent.mode in {"L", "RGB"}
    assert np.asarray(sent).max() > 200


@pytest.mark.evals
def test_live_gemini_describes_an_obvious_change(tmp_path: Path) -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if not os.environ.get("GEMMA_API_KEY"):
        pytest.skip("GEMMA_API_KEY is not set")

    blank = np.full((128, 128, 3), 255, dtype=np.uint8)
    square = blank.copy()
    square[32:96, 32:96] = (220, 30, 30)
    store = LocalArtifactStore(tmp_path)
    before = _save(Image.fromarray(blank), tmp_path / "before.png")
    after = _save(Image.fromarray(square), tmp_path / "after.png")

    result = RSICCTool().run({"image1": before, "image2": after}, store)

    assert result.text.strip()
    assert "red" in result.text.lower() or "square" in result.text.lower()
