from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

if TYPE_CHECKING:
    from google.genai import types

_MODEL = "gemini-3.5-flash-lite"
_PROMPT = (
    "Compare the visual differences between the before and after images strictly as a "
    "single sentence. Do not say what happened as if it is a news event. Do not search "
    "anything. Refer to the images strictly as the before image and the after image."
)

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


# pydantic model for the inputs to the tool
class RSICCInput(BaseModel):
    image1: str = Field(
        description="Path to the first image (before) relative to the local artifact workspace."
    )
    image2: str = Field(
        description="Path to the second image (after) relative to the local artifact workspace."
    )


class RSICCTool(Tool):
    """Describe the differences between a before and an after image as one sentence of text."""

    name = "rsicc_tool"
    description = "Analyze differences between two images and convert them into text."
    trust = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return RSICCInput

    @staticmethod
    def _image_part(path: Path) -> types.Part:
        from google.genai import types

        mime_type = _MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:  # non-supported image type
            raise ValueError(
                f"Unsupported image type {path.suffix}. Supported types: {', '.join(_MIME_TYPES)}"
            )

        return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Call Gemini to describe the differences between the two images."""
        # imported here so tool discovery still works when google-genai is not installed.
        from google import genai
        from google.genai import types

        request = RSICCInput.model_validate(arguments)
        image_paths = [store.resolve(request.image1), store.resolve(request.image2)]
        for relative, path in zip((request.image1, request.image2), image_paths, strict=True):
            if not path.is_file():
                raise FileNotFoundError(f"Image does not exist: {relative}")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        contents: list[types.PartUnionDict] = [
            "Before image:",
            self._image_part(image_paths[0]),
            "After image:",
            self._image_part(image_paths[1]),
            _PROMPT,
        ]
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
            ),
        )
        if not response.text:
            raise RuntimeError("Gemini returned no text for the image comparison")
        return ToolResult(text=response.text.strip())
