from __future__ import annotations

import importlib
import io
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore, SandboxDenied
from atlas.agent.contracts import Tool, ToolResult

if TYPE_CHECKING:
    from google.genai import types

_DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Stronger model used for a second attempt when the first answer looks unreliable.
_DEFAULT_FALLBACK_MODEL = "gemini-3.5-pro"
# Self-reported confidence below this escalates to the fallback model, then to a human.
_DEFAULT_MIN_CONFIDENCE = 0.6

_PROMPT = (
    "Compare the before image and the after image and fill in the JSON fields. "
    "change_detected: whether anything on the ground visibly changed. "
    "change_types: the kinds of change you see, empty if none. "
    "description: the visual differences as a single sentence; do not say what happened "
    "as if it is a news event, and refer to the images strictly as the before image and "
    "the after image. "
    "confidence: from 0 to 1, how sure you are of the answer. "
    "quality_flags: any problem with the images that could make a difference look like "
    "change, such as clouds, the two images not lining up, different sun angles or "
    "shadows, or too little detail. Do not search anything."
)

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
}
_GEOTIFF_SUFFIXES = frozenset({".tif", ".tiff"})

# Geometric pre-checks for GeoTIFF pairs. Failing any of them refuses the request,
# because Gemini would otherwise describe misregistration as change.
_MIN_OVERLAP_FRACTION = 0.5  # of the smaller footprint
_MAX_GSD_RATIO = 1.5  # coarser pixel size / finer pixel size
_MAX_RENDER_PIXELS = 2048  # longest side of the PNG sent to Gemini
# Scene cloud cover (percent) above which the result is flagged for review.
_MAX_CLOUD_COVER = 20.0
# GeoTIFF metadata keys that common catalogs use for scene cloud cover (GDAL cannot
# store a colon in a tag name, so STAC's eo:cloud_cover arrives via the input instead).
_CLOUD_COVER_TAGS = ("cloud_cover", "cloudcover", "cloudy_pixel_percentage")

ChangeType = Literal["vegetation", "water", "built", "terrain", "snow_ice", "other"]
QualityFlag = Literal["cloud_cover", "misalignment", "lighting_difference", "low_resolution"]


class GeoCheckFailed(ValueError):
    """A GeoTIFF pair failed a pre-execution check, so Gemini was never called."""


def _load_genai() -> tuple[Any, Any]:
    """Import the optional Gemini SDK."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for rsicc_tool. Install it with `uv sync --extra gemini`."
        ) from exc
    return genai, types


def _load_rasterio() -> Any:
    """Import the optional rasterio package, only needed for GeoTIFF inputs."""
    try:
        rasterio = importlib.import_module("rasterio")
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required for GeoTIFF inputs to rsicc_tool. Install it with "
            "`uv pip install rasterio`."
        ) from exc
    return rasterio


# pydantic model for the inputs to the tool
class RSICCInput(BaseModel):
    image1: str = Field(
        description="Path to the first image (before) relative to the local artifact workspace."
    )
    image2: str = Field(
        description="Path to the second image (after) relative to the local artifact workspace."
    )
    model: str | None = Field(
        default=None,
        description="Gemini model to use first. Leave empty to use the configured default.",
    )
    before_cloud_cover: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Catalog cloud cover percentage of the before scene, if known.",
    )
    after_cloud_cover: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Catalog cloud cover percentage of the after scene, if known.",
    )


class ChangeReport(BaseModel):
    """The structured answer Gemini is constrained to return."""

    change_detected: bool
    change_types: list[ChangeType]
    description: str
    # Self-rated by the model, so weakly calibrated. A placeholder until confidence comes
    # from something external, like agreement between runs or with a change detector.
    confidence: float = Field(ge=0, le=1)
    quality_flags: list[QualityFlag]


class Attempt(BaseModel):
    model: str
    report: ChangeReport


class GeoCheck(BaseModel):
    """What the GeoTIFF pre-checks measured. Absent for plain PNG/JPEG inputs."""

    crs: str
    overlap_fraction: float
    before_gsd: float
    after_gsd: float
    before_cloud_cover: float | None
    after_cloud_cover: float | None


class RSICCResult(BaseModel):
    """The tool's output contract, serialized as JSON into ``ToolResult.text``."""

    report: ChangeReport
    model: str
    escalated: bool
    needs_review: bool
    review_reasons: list[str]
    attempts: list[Attempt]
    geo_check: GeoCheck | None = None


def _cloud_cover_from_tags(tags: dict[str, str]) -> float | None:
    lowered = {key.lower(): value for key, value in tags.items()}
    for key in _CLOUD_COVER_TAGS:
        if key in lowered:
            try:
                return float(lowered[key])
            except ValueError:
                return None
    return None


def _render_band_stack(
    dataset: Any, bounds: tuple[float, float, float, float], shape: Any
) -> bytes:
    """Read the overlap window of a GeoTIFF and encode it as a contrast-stretched PNG."""
    import numpy as np
    from PIL import Image

    # Imported by name: rasterio is optional and ships no type stubs.
    enums = importlib.import_module("rasterio.enums")
    from_bounds = importlib.import_module("rasterio.windows").from_bounds
    ColorInterp, Resampling = enums.ColorInterp, enums.Resampling

    interp = {color: index + 1 for index, color in enumerate(dataset.colorinterp)}
    rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    if all(color in interp for color in rgb):
        bands = [interp[color] for color in rgb]
    elif dataset.count >= 3:
        bands = [1, 2, 3]
    else:
        bands = [1]

    height, width = shape
    data = dataset.read(
        bands,
        window=from_bounds(*bounds, transform=dataset.transform),
        out_shape=(len(bands), height, width),
        masked=True,
        resampling=Resampling.bilinear,
    )
    channels = []
    for band in data:
        valid = band.compressed()
        if valid.size == 0:
            raise GeoCheckFailed(f"{Path(dataset.name).name} has no valid pixels in the overlap")
        low, high = np.percentile(valid, (2, 98))
        scaled = (band.astype("float64") - low) / max(float(high - low), 1e-12)
        channels.append((np.clip(scaled.filled(0), 0, 1) * 255).astype(np.uint8))

    pixels = channels[0] if len(channels) == 1 else np.stack(channels, axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _check_and_render_geotiffs(
    before: Path, after: Path, request: RSICCInput
) -> tuple[bytes, bytes, GeoCheck]:
    """Verify a GeoTIFF pair lines up, then render the shared footprint of each to PNG."""
    rasterio = _load_rasterio()
    with rasterio.open(before) as first, rasterio.open(after) as second:
        for dataset, label in ((first, "before"), (second, "after")):
            if dataset.crs is None:
                raise GeoCheckFailed(f"The {label} image has no coordinate reference system")
        if first.crs != second.crs:
            raise GeoCheckFailed(
                f"The images use different coordinate reference systems "
                f"({first.crs} vs {second.crs}). Reproject one to match the other first."
            )

        a, b = first.bounds, second.bounds
        left, bottom = max(a.left, b.left), max(a.bottom, b.bottom)
        right, top = min(a.right, b.right), min(a.top, b.top)
        if right <= left or top <= bottom:
            raise GeoCheckFailed("The before and after images do not overlap on the ground")
        overlap = (right - left) * (top - bottom)
        smaller = min(
            (a.right - a.left) * (a.top - a.bottom), (b.right - b.left) * (b.top - b.bottom)
        )
        overlap_fraction = overlap / smaller
        if overlap_fraction < _MIN_OVERLAP_FRACTION:
            raise GeoCheckFailed(
                f"The images only share {overlap_fraction:.0%} of their footprint; at least "
                f"{_MIN_OVERLAP_FRACTION:.0%} is required"
            )

        before_gsd = max(abs(first.res[0]), abs(first.res[1]))
        after_gsd = max(abs(second.res[0]), abs(second.res[1]))
        if max(before_gsd, after_gsd) / min(before_gsd, after_gsd) > _MAX_GSD_RATIO:
            raise GeoCheckFailed(
                f"The images have very different pixel sizes ({before_gsd:g} vs {after_gsd:g} "
                f"{first.crs.linear_units or 'units'}). Resample them to a similar resolution first."
            )

        # Render both at the coarser resolution so they cover identical pixels.
        coarse = max(before_gsd, after_gsd)
        width, height = (right - left) / coarse, (top - bottom) / coarse
        scale = min(1.0, _MAX_RENDER_PIXELS / max(width, height))
        shape = (max(1, round(height * scale)), max(1, round(width * scale)))
        bounds = (left, bottom, right, top)
        before_png = _render_band_stack(first, bounds, shape)
        after_png = _render_band_stack(second, bounds, shape)

        geo_check = GeoCheck(
            crs=str(first.crs),
            overlap_fraction=round(overlap_fraction, 4),
            before_gsd=before_gsd,
            after_gsd=after_gsd,
            before_cloud_cover=(
                request.before_cloud_cover
                if request.before_cloud_cover is not None
                else _cloud_cover_from_tags(first.tags())
            ),
            after_cloud_cover=(
                request.after_cloud_cover
                if request.after_cloud_cover is not None
                else _cloud_cover_from_tags(second.tags())
            ),
        )
    return before_png, after_png, geo_check


class RSICCTool(Tool):
    """Describe the changes between a before and an after image as a structured report.

    Uploads both workspace images to the Gemini API, so the tool stays off unless the
    caller opts in. GeoTIFF pairs are checked for matching CRS, overlapping footprints
    and similar pixel size before anything is sent. Low-confidence or flagged answers
    are retried on a stronger model and then marked for human review.
    """

    name = "rsicc_tool"
    description = (
        "Compare a before and an after image (PNG/JPEG/WebP/HEIC, or a GeoTIFF pair) and "
        "return a JSON change report with change types, a one-sentence description, "
        "confidence, quality flags, and whether a human should review it."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "opt_in"
    # Inline Gemini requests cap the payload. Refuse anything larger before reading it.
    max_bytes: ClassVar[int] = 10_000_000

    def __init__(
        self,
        model: str | None = None,
        fallback_model: str | None = None,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        self.model = model or os.getenv("ATLAS_RSICC_MODEL") or _DEFAULT_MODEL
        self.fallback_model = (
            fallback_model or os.getenv("ATLAS_RSICC_FALLBACK_MODEL") or _DEFAULT_FALLBACK_MODEL
        )
        self.min_confidence = min_confidence

    @property
    def input_model(self) -> type[BaseModel]:
        return RSICCInput

    @staticmethod
    def _check_size(size: int, relative: str) -> None:
        if size > RSICCTool.max_bytes:
            raise SandboxDenied(
                f"Image {relative} exceeds the {RSICCTool.max_bytes} byte read limit"
            )

    @staticmethod
    def _image_part(path: Path, relative: str) -> types.Part:
        _, genai_types = _load_genai()

        mime_type = _MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:  # non-supported image type
            supported = [*_MIME_TYPES, *sorted(_GEOTIFF_SUFFIXES)]
            raise ValueError(
                f"Unsupported image type {path.suffix}. Supported types: {', '.join(supported)}"
            )
        RSICCTool._check_size(path.stat().st_size, relative)

        part: types.Part = genai_types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)
        return part

    @staticmethod
    def _png_part(data: bytes, relative: str) -> types.Part:
        _, genai_types = _load_genai()
        RSICCTool._check_size(len(data), relative)
        part: types.Part = genai_types.Part.from_bytes(data=data, mime_type="image/png")
        return part

    def _needs_escalation(self, report: ChangeReport) -> bool:
        return report.confidence < self.min_confidence or bool(report.quality_flags)

    @staticmethod
    def _ask(client: Any, genai_types: Any, model: str, contents: list[Any]) -> ChangeReport:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ChangeReport,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, ChangeReport):
            return parsed
        if not response.text:
            raise RuntimeError("Gemini returned no text for the image comparison")
        return ChangeReport.model_validate_json(response.text)

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        """Call Gemini to report the changes between the two images."""
        request = RSICCInput.model_validate(arguments)
        relatives = (request.image1, request.image2)
        image_paths = [store.resolve(request.image1), store.resolve(request.image2)]
        for relative, path in zip(relatives, image_paths, strict=True):
            if not path.is_file():
                raise FileNotFoundError(f"Image does not exist: {relative}")

        is_geotiff = [path.suffix.lower() in _GEOTIFF_SUFFIXES for path in image_paths]
        if any(is_geotiff) and not all(is_geotiff):
            raise GeoCheckFailed(
                "Both images must be GeoTIFFs, or neither; a GeoTIFF cannot be checked for "
                "alignment against an image without coordinates"
            )

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        # Imported here so tool discovery still works when the gemini extra is absent.
        genai, genai_types = _load_genai()
        geo_check: GeoCheck | None = None
        if all(is_geotiff):
            before_png, after_png, geo_check = _check_and_render_geotiffs(
                image_paths[0], image_paths[1], request
            )
            parts = [
                self._png_part(before_png, relatives[0]),
                self._png_part(after_png, relatives[1]),
            ]
        else:
            parts = [
                self._image_part(path, rel)
                for path, rel in zip(image_paths, relatives, strict=True)
            ]

        contents: list[types.PartUnionDict] = [
            "Before image:",
            parts[0],
            "After image:",
            parts[1],
            _PROMPT,
        ]
        client = genai.Client(api_key=api_key)

        first_model = request.model or self.model
        attempts = [
            Attempt(model=first_model, report=self._ask(client, genai_types, first_model, contents))
        ]
        escalated = False
        if self._needs_escalation(attempts[0].report) and self.fallback_model != first_model:
            escalated = True
            report = self._ask(client, genai_types, self.fallback_model, contents)
            attempts.append(Attempt(model=self.fallback_model, report=report))
        final = attempts[-1]

        # Fold scene metadata into the flags so downstream code checks one list.
        flags = list(final.report.quality_flags)
        cloud_covers = (
            [geo_check.before_cloud_cover, geo_check.after_cloud_cover]
            if geo_check
            else [request.before_cloud_cover, request.after_cloud_cover]
        )
        if (
            any(c is not None and c > _MAX_CLOUD_COVER for c in cloud_covers)
            and "cloud_cover" not in flags
        ):
            flags.append("cloud_cover")
        report = final.report.model_copy(update={"quality_flags": flags})

        review_reasons = []
        if report.confidence < self.min_confidence:
            review_reasons.append(
                f"confidence {report.confidence:.2f} is below {self.min_confidence:.2f}"
            )
        if report.quality_flags:
            review_reasons.append(f"quality flags: {', '.join(report.quality_flags)}")

        result = RSICCResult(
            report=report,
            model=final.model,
            escalated=escalated,
            needs_review=bool(review_reasons),
            review_reasons=review_reasons,
            attempts=attempts,
            geo_check=geo_check,
        )
        return ToolResult(text=result.model_dump_json())
