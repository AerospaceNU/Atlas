"""Opt-in Gemini change-captioning tool for plain images and GeoTIFF pairs."""

from __future__ import annotations

import importlib
import io
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from atlas.agent.artifacts import ArtifactError, LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult

if TYPE_CHECKING:
    import numpy as np
    from google.genai import types

_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_DEFAULT_FALLBACK_MODEL = "gemini-3.6-flash"
_DEFAULT_MIN_CONFIDENCE = 0.6
_MAX_DESCRIPTION_CHARS = 400
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
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
_MIN_OVERLAP_FRACTION = 0.5
_MAX_GSD_RATIO = 1.5
_MAX_RENDER_PIXELS = 2048
_MAX_CLOUD_COVER = 20.0
_CLOUD_COVER_TAGS = ("cloud_cover", "cloudcover", "cloudy_pixel_percentage")

ChangeType = Literal["vegetation", "water", "built", "terrain", "snow_ice", "other"]
QualityFlag = Literal["cloud_cover", "misalignment", "lighting_difference", "low_resolution"]
_NON_RETRYABLE_FLAGS: frozenset[str] = frozenset({"cloud_cover"})


class GeoCheckFailed(ValueError):
    """A GeoTIFF pair failed a pre-execution check before calling Gemini."""


class InvalidModelAnswer(RuntimeError):
    """Gemini answered with nothing or an invalid change report."""


def _load_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for rsicc_tool. Install it with `uv sync --extra gemini`."
        ) from exc
    return genai, types


def _load_rasterio() -> Any:
    try:
        return importlib.import_module("rasterio")
    except ImportError as exc:
        raise RuntimeError(
            "rasterio is required for GeoTIFF inputs to rsicc_tool. Install it with "
            "`uv sync --extra geotiff`."
        ) from exc


class RSICCInput(BaseModel):
    image1: str = Field(description="Path to the before image.")
    image2: str = Field(description="Path to the after image.")
    model: str | None = Field(
        default=None, description="Configured default or fallback Gemini model."
    )
    bands: list[int] | None = Field(
        default=None,
        min_length=1,
        max_length=3,
        description="GeoTIFF only: 1-based [R, G, B] or [grey] band indices.",
    )
    before_cloud_cover: float | None = Field(
        default=None, ge=0, le=100, description="Before scene cloud %."
    )
    after_cloud_cover: float | None = Field(
        default=None, ge=0, le=100, description="After scene cloud %."
    )

    @field_validator("bands")
    @classmethod
    def _bands_are_rgb_or_grey(cls, bands: list[int] | None) -> list[int] | None:
        if bands is not None:
            if len(bands) == 2:
                raise ValueError("bands must list one band (greyscale) or three (red, green, blue)")
            if min(bands) < 1:
                raise ValueError("band indices are 1-based")
        return bands


class ChangeReport(BaseModel):
    change_detected: bool
    change_types: list[ChangeType]
    description: str
    confidence: float = Field(ge=0, le=1)
    quality_flags: list[QualityFlag]

    @field_validator("description")
    @classmethod
    def _cap_description(cls, description: str) -> str:
        description = " ".join(description.split())
        if len(description) > _MAX_DESCRIPTION_CHARS:
            description = description[: _MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
        return description

    @model_validator(mode="after")
    def _types_need_a_change(self) -> Self:
        if not self.change_detected and self.change_types:
            raise ValueError("change_types must be empty when change_detected is false")
        return self


class Attempt(BaseModel):
    model: str
    report: ChangeReport | None = None
    error: str | None = None


class GeoCheck(BaseModel):
    crs: str
    overlap_fraction: float
    before_gsd: float
    after_gsd: float
    before_cloud_cover: float | None
    after_cloud_cover: float | None


class RSICCResult(BaseModel):
    report: ChangeReport
    model: str
    escalated: bool
    needs_review: bool
    review_reasons: list[str]
    attempts: list[Attempt]
    geo_check: GeoCheck | None = None


def _cloud_cover_from_tags(tags: dict[str, str]) -> float | None:
    """Read a valid 0-100 cloud cover percentage from GeoTIFF tags, trying all candidate keys."""
    lowered = {key.lower(): value for key, value in tags.items()}
    for key in _CLOUD_COVER_TAGS:
        raw = lowered.get(key)
        if raw is None:
            continue
        try:
            value = float(raw.strip().removesuffix("%"))
        except ValueError:
            continue
        if 0 <= value <= 100:
            return value
    return None


def _select_bands(dataset: Any, label: str, bands: list[int] | None) -> list[int]:
    if bands is not None:
        if max(bands) > dataset.count:
            raise GeoCheckFailed(
                f"The {label} image has {dataset.count} bands, so band {max(bands)} does not exist"
            )
        return list(bands)
    ColorInterp = importlib.import_module("rasterio.enums").ColorInterp
    interp = {color: index + 1 for index, color in enumerate(dataset.colorinterp)}
    rgb = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
    if all(color in interp for color in rgb):
        return [interp[color] for color in rgb]
    if dataset.count == 1:
        return [1]
    raise GeoCheckFailed(
        f"The {label} image has {dataset.count} bands and no red/green/blue colour tags, so "
        "the visible bands are unknown. Pass `bands` with 1-based RGB or greyscale indices."
    )


def _read_bands(
    dataset: Any,
    label: str,
    bands: list[int],
    bounds: tuple[float, float, float, float],
    shape: tuple[int, int],
) -> np.ma.MaskedArray[Any, Any]:
    import numpy as np

    enums = importlib.import_module("rasterio.enums")
    from_bounds = importlib.import_module("rasterio.windows").from_bounds
    height, width = shape
    data = dataset.read(
        bands,
        window=from_bounds(*bounds, transform=dataset.transform),
        out_shape=(len(bands), height, width),
        masked=True,
        resampling=enums.Resampling.bilinear,
    )
    data = np.ma.masked_invalid(data.astype("float64"))
    if any(band.count() == 0 for band in data):
        raise GeoCheckFailed(f"The {label} image has no valid pixels in the overlap")
    masked: np.ma.MaskedArray[Any, Any] = data
    return masked


def _stretch_and_encode(
    before: np.ma.MaskedArray[Any, Any], after: np.ma.MaskedArray[Any, Any], max_bytes: int
) -> tuple[bytes, bytes]:
    import numpy as np
    from PIL import Image

    stretched: tuple[list[Any], list[Any]] = ([], [])
    for b_band, a_band in zip(before, after, strict=True):
        valid = np.concatenate([b_band.compressed(), a_band.compressed()])
        low, high = np.nanpercentile(valid, (2, 98))
        span = max(float(high - low), 1e-12)
        for band, out in ((b_band, stretched[0]), (a_band, stretched[1])):
            out.append((np.clip((band - low) / span, 0, 1).filled(0) * 255).astype(np.uint8))

    imgs = [Image.fromarray(ch[0] if len(ch) == 1 else np.stack(ch, axis=-1)) for ch in stretched]
    while True:
        encoded = []
        for img in imgs:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            encoded.append(buf.getvalue())
        largest = max(len(d) for d in encoded)
        if largest <= max_bytes:
            return encoded[0], encoded[1]
        w, h = imgs[0].size
        if (w, h) == (1, 1):
            raise GeoCheckFailed(f"The rendered images cannot fit the {max_bytes} byte limit")
        factor = math.sqrt(max_bytes / largest) * 0.9
        size = (max(1, int(w * factor)), max(1, int(h * factor)))
        imgs = [img.resize(size, Image.Resampling.BOX) for img in imgs]


def _check_and_render_geotiffs(
    before: Path, after: Path, request: RSICCInput, max_bytes: int
) -> tuple[bytes, bytes, GeoCheck]:
    for path, relative in ((before, request.image1), (after, request.image2)):
        with path.open("rb") as handle:
            if handle.read(4) not in _TIFF_MAGIC:
                raise GeoCheckFailed(f"{relative} is not a TIFF file")

    rasterio = _load_rasterio()
    with rasterio.open(before) as first, rasterio.open(after) as second:
        for ds, label in ((first, "before"), (second, "after")):
            if ds.driver != "GTiff":
                raise GeoCheckFailed(f"The {label} image is not a GeoTIFF ({ds.driver})")
            if ds.crs is None:
                raise GeoCheckFailed(f"The {label} image has no coordinate reference system")
        if first.crs != second.crs:
            raise GeoCheckFailed(
                f"The images use different coordinate reference systems ({first.crs} vs {second.crs})."
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
        if smaller <= 0:
            raise GeoCheckFailed("One of the GeoTIFF images has zero ground area")
        overlap_fraction = overlap / smaller
        if overlap_fraction < _MIN_OVERLAP_FRACTION:
            raise GeoCheckFailed(
                f"The images only share {overlap_fraction:.0%} of their footprint; "
                f"at least {_MIN_OVERLAP_FRACTION:.0%} is required"
            )

        before_gsd = max(abs(first.res[0]), abs(first.res[1]))
        after_gsd = max(abs(second.res[0]), abs(second.res[1]))
        if before_gsd <= 0 or after_gsd <= 0:
            raise GeoCheckFailed("GeoTIFF image has zero pixel resolution")
        if max(before_gsd, after_gsd) / min(before_gsd, after_gsd) > _MAX_GSD_RATIO:
            raise GeoCheckFailed(
                f"The images have very different pixel sizes ({before_gsd:g} vs {after_gsd:g} "
                f"{first.crs.linear_units or 'units'})."
            )

        b_bands = _select_bands(first, "before", request.bands)
        a_bands = _select_bands(second, "after", request.bands)
        if len(b_bands) != len(a_bands):
            raise GeoCheckFailed(
                "One image renders as colour and the other as greyscale; pass `bands`."
            )

        coarse = max(before_gsd, after_gsd)
        width, height = (right - left) / coarse, (top - bottom) / coarse
        scale = min(1.0, _MAX_RENDER_PIXELS / max(width, height))
        shape = (max(1, round(height * scale)), max(1, round(width * scale)))
        bounds = (left, bottom, right, top)
        before_png, after_png = _stretch_and_encode(
            _read_bands(first, "before", b_bands, bounds, shape),
            _read_bands(second, "after", a_bands, bounds, shape),
            max_bytes,
        )
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
    """Compare a before and after image via Gemini and return a structured JSON change report."""

    name = "rsicc_tool"
    domain: ClassVar[str] = "satellite"
    description = (
        "Compare before/after images (PNG/JPEG/WebP/HEIC or GeoTIFF pair) via Gemini and return "
        "a JSON change report (untrusted model description, confidence, quality flags)."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "opt_in"
    max_bytes: ClassVar[int] = 10_000_000
    api_timeout_seconds: ClassVar[float] = 60.0

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

    def _read_image(self, path: Path, relative: str) -> tuple[bytes, str]:
        mime_type = _MIME_TYPES.get(path.suffix.lower())
        if mime_type is None:
            supported = [*_MIME_TYPES, *sorted(_GEOTIFF_SUFFIXES)]
            raise ValueError(
                f"Unsupported image type {path.suffix}. Supported types: {', '.join(supported)}"
            )
        if path.stat().st_size > self.max_bytes:
            raise ArtifactError(f"Image {relative} exceeds the {self.max_bytes} byte read limit")
        return path.read_bytes(), mime_type

    def _needs_escalation(self, report: ChangeReport) -> bool:
        retryable = {flag for flag in report.quality_flags if flag not in _NON_RETRYABLE_FLAGS}
        return report.confidence < self.min_confidence or bool(retryable)

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
            raise InvalidModelAnswer("Gemini returned no text for the image comparison")
        try:
            return ChangeReport.model_validate_json(response.text)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, err['loc'])) or 'answer'}: {err['msg']}"
                for err in exc.errors()
            )
            raise InvalidModelAnswer(
                f"Gemini returned an invalid change report ({problems})"
            ) from exc

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = RSICCInput.model_validate(arguments)
        first_model = request.model or self.model
        allowed = (self.model, self.fallback_model)
        if first_model not in allowed:
            raise ValueError(
                f"Model {first_model!r} is not allowed. Use one of: {', '.join(allowed)}"
            )

        relatives = (request.image1, request.image2)
        image_paths = [store.resolve_read(request.image1), store.resolve_read(request.image2)]
        for relative, path in zip(relatives, image_paths, strict=True):
            if not path.is_file():
                raise FileNotFoundError(f"Image does not exist: {relative}")

        is_geotiff = [path.suffix.lower() in _GEOTIFF_SUFFIXES for path in image_paths]
        if any(is_geotiff) and not all(is_geotiff):
            raise GeoCheckFailed(
                "Both images must be GeoTIFFs, or neither; a GeoTIFF cannot be checked for "
                "alignment against an image without coordinates"
            )

        geo_check: GeoCheck | None = None
        if all(is_geotiff):
            before_png, after_png, geo_check = _check_and_render_geotiffs(
                image_paths[0], image_paths[1], request, self.max_bytes
            )
            images = [(before_png, "image/png"), (after_png, "image/png")]
        else:
            images = [
                self._read_image(p, rel) for p, rel in zip(image_paths, relatives, strict=True)
            ]

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        genai, genai_types = _load_genai()
        parts: list[types.Part] = [
            genai_types.Part.from_bytes(data=data, mime_type=mime) for data, mime in images
        ]
        contents: list[types.PartUnionDict] = [
            "Before image:",
            parts[0],
            "After image:",
            parts[1],
            _PROMPT,
        ]
        client = genai.Client(
            api_key=api_key, http_options={"timeout": int(self.api_timeout_seconds * 1000)}
        )

        cloud_covers = (
            [geo_check.before_cloud_cover, geo_check.after_cloud_cover]
            if geo_check
            else [request.before_cloud_cover, request.after_cloud_cover]
        )
        cloudy = any(c is not None and c > _MAX_CLOUD_COVER for c in cloud_covers)

        attempts: list[Attempt] = []
        try:
            attempts.append(
                Attempt(
                    model=first_model, report=self._ask(client, genai_types, first_model, contents)
                )
            )
        except InvalidModelAnswer as exc:
            attempts.append(Attempt(model=first_model, error=str(exc)))

        first_report = attempts[0].report
        escalated = False
        if (
            first_report is None or self._needs_escalation(first_report)
        ) and self.fallback_model != first_model:
            escalated = True
            try:
                attempts.append(
                    Attempt(
                        model=self.fallback_model,
                        report=self._ask(client, genai_types, self.fallback_model, contents),
                    )
                )
            except Exception as exc:
                attempts.append(
                    Attempt(model=self.fallback_model, error=f"{type(exc).__name__}: {exc}")
                )

        answered = [i for i, a in enumerate(attempts) if a.report]
        if not answered:
            errors = "; ".join(f"{a.model}: {a.error}" for a in attempts)
            raise InvalidModelAnswer(f"Gemini gave no usable change report ({errors})")
        final = attempts[answered[-1]]
        assert final.report is not None

        flags = list(final.report.quality_flags)
        if cloudy and "cloud_cover" not in flags:
            flags.append("cloud_cover")
        report = final.report.model_copy(update={"quality_flags": flags})

        review_reasons = [
            f"fallback model {a.model} failed, so the first answer was kept: {a.error}"
            for a in attempts[answered[-1] + 1 :]
        ]
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
        return ToolResult(
            text=result.model_dump_json(),
            summary=f"change={report.change_detected} ({final.model})",
        )
