"""Agent tool for NDVI, McFeeters NDWI, and NBR."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolDefinition, ToolResult
from atlas.agent.tools._schema import slim_definition
from atlas.data.indices import MAX_BAND_BYTES, compute_indices

_LABELS = {
    "ndvi": "NDVI",
    "ndwi_mcfeeters": "NDWI (McFeeters)",
    "nbr": "NBR",
}


class SpectralIndicesInput(BaseModel):
    """Paths to surface-reflectance arrays and optional calibration."""

    red: str = Field(description="Red .npy path.")
    nir: str = Field(description="NIR .npy array.")
    green: str | None = Field(default=None, description="Green .npy array; needed for NDWI.")
    swir22: str | None = Field(default=None, description="SWIR2.2 .npy array; needed for NBR.")
    out: str = Field(default="indices", description="Output directory.")
    nodata: float | dict[str, float] | None = Field(
        default=None, description="Fill value (or {band: v}) to NaN."
    )
    mask: str | None = Field(default=None, description="Boolean .npy mask; True = nodata.")
    scale: float | dict[str, float] | None = Field(
        default=None,
        description="Multiplier (or {band: v}): stored*scale+offset.",
    )
    offset: float | dict[str, float] | None = Field(
        default=None, description="Added after scale (or {band: v})."
    )
    overwrite: bool = Field(default=False, description="Replace existing index files.")
    preset: Literal["s2-l2a", "landsat-c2"] | None = Field(
        default=None,
        description="Sets scale and offset: s2-l2a 1e-4/-0.1, landsat-c2 2.75e-5/-0.2.",
    )


class SpectralIndicesTool(Tool):
    """Compute spectral indices from surface-reflectance arrays in the workspace."""

    name = "spectral_indices"
    domain: ClassVar[str] = "satellite"
    description = (
        "Legacy NDVI, NDWI (ndwi_mcfeeters) and NBR from surface-reflectance .npy arrays; for "
        "stacks use band_math (index=...). Needs red and NIR. Inputs "
        "are capped at 16777216 elements (memory guard)."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return SpectralIndicesInput

    def definition(self) -> ToolDefinition:
        """Return the advertised schema without null branches or the model docstring."""
        return slim_definition(super().definition())

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = SpectralIndicesInput.model_validate(arguments)
        out_raw = request.out.strip()
        if out_raw == "/":
            raise ValueError("Output directory cannot be filesystem root '/'")
        out_dir = store.resolve(out_raw or ".")
        bands: dict[str, NDArray[np.generic]] = {
            "red": _load(store, request.red, "red"),
            "nir": _load(store, request.nir, "nir"),
        }
        if request.green is not None:
            bands["green"] = _load(store, request.green, "green")
        if request.swir22 is not None:
            bands["swir22"] = _load(store, request.swir22, "swir22")
        mask = _load(store, request.mask, "mask") if request.mask is not None else None
        result = compute_indices(
            bands,
            nodata=request.nodata,
            mask=mask,
            scale=request.scale,
            offset=request.offset,
            preset=request.preset,
        )
        if not result:
            raise ValueError("red and NIR are required before an index can be computed")
        planned: list[tuple[str, Path]] = [(name, out_dir / f"{name}.npy") for name in result]
        for _, path in planned:
            if path.exists() and not request.overwrite:
                raise FileExistsError(
                    f"refusing to overwrite {store.relative(path)}; pass overwrite to replace it"
                )
        artifacts: list[str] = []
        lines: list[str] = []
        for name, path in planned:
            array = result[name]
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array)
            stored = store.relative(path)
            artifacts.append(stored)
            finite = array[np.isfinite(array)]
            if finite.size:
                summary = (
                    f"min {float(finite.min()):.6g}, max {float(finite.max()):.6g}, "
                    f"finite {int(finite.size)}"
                )
            else:
                summary = "no finite pixels"
            lines.append(f"{_LABELS[name]} -> {stored} ({summary})")
        return ToolResult(text="\n".join(lines), artifacts=artifacts)


def _load(store: LocalArtifactStore, relative: str, label: str) -> NDArray[np.generic]:
    """Load one numeric array from the artifact workspace."""
    path = store.resolve_read(relative)
    if not path.is_file():
        raise FileNotFoundError(f"{label} file does not exist: {relative}")
    size = path.stat().st_size
    if size > MAX_BAND_BYTES:
        raise ValueError(f"{label} is {size} bytes, above the limit of {MAX_BAND_BYTES}")
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a numeric .npy array") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} is not a numeric .npy array")
    return array
