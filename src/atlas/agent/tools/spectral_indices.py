"""Agent tool for NDVI, McFeeters NDWI, and NBR."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

from atlas.agent.artifacts import LocalArtifactStore
from atlas.agent.contracts import Tool, ToolResult
from atlas.data.indices import compute_indices

_LABELS = {
    "ndvi": "NDVI",
    "ndwi_mcfeeters": "NDWI (McFeeters)",
    "nbr": "NBR",
}


class SpectralIndicesInput(BaseModel):
    """Paths to surface-reflectance arrays and optional calibration."""

    red: str = Field(description="Workspace path of the red .npy array.")
    nir: str = Field(description="Workspace path of the NIR .npy array.")
    green: str | None = Field(
        default=None,
        description="Green .npy array. Required for McFeeters NDWI.",
    )
    swir22: str | None = Field(
        default=None,
        description="SWIR2.2 .npy array. Required for NBR.",
    )
    out: str = Field(
        default="indices",
        description="Workspace directory for the index .npy files.",
    )
    nodata: float | dict[str, float] | None = Field(
        default=None,
        description="Fill value, or band name to fill value. Matches become NaN.",
    )
    mask: str | None = Field(
        default=None,
        description="Boolean .npy mask. True marks nodata for every band.",
    )
    scale: float | dict[str, float] | None = Field(
        default=None,
        description=(
            "Multiply stored digital numbers. Surface reflectance is "
            "stored * scale + offset. Sentinel-2 L2A scale is 1e-4."
        ),
    )
    offset: float | dict[str, float] | None = Field(
        default=None,
        description=(
            "Add after scale. Sentinel-2 L2A's published offset of -1000 is "
            "passed as -0.1 with scale 1e-4. Landsat Collection 2 uses "
            "scale 2.75e-5 and offset -0.2."
        ),
    )


class SpectralIndicesTool(Tool):
    """Compute spectral indices from surface-reflectance arrays in the workspace."""

    name = "spectral_indices"
    description = (
        "Compute NDVI, McFeeters NDWI (ndwi_mcfeeters), and NBR from surface "
        "reflectance .npy arrays once red and NIR are available. Optional scale "
        "and offset convert digital numbers (reflectance = stored * scale + offset). "
        "Sentinel-2 L2A: scale 1e-4 and offset -0.1 (published DN offset -1000). "
        "Landsat Collection 2: scale 2.75e-5 and offset -0.2. A nodata fill value "
        "or a boolean mask (True = nodata) becomes NaN."
    )
    trust: ClassVar[Literal["default", "opt_in"]] = "default"

    @property
    def input_model(self) -> type[BaseModel]:
        return SpectralIndicesInput

    def run(self, arguments: dict[str, Any], store: LocalArtifactStore) -> ToolResult:
        request = SpectralIndicesInput.model_validate(arguments)
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
        )
        if not result:
            raise ValueError("red and NIR are required before an index can be computed")
        artifacts: list[str] = []
        lines: list[str] = []
        for name, array in result.items():
            relative = f"{request.out.rstrip('/')}/{name}.npy"
            path = store.resolve(relative)
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
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label} is not a numeric .npy array") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} is not a numeric .npy array")
    return array
