r"""Convert OmniCloudMask V4 weights to the single ONNX file the `cloud_mask` pack pins.

Maintainer step, not part of the Atlas runtime: it has no project dependency and the
torch stack is imported inside functions only. Run it in a throwaway environment:

    uv run --no-project --isolated --with torch==2.15.0 --with onnx \
        --with onnxruntime --with segmentation-models-pytorch==0.6.0 --with safetensors \
        python tools/models/convert_omnicloudmask.py WEIGHTS_DIR OUT.onnx

WEIGHTS_DIR holds the two V4 safetensors files from the upstream Hugging Face repo
`NickWright/OmniCloudMask` (revision pinned in `model.toml`). The graph is the legacy
TorchScript export (opset 17) with dynamic batch, height and width. It takes raw (B,3,H,W)
float32 red, green, NIR, z-scores each band per input with zeros and NaN excluded (upstream
`channel_norm`), runs both models and averages their logits. The script checks the ONNX file
against torch and prints its size and sha256; the pin in `model.toml` follows the file the
owner publishes. Exit status is 1 when the self-parity check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import warnings
from pathlib import Path
from typing import Any

SPECS: tuple[tuple[str, str], ...] = (
    (
        "tu-regnety_004",
        "PM_model_OCM_7.97_R_G_NIR_3_smp_regnety_004.pycls_in1k_PT_state.safetensors",
    ),
    (
        "tu-edgenext_small",
        "PM_model_OCM_7.97_R_G_NIR_3_smp_edgenext_small.usi_in1k_PT_state.safetensors",
    ),
)
OPSET = 17
EXPORT_SIZE = 512
PARITY_SIZES = (512, 256, 384, 509, 1000)
MAX_LOGIT_DIFF = 1e-3


def build_pipeline(weights_dir: Path) -> Any:
    """Return the z-score plus two-model ensemble module (strict state-dict loads)."""
    import segmentation_models_pytorch as smp
    import torch
    from safetensors.torch import load_file

    class Ensemble(torch.nn.Module):  # type: ignore[misc]
        """Mean of raw logits over the models, as upstream `infer_batch`."""

        def __init__(self, models: list[Any]) -> None:
            super().__init__()
            self.models = torch.nn.ModuleList(models)

        def forward(self, x: Any) -> Any:
            return torch.mean(torch.stack([m(x) for m in self.models]), dim=0)

    class Pipeline(torch.nn.Module):  # type: ignore[misc]
        """Raw (B,3,H,W) -> per-input per-band z-score (nodata is 0 or NaN) -> logits."""

        def __init__(self, ensemble: Any) -> None:
            super().__init__()
            self.ensemble = ensemble

        def forward(self, x: Any) -> Any:
            valid = ~(torch.isnan(x) | (x == 0))
            zero = torch.zeros_like(x)
            count = valid.sum(dim=(2, 3), keepdim=True).clamp(min=1)
            mean = torch.where(valid, x, zero).sum(dim=(2, 3), keepdim=True) / count
            centered = torch.where(valid, x - mean, zero)
            std = ((centered**2).sum(dim=(2, 3), keepdim=True) / count).sqrt()
            std = torch.where(std == 0, torch.ones_like(std), std)
            return self.ensemble(centered / std)

    models = []
    for encoder, filename in SPECS:
        model = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=4)
        model.load_state_dict(load_file(str(weights_dir / filename)))
        models.append(model.eval())
    return Pipeline(Ensemble(models)).eval()


def export_onnx(pipeline: Any, out_path: Path) -> None:
    """Write the legacy TorchScript ONNX export with dynamic batch, height and width."""
    import onnx
    import torch

    generator = torch.Generator().manual_seed(0)
    dummy = torch.randn(1, 3, EXPORT_SIZE, EXPORT_SIZE, generator=generator)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            pipeline,
            (dummy,),
            str(out_path),
            dynamo=False,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "b", 2: "h", 3: "w"}, "logits": {0: "b", 2: "h", 3: "w"}},
            opset_version=OPSET,
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(str(out_path)))


def synthetic_scene(size: int, seed: int) -> Any:
    """Smooth multi-scale fields scaled like 0-10000 DN with a nodata corner of zeros."""
    import torch

    g = torch.Generator().manual_seed(seed)
    acc = torch.zeros(1, 3, size, size)
    for cells, amp in ((4, 1.0), (16, 0.5), (64, 0.25)):
        low = torch.rand(1, 3, cells, cells, generator=g)
        up = torch.nn.functional.interpolate(low, size=(size, size), mode="bicubic")
        acc += amp * up
    acc += 0.05 * torch.randn(1, 3, size, size, generator=g)
    acc = (acc - acc.min()) / (acc.max() - acc.min())
    dn = (acc * 9000 + 200).round()
    dn[:, :, : size // 8, : size // 6] = 0
    return dn.float()


def decisive_disagreements(ref: Any, got: Any) -> int:
    """Count argmax disagreements that are not near-ties.

    A pixel whose two best reference logits sit within `MAX_LOGIT_DIFF` of each other may
    flip between runtimes on float rounding alone; it is not a conversion fault.
    """
    import numpy as np

    flipped = ref.argmax(1) != got.argmax(1)
    top_two = np.partition(ref, -2, axis=1)[:, -2:, :, :]
    margin = top_two[:, 1] - top_two[:, 0]
    return int((flipped & (margin > MAX_LOGIT_DIFF)).sum())


def check_parity(pipeline: Any, onnx_path: Path) -> list[dict[str, Any]]:
    """Compare ONNX Runtime with torch on synthetic scenes of several sizes."""
    import onnxruntime as ort
    import torch

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rows: list[dict[str, Any]] = []
    for size in PARITY_SIZES:
        x = synthetic_scene(size, seed=size + 1)
        with torch.no_grad():
            ref = pipeline(x).numpy()
        got = session.run(None, {"input": x.numpy()})[0]
        rows.append(
            {
                "size": size,
                "max_abs_logit_diff": float(abs(ref - got).max()),
                "argmax_disagree_px": decisive_disagreements(ref, got),
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Convert, self-check against torch and print size and sha256."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("weights_dir", type=Path, help="directory with the V4 safetensors files")
    parser.add_argument("out", type=Path, help="output .onnx path")
    args = parser.parse_args(argv)

    pipeline = build_pipeline(args.weights_dir)
    export_onnx(pipeline, args.out)
    rows = check_parity(pipeline, args.out)
    for row in rows:
        print(row)
    ok = all(
        r["argmax_disagree_px"] == 0 and r["max_abs_logit_diff"] < MAX_LOGIT_DIFF for r in rows
    )
    print(f"bytes: {args.out.stat().st_size}")
    print(f"sha256: {sha256_file(args.out)}")
    print("self-parity:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
