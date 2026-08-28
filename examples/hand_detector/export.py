# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export floating-point or calibrated quantized palm-detector Circle models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import tico
import torch

from examples.hand_detector._support.circle import save_layout_optimized_circle
from examples.hand_detector._support.data import load_npy_inputs, make_synthetic_inputs
from examples.hand_detector._support.quantization import (
    export_quantized_circle,
    quantization_label,
    quantization_name,
    quantize_candidate,
)
from examples.hand_detector._support.verify_circle_layout import verify_circle_layout
from examples.hand_detector._support.verify_circle_resize import (
    read_resize_bilinear_options,
)
from examples.hand_detector._support.verify_quantized_circle import (
    verify_quantized_circle,
)
from examples.hand_detector.hand_detector import (
    load_nhwc_hand_detector,
    lower_resize_bilinear_to_tconv,
)


DIRECTORY = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    """Parse the export mode and its options."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    float_parser = subparsers.add_parser("float", help="Export the FP32 model.")
    _add_model_arguments(float_parser)
    float_parser.add_argument(
        "--output",
        type=Path,
        default=DIRECTORY / "hand_detector_float.circle",
    )
    float_parser.add_argument("--skip-verification", action="store_true")

    quantized = subparsers.add_parser(
        "quantized",
        help="Calibrate and export UINT8 and/or INT16 models.",
    )
    _add_model_arguments(quantized)
    quantized.add_argument("--calibration-dir", type=Path)
    quantized.add_argument("--calibration-offset", type=int, default=0)
    quantized.add_argument("--calibration-limit", type=int)
    quantized.add_argument("--synthetic-calibration-samples", type=int, default=32)
    quantized.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[8, 16],
        choices=[8, 16],
    )
    quantized.add_argument("--output-dir", type=Path, default=DIRECTORY / "exported")
    quantized.add_argument("--output-prefix", default="hand_detector")
    quantized.add_argument(
        "--manifest-json",
        type=Path,
        default=DIRECTORY / "exported" / "manifest.json",
    )
    quantized.add_argument("--skip-verification", action="store_true")
    quantized.add_argument(
        "--resize-tconv",
        action="store_true",
        help=(
            "Lower every 2x half-pixel RESIZE_BILINEAR to an equivalent "
            "fixed-weight TRANSPOSE_CONV."
        ),
    )
    quantized.add_argument(
        "--resize-tconv-groups",
        type=int,
        default=4,
        help=(
            "Split each lowered TRANSPOSE_CONV into this many channel groups "
            "joined by one CONCATENATION. The interpolation weight is "
            "diagonal, so the result is bit-identical while the dense "
            "constant weight shrinks by the group factor."
        ),
    )

    return parser.parse_args()


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--weights",
        type=Path,
        default=DIRECTORY / "hand_detector_float.pt",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DIRECTORY / "hand_detector_spec.json",
    )


def main() -> None:
    """Dispatch floating-point or quantized export."""
    args = parse_args()
    if args.mode == "float":
        _export_float(args)
    elif args.mode == "quantized":
        _export_quantized(args)
    else:
        raise RuntimeError(f"Unhandled export mode: {args.mode}")


def _export_float(args: argparse.Namespace) -> None:
    model = load_nhwc_hand_detector(args.weights, args.spec).eval()
    with torch.inference_mode():
        circle_model = tico.convert(model, model.get_example_inputs())
    output, result = save_layout_optimized_circle(circle_model, args.output)
    print(f"Wrote {output}")
    print(f"Circle layout optimization reported {result.changes} changes.")
    if args.skip_verification:
        return
    layout = verify_circle_layout(output)
    resize_options = read_resize_bilinear_options(output)
    expected = [(False, True), (False, True)]
    if resize_options != expected:
        raise RuntimeError(
            f"Expected ResizeBilinear options {expected}, found {resize_options}."
        )
    print(f"Verified Circle input shapes: {layout['input_shapes']}")
    print(f"Remaining Circle Transpose operators: {layout['transpose_count']}")
    print("Verified 2 Circle RESIZE_BILINEAR operators.")


def _export_quantized(args: argparse.Namespace) -> None:
    model = load_nhwc_hand_detector(args.weights, args.spec).eval()
    lowered_resize_positions: tuple[int, ...] = ()
    if args.resize_tconv:
        lowered_resize_positions = lower_resize_bilinear_to_tconv(
            model.detector,
            groups=args.resize_tconv_groups,
        )
        print(
            "Lowered RESIZE_BILINEAR operations "
            f"{lowered_resize_positions} to TRANSPOSE_CONV "
            f"(groups={args.resize_tconv_groups})."
        )
    if args.calibration_dir is None:
        calibration = make_synthetic_inputs(
            args.synthetic_calibration_samples,
            seed=20260728,
        )
        print("Using synthetic calibration inputs for a smoke test only.")
    else:
        calibration = load_npy_inputs(
            args.calibration_dir,
            args.calibration_limit,
            offset=args.calibration_offset,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models: dict[str, dict[str, Any]] = {}
    for bit_width in args.bits:
        print(f"Preparing {quantization_label(bit_width)} model...")
        candidate = quantize_candidate(model, bit_width, calibration)
        dtype_name = quantization_name(bit_width)
        output = args.output_dir / f"{args.output_prefix}_{dtype_name}.circle"
        export_quantized_circle(candidate, output)
        if args.skip_verification:
            summary: dict[str, Any] = {
                "path": str(output),
                "size_bytes": output.stat().st_size,
                "verification_skipped": True,
            }
        else:
            if args.resize_tconv:
                # Each lowered resize removes one RESIZE_BILINEAR and adds two
                # replicate-padding CONCATENATION operators, plus one channel
                # CONCATENATION when the TRANSPOSE_CONV is split into groups.
                concats_per_resize = 2 + (1 if args.resize_tconv_groups > 1 else 0)
                summary = verify_quantized_circle(
                    output,
                    bit_width,
                    expected_resize_count=0,
                    expected_concat_count=(
                        2 + concats_per_resize * len(lowered_resize_positions)
                    ),
                )
            else:
                summary = verify_quantized_circle(output, bit_width)
            summary["layout"] = verify_circle_layout(output)
        summary["sha256"] = _sha256(output)
        models[dtype_name] = summary
        print(f"Wrote {output} ({output.stat().st_size} bytes).")

    manifest = {
        "input_layout": "NHWC",
        "input_shape": [1, 192, 192, 3],
        "calibration_samples": len(calibration),
        "synthetic_calibration": args.calibration_dir is None,
        "resize_tconv": bool(args.resize_tconv),
        "resize_tconv_groups": (
            int(args.resize_tconv_groups) if args.resize_tconv else None
        ),
        "models": models,
    }
    args.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_json.write_text(
        json.dumps(manifest, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {args.manifest_json}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
