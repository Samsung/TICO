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

"""Verify torch.export or Circle artifacts produced by the hand-detector example."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch

from examples.hand_detector._support.verify_circle_layout import verify_circle_layout
from examples.hand_detector._support.verify_circle_resize import (
    read_resize_bilinear_options,
)
from examples.hand_detector._support.verify_quantized_circle import (
    verify_quantized_circle,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector


DIRECTORY = Path(__file__).resolve().parent
RESIZE_TARGET = "circle_custom.resize_bilinear.default"


def parse_args() -> argparse.Namespace:
    """Parse the artifact type and expected properties."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    torch_parser = subparsers.add_parser("torch", help="Verify torch.export.")
    torch_parser.add_argument(
        "--weights",
        type=Path,
        default=DIRECTORY / "hand_detector_float.pt",
    )
    torch_parser.add_argument(
        "--spec",
        type=Path,
        default=DIRECTORY / "hand_detector_spec.json",
    )

    circle_parser = subparsers.add_parser("circle", help="Verify FP32 Circle layout.")
    circle_parser.add_argument("circle", type=Path)
    circle_parser.add_argument("--expected-resize-count", type=int, default=2)

    quantized = subparsers.add_parser(
        "quantized",
        help="Verify quantized Circle metadata and layout.",
    )
    quantized.add_argument("circle", type=Path)
    quantized.add_argument("--bits", type=int, required=True, choices=[8, 16])
    return parser.parse_args()


def main() -> None:
    """Dispatch the selected verification path."""
    args = parse_args()
    if args.mode == "torch":
        _verify_torch(args.weights, args.spec)
    elif args.mode == "circle":
        _verify_circle(args.circle, args.expected_resize_count)
    elif args.mode == "quantized":
        summary = verify_quantized_circle(args.circle, args.bits)
        summary["layout"] = verify_circle_layout(args.circle)
        print(f"Verified {args.circle}")
    else:
        raise RuntimeError(f"Unhandled verification mode: {args.mode}")


def _verify_torch(weights: Path, spec: Path) -> None:
    model = load_nhwc_hand_detector(weights, spec).eval()
    exported = torch.export.export(model, model.get_example_inputs(), strict=True)
    placeholders = [node for node in exported.graph.nodes if node.op == "placeholder"]
    user_inputs = [
        node
        for node in placeholders
        if tuple(getattr(node.meta.get("val"), "shape", ())) == (1, 192, 192, 3)
    ]
    if len(user_inputs) != 1:
        raise RuntimeError("Expected one NHWC image placeholder [1, 192, 192, 3].")

    call_nodes = [node for node in exported.graph.nodes if node.op == "call_function"]
    counts = Counter(str(node.target) for node in call_nodes)
    resize_nodes = [node for node in call_nodes if str(node.target) == RESIZE_TARGET]
    expected_options = [([12, 12], False, True), ([24, 24], False, True)]
    actual_options = [
        (list(node.args[1]), bool(node.args[2]), bool(node.args[3]))
        for node in resize_nodes
    ]
    if actual_options != expected_options:
        raise RuntimeError(
            f"Expected ResizeBilinear options {expected_options}, "
            f"found {actual_options}."
        )
    expected_counts = {
        "circle_custom.conv2d.padding": 5,
        "circle_custom.depthwise_conv2d.padding": 28,
        "aten.pad.default": 3,
        "aten.cat.default": 2,
    }
    actual_counts = {target: counts[target] for target in expected_counts}
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"Expected structural counts {expected_counts}, found {actual_counts}."
        )
    forbidden = {
        "aten.slice.Tensor",
        "aten.mul.Tensor",
        "aten.upsample_bilinear2d.default",
        "aten.upsample_bilinear2d.vec",
    }
    present = {target: counts[target] for target in forbidden if counts[target]}
    if present:
        raise RuntimeError(f"ResizeBilinear was unexpectedly decomposed: {present}")
    print("Verified NHWC torch.export input and two opaque ResizeBilinear nodes.")


def _verify_circle(path: Path, expected_resize_count: int) -> None:
    layout = verify_circle_layout(path)
    options = read_resize_bilinear_options(path)
    expected_options = [(False, True)] * expected_resize_count
    if options != expected_options:
        raise RuntimeError(
            f"Expected ResizeBilinear options {expected_options}, found {options}."
        )
    print(f"Verified {path}")
    print(f"Input shapes: {layout['input_shapes']}")
    print(f"Remaining Transpose operators: {layout['transpose_count']}")
    print(f"ResizeBilinear operators: {len(options)}")


if __name__ == "__main__":
    main()
