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

"""Analyze conditional regular/depthwise W8 sensitivity under A16 activations."""

from __future__ import annotations

import argparse
import json
import math

from pathlib import Path

import torch
from tico.quantization.analysis import make_output_adapter

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector._support.conditional_weight_sensitivity import (
    print_conditional_weight_sensitivity,
    run_conditional_weight_sensitivity,
)
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)
_EXPECTED_TARGET = {
    "regular-float": "depthwise",
    "depthwise-float": "regular",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    command: str = "conditional-weight-sensitivity",
) -> argparse.ArgumentParser:
    """Register the command with ``examples.hand_detector.analyze``."""
    parser = subparsers.add_parser(
        command,
        help=(
            "Analyze depthwise weights under F1 regular-FP or regular weights "
            "under F2 depthwise-FP, with independent and greedy ranking."
        ),
    )
    _add_arguments(parser)
    return parser


def _add_arguments(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument("--calibration-dir", type=Path)
    parser.add_argument("--calibration-offset", type=int, default=0)
    parser.add_argument("--calibration-limit", type=int)
    parser.add_argument("--synthetic-calibration-samples", type=int, default=200)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--evaluation-offset", type=int, default=0)
    parser.add_argument("--evaluation-limit", type=int)
    parser.add_argument("--synthetic-evaluation-samples", type=int, default=79)
    parser.add_argument("--require-disjoint", action="store_true")
    parser.add_argument("--uint8-percentile", type=float, default=99.99)
    parser.add_argument(
        "--int16-observer",
        choices=("minmax", "percentile"),
        default="minmax",
    )
    parser.add_argument("--int16-percentile", type=float, default=99.99)
    parser.add_argument("--max-samples", type=int, default=524_288)
    parser.add_argument("--samples-per-batch", type=int, default=4_096)
    parser.add_argument("--sampling-seed", type=int, default=20260803)
    parser.add_argument(
        "--baseline-family",
        choices=("regular-float", "depthwise-float"),
        required=True,
        help="Use F1 regular-FP or F2 depthwise-FP as the fixed baseline.",
    )
    parser.add_argument(
        "--target-family",
        choices=("regular", "depthwise"),
        help=(
            "Optional explicit target-family check. It must be depthwise for "
            "regular-float and regular for depthwise-float."
        ),
    )
    parser.add_argument(
        "--granularity",
        choices=("semantic", "site"),
        default="semantic",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Optional ordered subset of target semantic or site group names.",
    )
    parser.add_argument(
        "--skip-greedy",
        action="store_true",
        help="Run only independent conditional leave-one-group-float ranking.",
    )
    parser.add_argument(
        "--max-greedy-steps",
        type=int,
        default=0,
        help="Maximum greedy selections; zero allows every target group.",
    )
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=0.0,
        help="Minimum incremental regressor MAE improvement for a greedy step.",
    )
    parser.add_argument(
        "--auxiliary-tolerance",
        type=float,
        default=0.0,
        help="Maximum incremental classifier MAE regression per greedy step.",
    )
    parser.add_argument("--target-regressor-mae", type=float, default=0.1)
    parser.add_argument("--target-classifier-mae", type=float, default=0.1)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Independent rows printed to the console; zero prints all groups.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=(DIRECTORY / "reports" / "conditional_weight_sensitivity.json"),
    )


def main() -> None:
    """Run the standalone command."""
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
    """Load data, run the analysis, print it, and write JSON."""
    _validate_args(args)
    _validate_disjoint(args)
    device = torch.device(args.device)
    float_model = load_nhwc_hand_detector(args.weights, args.spec).to(device).eval()
    calibration = _load_samples(
        args.calibration_dir,
        args.calibration_limit,
        args.calibration_offset,
        args.synthetic_calibration_samples,
        args.sampling_seed,
    )
    evaluation = _load_samples(
        args.evaluation_dir,
        args.evaluation_limit,
        args.evaluation_offset,
        args.synthetic_evaluation_samples,
        args.sampling_seed + 1,
    )
    calibration = tuple(sample.to(device=device) for sample in calibration)
    evaluation = tuple(sample.to(device=device) for sample in evaluation)
    report = run_conditional_weight_sensitivity(
        float_model,
        calibration,
        evaluation,
        baseline_family=args.baseline_family,
        uint8_percentile=args.uint8_percentile,
        int16_observer=args.int16_observer,
        int16_percentile=args.int16_percentile,
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        sampling_seed=args.sampling_seed,
        requested_groups=args.groups,
        granularity=args.granularity,
        run_greedy=not args.skip_greedy,
        max_greedy_steps=args.max_greedy_steps,
        minimum_improvement=args.minimum_improvement,
        auxiliary_tolerance=args.auxiliary_tolerance,
        target_regressor_mae=args.target_regressor_mae,
        target_classifier_mae=args.target_classifier_mae,
        output_adapter=OUTPUT_ADAPTER,
    )
    report["metadata"].update(
        {
            "calibration_samples": len(calibration),
            "evaluation_samples": len(evaluation),
            "device": str(device),
        }
    )
    print_conditional_weight_sensitivity(report, top_k=args.top_k)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nWrote {args.report_json}")


def _load_samples(
    directory: Path | None,
    limit: int | None,
    offset: int,
    synthetic_count: int,
    seed: int,
):
    if directory is None:
        return make_synthetic_inputs(synthetic_count, seed=seed)
    return load_npy_inputs(directory, limit, offset=offset)


def _validate_args(args: argparse.Namespace) -> None:
    expected = _EXPECTED_TARGET[args.baseline_family]
    if args.target_family is not None and args.target_family != expected:
        raise ValueError(
            f"--baseline-family {args.baseline_family} requires "
            f"--target-family {expected}."
        )
    if args.max_greedy_steps < 0 or args.top_k < 0:
        raise ValueError("Greedy step and top-k counts must be nonnegative.")
    for name, value in (
        ("minimum_improvement", args.minimum_improvement),
        ("auxiliary_tolerance", args.auxiliary_tolerance),
        ("target_regressor_mae", args.target_regressor_mae),
        ("target_classifier_mae", args.target_classifier_mae),
    ):
        if not math.isfinite(value):
            raise ValueError(f"--{name.replace('_', '-')} must be finite.")
    if args.minimum_improvement < 0.0 or args.auxiliary_tolerance < 0.0:
        raise ValueError("Improvement and tolerance must be nonnegative.")
    if args.target_regressor_mae <= 0.0 or args.target_classifier_mae <= 0.0:
        raise ValueError("Target MAEs must be positive.")


def _validate_disjoint(args: argparse.Namespace) -> None:
    if not args.require_disjoint:
        return
    if args.calibration_dir is None or args.evaluation_dir is None:
        return
    calibration_paths = list_npy_inputs(args.calibration_dir)[
        args.calibration_offset : (
            None
            if args.calibration_limit is None
            else args.calibration_offset + args.calibration_limit
        )
    ]
    evaluation_paths = list_npy_inputs(args.evaluation_dir)[
        args.evaluation_offset : (
            None
            if args.evaluation_limit is None
            else args.evaluation_offset + args.evaluation_limit
        )
    ]
    overlap = {path.resolve() for path in calibration_paths}.intersection(
        path.resolve() for path in evaluation_paths
    )
    if overlap:
        raise ValueError(
            f"Calibration and evaluation selections overlap by {len(overlap)} files."
        )


if __name__ == "__main__":
    main()
