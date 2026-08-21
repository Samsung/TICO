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

"""Measure P0-P4 W8/A8/A16 numerical floors for the hand detector."""

from __future__ import annotations

import argparse
import json

from pathlib import Path

import torch

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.precision_ablation import (
    print_precision_floor_report,
    run_precision_floor_ablation,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector
from tico.quantization.analysis import make_output_adapter


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    command: str = "precision-ablation",
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        command,
        help="Measure output, activation, and W8 numerical floors using P0-P4.",
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
        help=(
            "MinMax is the default floor estimate because INT16 has enough "
            "resolution to retain the observed range without tail clipping."
        ),
    )
    parser.add_argument("--int16-percentile", type=float, default=99.99)
    parser.add_argument("--max-samples", type=int, default=524_288)
    parser.add_argument("--samples-per-batch", type=int, default=4_096)
    parser.add_argument("--sampling-seed", type=int, default=20260803)
    parser.add_argument("--target-regressor-mae", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "precision_ablation_phase1.json",
    )


def main() -> None:
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
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
    report = run_precision_floor_ablation(
        float_model,
        calibration,
        evaluation,
        uint8_percentile=args.uint8_percentile,
        int16_observer=args.int16_observer,
        int16_percentile=args.int16_percentile,
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        sampling_seed=args.sampling_seed,
        target_regressor_mae=args.target_regressor_mae,
        output_adapter=OUTPUT_ADAPTER,
    )
    report["metadata"].update(
        {
            "calibration_samples": len(calibration),
            "evaluation_samples": len(evaluation),
            "device": str(device),
        }
    )
    print_precision_floor_report(report)
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
