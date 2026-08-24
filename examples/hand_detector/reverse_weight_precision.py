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

"""Search FP-to-W8 parameter subsets that preserve W8/A16 error targets."""

from __future__ import annotations

import argparse
import json
import math

from pathlib import Path

import torch
from tico.quantization.analysis import make_output_adapter

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.reverse_weight_precision import (
    print_reverse_weight_precision,
    run_reverse_weight_precision_diagnostic,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    command: str = "reverse-weight-precision",
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        command,
        help=(
            "Start from P3 FP weights and greedily or with a beam search restore "
            "W8 groups while maintaining REG/CLS MAE targets."
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
        "--granularity",
        choices=("semantic", "site"),
        default="semantic",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help="Optional ordered subset of semantic or site weight groups.",
    )
    parser.add_argument(
        "--skip-greedy",
        action="store_true",
        help="Run only the independent one-group W8 costs and optional beam.",
    )
    parser.add_argument(
        "--greedy-selection-objective",
        choices=("primary-cost", "parameter-efficiency"),
        default="primary-cost",
        help=(
            "Choose the smallest REG increase or the smallest REG increase per "
            "newly quantized parameter."
        ),
    )
    parser.add_argument(
        "--max-greedy-steps",
        type=int,
        default=0,
        help="Maximum reverse-greedy transitions; zero allows every group.",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=0,
        help="Enable reverse beam search with this width; zero disables it.",
    )
    parser.add_argument(
        "--beam-exploration-slots",
        type=int,
        default=0,
        help=(
            "Beam slots reserved for high-W8-coverage states that may temporarily "
            "violate the final targets but remain below search ceilings."
        ),
    )
    parser.add_argument(
        "--beam-candidate-count",
        type=int,
        default=0,
        help=(
            "Use the lowest-cost independent groups for beam search; zero uses "
            "every selected group."
        ),
    )
    parser.add_argument(
        "--max-beam-steps",
        type=int,
        default=0,
        help="Maximum beam depth; zero allows every beam candidate group.",
    )
    parser.add_argument("--target-regressor-mae", type=float, default=0.1)
    parser.add_argument("--target-classifier-mae", type=float, default=0.1)
    parser.add_argument(
        "--search-regressor-ceiling",
        type=float,
        help="Temporary beam ceiling; defaults to the all-W8 P2 REG MAE.",
    )
    parser.add_argument(
        "--search-classifier-ceiling",
        type=float,
        help="Temporary beam ceiling; defaults to the all-W8 P2 CLS MAE.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Independent rows printed to the console; zero prints every group.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "reverse_weight_precision.json",
    )


def main() -> None:
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
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
    report = run_reverse_weight_precision_diagnostic(
        float_model,
        calibration,
        evaluation,
        uint8_percentile=args.uint8_percentile,
        int16_observer=args.int16_observer,
        int16_percentile=args.int16_percentile,
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        sampling_seed=args.sampling_seed,
        requested_groups=args.groups,
        granularity=args.granularity,
        run_greedy=not args.skip_greedy,
        greedy_selection_objective=args.greedy_selection_objective,
        max_greedy_steps=args.max_greedy_steps,
        beam_width=args.beam_width,
        beam_exploration_slots=args.beam_exploration_slots,
        beam_candidate_count=args.beam_candidate_count,
        max_beam_steps=args.max_beam_steps,
        target_regressor_mae=args.target_regressor_mae,
        target_classifier_mae=args.target_classifier_mae,
        search_regressor_ceiling=args.search_regressor_ceiling,
        search_classifier_ceiling=args.search_classifier_ceiling,
        output_adapter=OUTPUT_ADAPTER,
    )
    report["metadata"].update(
        {
            "calibration_samples": len(calibration),
            "evaluation_samples": len(evaluation),
            "device": str(device),
        }
    )
    print_reverse_weight_precision(report, top_k=args.top_k)
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
    for name in (
        "max_greedy_steps",
        "beam_width",
        "beam_exploration_slots",
        "beam_candidate_count",
        "max_beam_steps",
        "top_k",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative.")
    if args.beam_width == 0 and args.beam_exploration_slots != 0:
        raise ValueError(
            "--beam-exploration-slots must be zero when beam search is disabled."
        )
    if args.beam_width > 0 and args.beam_exploration_slots >= args.beam_width:
        raise ValueError("--beam-exploration-slots must be smaller than --beam-width.")
    for name in (
        "target_regressor_mae",
        "target_classifier_mae",
        "search_regressor_ceiling",
        "search_classifier_ceiling",
    ):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")


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
