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

"""Search legal UINT8/INT16 precision regions for the hand detector."""

from __future__ import annotations

import argparse
import json

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from tico.quantization.analysis import make_output_adapter

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.legal_mixed_precision import (
    Precision,
    PrecisionCostWeights,
    print_legal_mixed_precision_report,
    run_legal_mixed_precision_search,
)
from examples.hand_detector.hand_detector import load_nhwc_hand_detector


DIRECTORY = Path(__file__).resolve().parent
OUTPUT_ADAPTER = make_output_adapter(OUTPUT_NAMES)


def build_parser() -> argparse.ArgumentParser:
    """Return the standalone command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    return parser


def add_subparser(
    subparsers: argparse._SubParsersAction,
    *,
    command: str = "legal-mixed-precision",
) -> argparse.ArgumentParser:
    """Register this workflow under the hand-detector analysis CLI."""
    parser = subparsers.add_parser(
        command,
        help=(
            "Measure legal W8A8/W16A16 floors and search semantic UINT8 "
            "demotions from an all-INT16 entry."
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
        "--regressor-output-precision",
        choices=tuple(value.value for value in Precision),
        default=Precision.INT16.value,
    )
    parser.add_argument(
        "--classifier-output-precision",
        choices=tuple(value.value for value in Precision),
        default=Precision.UINT8.value,
    )
    parser.add_argument("--target-regressor-mae", type=float, default=0.1)
    parser.add_argument("--target-classifier-mae", type=float, default=0.1)

    parser.add_argument(
        "--granularity",
        choices=("semantic",),
        default="semantic",
        help="The first implementation searches semantic precision regions.",
    )
    parser.add_argument(
        "--search",
        choices=("none", "reverse-greedy", "reverse-beam"),
        default="reverse-beam",
    )
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=0,
        help="Limit sensitivity-ranked regions; zero keeps every region.",
    )
    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=0,
        help="Limit demotion depth; zero searches until no transition remains.",
    )
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument(
        "--search-even-if-entry-infeasible",
        action="store_true",
        help=(
            "Permit reverse search even when the all-INT16 entry misses the "
            "requested targets."
        ),
    )

    parser.add_argument("--parameter-cost-weight", type=float, default=1.0)
    parser.add_argument("--activation-cost-weight", type=float, default=1.0)
    parser.add_argument("--boundary-cost-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "legal_mixed_precision.json",
    )
    parser.add_argument(
        "--assignment-json",
        type=Path,
        default=(DIRECTORY / "reports" / "legal_mixed_precision_assignment.json"),
        help="Write the selected precision map as a compact follow-up artifact.",
    )


def main() -> None:
    """Run the standalone command."""
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
    """Load data, execute the legal search, and write report artifacts."""
    if args.granularity != "semantic":
        raise ValueError("Only semantic precision granularity is supported.")
    _validate_disjoint(args)
    device = torch.device(args.device)
    float_model = (
        load_nhwc_hand_detector(
            args.weights,
            args.spec,
            map_location=device,
        )
        .to(device)
        .eval()
    )
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

    report = run_legal_mixed_precision_search(
        float_model,
        calibration,
        evaluation,
        uint8_percentile=args.uint8_percentile,
        int16_observer=args.int16_observer,
        int16_percentile=args.int16_percentile,
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        sampling_seed=args.sampling_seed,
        regressor_output_precision=Precision(args.regressor_output_precision),
        classifier_output_precision=Precision(args.classifier_output_precision),
        target_regressor_mae=args.target_regressor_mae,
        target_classifier_mae=args.target_classifier_mae,
        search=args.search,
        beam_width=args.beam_width,
        candidate_count=args.candidate_count,
        max_search_steps=args.max_search_steps,
        skip_sensitivity=args.skip_sensitivity,
        search_even_if_entry_infeasible=(args.search_even_if_entry_infeasible),
        cost_weights=PrecisionCostWeights(
            parameter=args.parameter_cost_weight,
            activation=args.activation_cost_weight,
            boundary=args.boundary_cost_weight,
        ),
        output_adapter=OUTPUT_ADAPTER,
        progress_callback=_print_progress,
    )
    report["metadata"].update(
        {
            "calibration_samples": len(calibration),
            "evaluation_samples": len(evaluation),
            "device": str(device),
        }
    )
    print_legal_mixed_precision_report(report)
    _write_json(args.report_json, report)
    _write_json(
        args.assignment_json,
        _assignment_artifact(report, args.report_json),
    )
    print(f"\nWrote {args.report_json}")
    print(f"Wrote {args.assignment_json}")


def _print_progress(event: Mapping[str, Any]) -> None:
    if event.get("event") != "assignment_finish":
        return
    print(
        f"[{int(event['index']):4d}] "
        f"U8_REGIONS={int(event['uint8_region_count']):2d} "
        f"REG={float(event['regressor_mae']):.6e} "
        f"CLS={float(event['classifier_mae']):.6e} "
        f"TARGET={str(bool(event['meets_targets'])):5s} "
        f"QBOUND={int(event['dtype_transition_count']):2d} "
        f"COST={float(event['normalized_cost']):.6f}"
    )


def _assignment_artifact(
    report: Mapping[str, Any],
    report_path: Path,
) -> dict[str, Any]:
    selected = report["selected_assignment"]
    if not isinstance(selected, Mapping):
        raise TypeError("Selected precision assignment must be a mapping.")
    return {
        "format": "tico_hand_detector_legal_mixed_precision_v1",
        "source_report": str(report_path),
        "precision_map": dict(selected["precision_map"]),
        "uint8_regions": list(selected["uint8_regions"]),
        "int16_regions": list(selected["int16_regions"]),
        "outputs": {
            name: dict(metrics) for name, metrics in selected["outputs"].items()
        },
        "cost": dict(selected["cost"]),
        "contract": dict(selected["contract"]),
        "meets_targets": bool(selected["meets_targets"]),
        "search_metadata": {
            "target_regressor_mae": report["metadata"]["target_regressor_mae"],
            "target_classifier_mae": report["metadata"]["target_classifier_mae"],
            "regressor_output_precision": report["metadata"][
                "regressor_output_precision"
            ],
            "classifier_output_precision": report["metadata"][
                "classifier_output_precision"
            ],
        },
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False),
        encoding="utf-8",
    )


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
