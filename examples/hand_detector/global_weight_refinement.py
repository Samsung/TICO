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

"""Run global end-to-end scale and AdaRound refinement from a joint checkpoint."""

from __future__ import annotations

import argparse
import json
import math

from pathlib import Path

import torch
from tico.quantization.algorithm.adaround import (
    GlobalWeightRefinementConfig,
    JointAdaRoundObjective,
)
from tico.quantization.analysis import make_output_adapter

from examples.hand_detector._support.analysis import OUTPUT_NAMES
from examples.hand_detector._support.data import (
    list_npy_inputs,
    load_npy_inputs,
    make_synthetic_inputs,
)
from examples.hand_detector._support.global_weight_refinement import (
    run_hand_detector_global_weight_refinement,
)
from examples.hand_detector._support.joint_adaround import (
    apply_joint_adaround_checkpoint,
    save_joint_adaround_checkpoint,
)
from examples.hand_detector._support.multistart_reconstruction import (
    split_reconstruction_samples_three_way,
)
from examples.hand_detector._support.weight_precision_sensitivity import (
    build_w8a16_candidate,
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
    command: str = "global-weight-refinement",
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        command,
        help=(
            "Jointly refine all selected Conv W8 scales and rounding from a "
            "saved joint-AdaRound checkpoint using final teacher outputs."
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
        "--load-checkpoint",
        type=Path,
        required=True,
        help=(
            "Joint DW/PW checkpoint providing the entry hard codes, learned "
            "weight scales, and all activation qparams."
        ),
    )
    parser.add_argument(
        "--save-checkpoint",
        type=Path,
        help="Save the accepted global hard weights and exact affine qparams.",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        help=(
            "Optional semantic group subset. By default every regular, "
            "depthwise, regressor-head, and classifier-head Conv is refined."
        ),
    )
    parser.add_argument("--require-accept", action="store_true")

    parser.add_argument("--selection-count", type=int, default=40)
    parser.add_argument("--acceptance-count", type=int, default=40)
    parser.add_argument("--selection-seed", type=int, default=20260803)
    parser.add_argument(
        "--selection-score-metric",
        choices=("mae", "mse", "rmse", "relative_mae"),
        default="mae",
    )
    parser.add_argument("--classifier-limit", type=float, default=0.1)
    parser.add_argument(
        "--relative-classifier-tolerance",
        type=float,
        help="Optional classifier tolerance relative to the loaded entry state.",
    )
    parser.add_argument("--minimum-selection-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-acceptance-improvement", type=float, default=1e-3)

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--alpha-learning-rate", type=float, default=1e-5)
    parser.add_argument("--scale-learning-rate", type=float, default=3e-6)
    parser.add_argument("--classifier-loss-weight", type=float, default=0.25)
    parser.add_argument("--rounding-loss-weight", type=float, default=1e-3)
    parser.add_argument("--scale-loss-weight", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.2)
    parser.add_argument("--beta-start", type=float, default=20.0)
    parser.add_argument("--beta-end", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=-0.1)
    parser.add_argument("--zeta", type=float, default=1.1)
    parser.add_argument("--max-scale-ratio", type=float, default=1.25)
    parser.add_argument(
        "--checkpoint-alpha-initialization",
        choices=("preserve", "constant"),
        default="preserve",
        help=(
            "Preserve the fractional AdaRound magnitude or reset every "
            "checkpoint decision to one exact constant magnitude."
        ),
    )
    parser.add_argument(
        "--checkpoint-alpha-initial-magnitude",
        type=float,
        default=0.05,
        help="Exact |alpha| used by constant checkpoint initialization.",
    )
    parser.add_argument(
        "--checkpoint-alpha-minimum-magnitude",
        type=float,
        default=1.5,
        help="Minimum initial |alpha| for checkpoint floor/ceil decisions.",
    )
    parser.add_argument(
        "--freeze-scale",
        action="store_true",
        help="Remove scale tensors from autograd and the optimizer.",
    )
    parser.add_argument(
        "--training-loss",
        choices=("raw-mae", "normalized-l1"),
        default="raw-mae",
        help="Teacher loss used by the optimization forward.",
    )
    parser.add_argument(
        "--checkpoint-flip-budget",
        type=int,
        help=(
            "Maximum total checkpoint floor/ceil decisions allowed to flip; "
            "omit for no projection and use zero to freeze all decisions."
        ),
    )
    parser.add_argument(
        "--checkpoint-flip-projection-margin",
        type=float,
        default=1e-4,
        help="Signed alpha margin used when projecting excess flips back.",
    )
    parser.add_argument(
        "--checkpoint-anchor-loss-weight",
        type=float,
        default=1e-2,
        help="Initial weight of the checkpoint decision-margin anchor.",
    )
    parser.add_argument(
        "--checkpoint-anchor-margin",
        type=float,
        default=0.5,
        help="Required signed-alpha margin around checkpoint decisions.",
    )
    parser.add_argument(
        "--checkpoint-anchor-fraction",
        type=float,
        default=0.2,
        help="Fraction of optimizer steps over which the anchor decays to zero.",
    )
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--initialization-metric-tolerance",
        type=float,
        default=1e-4,
        help="Absolute MAE tolerance for checkpoint initialization parity.",
    )
    parser.add_argument(
        "--initialization-metric-relative-tolerance",
        type=float,
        default=1e-3,
        help="Relative MAE tolerance for checkpoint initialization parity.",
    )
    parser.add_argument("--optimization-seed", type=int, default=20260831)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--report-json",
        type=Path,
        default=DIRECTORY / "reports" / "global_weight_refinement.json",
    )


def main() -> None:
    run(build_parser().parse_args())


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    _validate_disjoint(args)
    if not args.load_checkpoint.is_file():
        raise FileNotFoundError(
            f"Global refinement checkpoint does not exist: " f"{args.load_checkpoint}"
        )
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
    data_split = split_reconstruction_samples_three_way(
        calibration,
        args.selection_count,
        args.acceptance_count,
        seed=args.selection_seed,
    )
    candidate, p2_metadata = build_w8a16_candidate(
        float_model,
        calibration,
        uint8_percentile=args.uint8_percentile,
        int16_observer=args.int16_observer,
        int16_percentile=args.int16_percentile,
        max_samples=args.max_samples,
        samples_per_batch=args.samples_per_batch,
        sampling_seed=args.sampling_seed,
    )
    checkpoint_replay = apply_joint_adaround_checkpoint(
        candidate,
        args.load_checkpoint,
    )
    config = GlobalWeightRefinementConfig(
        steps=args.steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_interval=args.evaluation_interval,
        alpha_learning_rate=args.alpha_learning_rate,
        scale_learning_rate=args.scale_learning_rate,
        primary_output="regressors",
        auxiliary_output="classifiers",
        auxiliary_loss_weight=args.classifier_loss_weight,
        rounding_loss_weight=args.rounding_loss_weight,
        scale_loss_weight=args.scale_loss_weight,
        warmup_fraction=args.warmup_fraction,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        gamma=args.gamma,
        zeta=args.zeta,
        max_scale_ratio=args.max_scale_ratio,
        checkpoint_alpha_minimum_magnitude=(args.checkpoint_alpha_minimum_magnitude),
        checkpoint_alpha_initialization=(args.checkpoint_alpha_initialization),
        checkpoint_alpha_initial_magnitude=(args.checkpoint_alpha_initial_magnitude),
        freeze_scale=args.freeze_scale,
        training_loss=args.training_loss.replace("-", "_"),
        checkpoint_flip_budget=args.checkpoint_flip_budget,
        checkpoint_flip_projection_margin=(args.checkpoint_flip_projection_margin),
        checkpoint_anchor_loss_weight=args.checkpoint_anchor_loss_weight,
        checkpoint_anchor_margin=args.checkpoint_anchor_margin,
        checkpoint_anchor_fraction=args.checkpoint_anchor_fraction,
        gradient_clip_norm=args.gradient_clip_norm,
        initialization_metric_tolerance=(args.initialization_metric_tolerance),
        initialization_metric_relative_tolerance=(
            args.initialization_metric_relative_tolerance
        ),
        seed=args.optimization_seed,
    )
    relative = (
        {"classifiers": args.relative_classifier_tolerance}
        if args.relative_classifier_tolerance is not None
        else {}
    )
    absolute = {"classifiers": args.classifier_limit}
    selection_objective = JointAdaRoundObjective(
        primary_output="regressors",
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_selection_improvement,
        absolute_output_limits=absolute,
        relative_output_tolerances=relative,
    )
    acceptance_objective = JointAdaRoundObjective(
        primary_output="regressors",
        primary_metric=args.selection_score_metric,
        minimum_improvement=args.minimum_acceptance_improvement,
        absolute_output_limits=absolute,
        relative_output_tolerances=relative,
    )

    def progress(checkpoint) -> None:
        outputs = checkpoint.selection_outputs
        hard = checkpoint.hard_state_statistics
        diagnostics = ""
        if hard is not None:
            diagnostics = (
                f" code_chg={100.0 * hard.changed_from_checkpoint_ratio:6.3f}%"
                f" sign_flip={100.0 * hard.checkpoint_sign_flip_ratio:6.3f}%"
                f" scale=[{hard.scale_ratio_minimum:.5f},"
                f"{hard.scale_ratio_maximum:.5f}]"
                f" clipped={hard.clipped_code_count}"
            )
        alpha = checkpoint.alpha_absolute_histogram
        if alpha is not None and alpha.percentile_50 is not None:
            diagnostics += (
                f" alpha_abs[p50={alpha.percentile_50:.2e},"
                f"p90={alpha.percentile_90:.2e},"
                f"p99={alpha.percentile_99:.2e}]"
            )
        gradient = checkpoint.alpha_gradient_absolute_histogram
        if gradient is not None and gradient.percentile_50 is not None:
            diagnostics += (
                f" grad_abs[p50={gradient.percentile_50:.2e},"
                f"p90={gradient.percentile_90:.2e},"
                f"p99={gradient.percentile_99:.2e},"
                f"zero={100.0 * gradient.zero_ratio:.1f}%]"
            )
        budget = checkpoint.flip_budget_statistics
        if budget is not None and budget.budget is not None:
            diagnostics += (
                f" flip_budget={budget.after_count}/{budget.budget}"
                f" projected={budget.projected_count}"
            )
        print(
            f"checkpoint step={checkpoint.step:4d} "
            f"REG_MAE={float(outputs['regressors']['mae']):.6e} "
            f"CLS_MAE={float(outputs['classifiers']['mae']):.6e} "
            f"best={'yes' if checkpoint.selected_as_best else 'no'}"
            f"{diagnostics}"
        )

    report = run_hand_detector_global_weight_refinement(
        float_model,
        candidate,
        data_split=data_split,
        evaluation_samples=evaluation,
        config=config,
        selection_objective=selection_objective,
        acceptance_objective=acceptance_objective,
        output_adapter=OUTPUT_ADAPTER,
        requested_groups=args.groups,
        progress_callback=progress,
        device=device,
    )
    refinement = report["global_refinement"]
    accepted = bool(refinement["accepted"])
    payload = {
        "analysis": "global_end_to_end_weight_refinement",
        "metadata": {
            **p2_metadata,
            "calibration_samples": len(calibration),
            "evaluation_samples": len(evaluation),
            "selection_count": args.selection_count,
            "acceptance_count": args.acceptance_count,
            "selection_seed": args.selection_seed,
            "selection_score_metric": args.selection_score_metric,
            "classifier_limit": args.classifier_limit,
            "relative_classifier_tolerance": (args.relative_classifier_tolerance),
            "minimum_selection_improvement": (args.minimum_selection_improvement),
            "minimum_acceptance_improvement": (args.minimum_acceptance_improvement),
            "steps": args.steps,
            "gradient_accumulation_steps": (args.gradient_accumulation_steps),
            "evaluation_interval": args.evaluation_interval,
            "alpha_learning_rate": args.alpha_learning_rate,
            "scale_learning_rate": args.scale_learning_rate,
            "classifier_loss_weight": args.classifier_loss_weight,
            "rounding_loss_weight": args.rounding_loss_weight,
            "scale_loss_weight": args.scale_loss_weight,
            "warmup_fraction": args.warmup_fraction,
            "beta_start": args.beta_start,
            "beta_end": args.beta_end,
            "gamma": args.gamma,
            "zeta": args.zeta,
            "max_scale_ratio": args.max_scale_ratio,
            "checkpoint_alpha_initialization": (args.checkpoint_alpha_initialization),
            "checkpoint_alpha_initial_magnitude": (
                args.checkpoint_alpha_initial_magnitude
            ),
            "checkpoint_alpha_minimum_magnitude": (
                args.checkpoint_alpha_minimum_magnitude
            ),
            "freeze_scale": args.freeze_scale,
            "training_loss": args.training_loss,
            "checkpoint_flip_budget": args.checkpoint_flip_budget,
            "checkpoint_flip_projection_margin": (
                args.checkpoint_flip_projection_margin
            ),
            "checkpoint_anchor_loss_weight": (args.checkpoint_anchor_loss_weight),
            "checkpoint_anchor_margin": args.checkpoint_anchor_margin,
            "checkpoint_anchor_fraction": args.checkpoint_anchor_fraction,
            "initialization_metric_tolerance": (args.initialization_metric_tolerance),
            "initialization_metric_relative_tolerance": (
                args.initialization_metric_relative_tolerance
            ),
            "optimization_seed": args.optimization_seed,
            "device": str(device),
            "requested_groups": args.groups,
            "checkpoint_replay": checkpoint_replay,
        },
        **report,
    }
    checkpoint = None
    if args.save_checkpoint is not None and (accepted or not args.require_accept):
        checkpoint_path = save_joint_adaround_checkpoint(
            candidate,
            args.save_checkpoint,
            metadata={
                "source_report": str(args.report_json),
                "source_checkpoint": str(args.load_checkpoint),
                "analysis": payload["analysis"],
                "accepted": accepted,
                "best_step": refinement["best_step"],
                "final_evaluation": payload["final_evaluation"],
                "global_weight_groups": refinement["weight_groups"],
            },
        )
        checkpoint = {"path": checkpoint_path}
        payload["checkpoint"] = checkpoint
    _print_report(payload)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"\nWrote {args.report_json}")
    if checkpoint is not None:
        print(f"Wrote {checkpoint['path']}")
    if args.require_accept and not accepted:
        raise RuntimeError(
            "Global refinement was not accepted; the report was written, "
            "but no deployment checkpoint was produced."
        )


def _print_report(report: dict[str, object]) -> None:
    baseline = report["baseline_evaluation"]
    scale_only = report["scale_only_evaluation"]
    selected = report["selected_evaluation"]
    final = report["final_evaluation"]
    refinement = report["global_refinement"]
    assert isinstance(baseline, dict)
    assert isinstance(scale_only, dict)
    assert isinstance(selected, dict)
    assert isinstance(final, dict)
    assert isinstance(refinement, dict)
    print("\nGlobal end-to-end Conv weight refinement")
    print(
        "Loaded entry:       "
        f"REG_MAE={_mae(baseline, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(baseline, 'classifiers'):.6e}"
    )
    print(
        "Learned scale RTN:  "
        f"REG_MAE={_mae(scale_only, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(scale_only, 'classifiers'):.6e}"
    )
    print(
        "Scale + AdaRound:   "
        f"REG_MAE={_mae(selected, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(selected, 'classifiers'):.6e}"
    )
    print(
        "Final committed:    "
        f"REG_MAE={_mae(final, 'regressors'):.6e}, "
        f"CLS_MAE={_mae(final, 'classifiers'):.6e}"
    )
    print(
        f"result={'ACCEPT' if refinement['accepted'] else 'ROLLBACK'}, "
        f"best_step={int(refinement['best_step'])}, "
        f"weights={int(refinement['weight_group_count'])}"
    )
    print("acceptance: " + str(refinement["acceptance_reason"]))


def _mae(outputs: dict[str, object], name: str) -> float:
    value = outputs[name]
    assert isinstance(value, dict)
    return float(value["mae"])


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
    if args.selection_count <= 0 or args.acceptance_count <= 0:
        raise ValueError("selection-count and acceptance-count must be positive.")
    if args.selection_count + args.acceptance_count >= (
        args.calibration_limit or args.synthetic_calibration_samples
    ):
        raise ValueError(
            "selection-count + acceptance-count must be smaller than the "
            "calibration sample count."
        )
    for name in (
        "classifier_limit",
        "alpha_learning_rate",
        "checkpoint_alpha_minimum_magnitude",
        "checkpoint_alpha_initial_magnitude",
        "checkpoint_flip_projection_margin",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    scale_learning_rate = float(args.scale_learning_rate)
    if not math.isfinite(scale_learning_rate) or (
        scale_learning_rate < 0.0 if args.freeze_scale else scale_learning_rate <= 0.0
    ):
        requirement = "nonnegative" if args.freeze_scale else "positive"
        raise ValueError(f"--scale-learning-rate must be finite and {requirement}.")
    if args.checkpoint_flip_budget is not None and (args.checkpoint_flip_budget < 0):
        raise ValueError("--checkpoint-flip-budget must be nonnegative.")
    for name in (
        "classifier_loss_weight",
        "rounding_loss_weight",
        "scale_loss_weight",
        "checkpoint_anchor_loss_weight",
        "checkpoint_anchor_margin",
        "minimum_selection_improvement",
        "minimum_acceptance_improvement",
        "initialization_metric_tolerance",
        "initialization_metric_relative_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative.")
    if args.relative_classifier_tolerance is not None and (
        not math.isfinite(args.relative_classifier_tolerance)
        or args.relative_classifier_tolerance < 0.0
    ):
        raise ValueError("--relative-classifier-tolerance must be nonnegative.")
    if not 0.0 <= args.checkpoint_anchor_fraction <= 1.0:
        raise ValueError("--checkpoint-anchor-fraction must be in [0, 1].")
    if args.save_checkpoint is not None and args.report_json == args.save_checkpoint:
        raise ValueError("Report and checkpoint paths must differ.")


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
