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

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from tico.circle.document import CircleDocument
from tico.circle.errors import CircleError
from tico.circle.inspect import format_document, summarize_document
from tico.circle.operations import (
    extract_by_operator_indices,
    extract_by_tensor_patterns,
    SignaturePolicy,
)

from tico.circle.passes import (
    CanonicalizeEquivalentOpsPass,
    CircleOptimizationPreset,
    CirclePass,
    CirclePassContext,
    CirclePassManager,
    CirclePassStrategy,
    CommonSubexpressionEliminationPass,
    ConstantFoldingProfile,
    create_optimization_preset,
    EliminateIdentityOpsPass,
    EliminateTransposeBoundedLayoutRegionPass,
    FoldConstantsPass,
    FuseCompositeOpsPass,
    FuseLegacyFCGeluFCPass,
    FuseLinearOpsPass,
    FuseTransposeConvSlicePass,
    LegalizeDynamicFullyConnectedPass,
    O1LegacyCompatibilityOptions,
    O1LegalizationOptions,
    O1OptimizationOptions,
    O1PipelineOptions,
    ResolveLegacyCustomOpsPass,
    SimplifyArithmeticPass,
    SimplifyReductionOpsPass,
    SimplifyViewOpsPass,
)
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass
from tico.circle.selector import parse_operator_spec

LOGGER = logging.getLogger("tico.circle.cli")


_DEFAULT_PASSES = "dce,compact"
_PASS_REGISTRY: dict[str, type[CirclePass]] = {
    "canonicalize-equivalent-ops": CanonicalizeEquivalentOpsPass,
    "cse": CommonSubexpressionEliminationPass,
    "eliminate-identity-ops": EliminateIdentityOpsPass,
    "eliminate-transpose-bounded-layout-region": (
        EliminateTransposeBoundedLayoutRegionPass
    ),
    "fold-constants": FoldConstantsPass,
    "fuse-composite-ops": FuseCompositeOpsPass,
    "fuse-legacy-fc-gelu-fc": FuseLegacyFCGeluFCPass,
    "fuse-linear-ops": FuseLinearOpsPass,
    "fuse-transpose-conv-slice": FuseTransposeConvSlicePass,
    "legalize-dynamic-fully-connected": LegalizeDynamicFullyConnectedPass,
    "resolve-legacy-custom-ops": ResolveLegacyCustomOpsPass,
    "simplify-arithmetic": SimplifyArithmeticPass,
    "simplify-reduction-ops": SimplifyReductionOpsPass,
    "simplify-view-ops": SimplifyViewOpsPass,
    "dce": DeadCodeEliminationPass,
    "compact": CompactIndicesPass,
}
_DEPRECATED_PASS_NAMES: dict[str, str] = {
    "canonicalize-arithmetic": "simplify-arithmetic",
    "fold-constant-subgraph": "fold-constants",
    "fold-heavy-constant-subgraph": (
        "fold-constants with --constant-folding-profile heavy"
    ),
    "remove-no-op-operators": "eliminate-identity-ops",
}


def _build_parser() -> argparse.ArgumentParser:
    """Create the tico-circle command-line parser."""

    parser = argparse.ArgumentParser(
        prog="tico-circle",
        description="Inspect and transform exported Circle model artifacts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging on standard error.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Print model and graph structure."
    )
    inspect_parser.add_argument("input", help="Input .circle path or '-' for stdin.")
    inspect_parser.add_argument(
        "--subgraph", type=int, help="Limit detailed output to one subgraph."
    )
    inspect_parser.add_argument(
        "--tensors", action="store_true", help="Include tensor details."
    )
    inspect_parser.add_argument(
        "--operators", action="store_true", help="Include operator details."
    )
    inspect_parser.add_argument(
        "--json", action="store_true", help="Print the summary as JSON."
    )
    inspect_parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Check internal references and graph bookkeeping; "
            "print findings to stderr."
        ),
    )
    inspect_parser.set_defaults(handler=_inspect_command)

    verify_parser = subparsers.add_parser(
        "verify",
        help="Check internal Circle references and graph bookkeeping.",
        description=(
            "Check index ranges, graph producer/consumer bookkeeping, buffers, "
            "signatures, metadata, and subgraph references. This command does not "
            "run inference or validate numerical accuracy or backend compatibility."
        ),
    )
    verify_parser.add_argument("input", help="Input .circle path or '-' for stdin.")
    verify_parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Return a non-zero status when warnings are present.",
    )
    verify_parser.set_defaults(handler=_verify_command)

    extract_parser = subparsers.add_parser(
        "extract", help="Extract an operator region into a new Circle model."
    )
    extract_parser.add_argument("input", help="Input .circle path or '-' for stdin.")
    extract_parser.add_argument(
        "-o", "--output", required=True, help="Output .circle path or '-' for stdout."
    )
    extract_parser.add_argument(
        "--subgraph", type=int, default=0, help="Source subgraph index."
    )
    extract_parser.add_argument(
        "--ops",
        help="Comma-separated operator indices or inclusive ranges such as 0-10,15.",
    )
    extract_parser.add_argument(
        "--from-tensor",
        action="append",
        default=[],
        help="Regular expression for a source tensor. May be repeated.",
    )
    extract_parser.add_argument(
        "--to-tensor",
        action="append",
        default=[],
        help="Regular expression for a destination tensor. May be repeated.",
    )
    extract_parser.add_argument(
        "--full-match",
        action="store_true",
        help="Require tensor patterns to match complete names.",
    )
    extract_parser.add_argument(
        "--keep-other-subgraphs",
        action="store_true",
        help="Retain subgraphs other than the selected source subgraph.",
    )
    extract_parser.add_argument(
        "--preserve-compatible-signatures",
        action="store_true",
        help="Retain only signatures whose mappings exactly match the new graph I/O.",
    )
    extract_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip internal-consistency checks for the extracted model.",
    )
    extract_parser.set_defaults(handler=_extract_command)

    optimize_parser = subparsers.add_parser(
        "optimize", help="Run optimization and cleanup passes over a Circle model."
    )
    optimize_parser.add_argument("input", help="Input .circle path or '-' for stdin.")
    optimize_parser.add_argument(
        "-o", "--output", required=True, help="Output .circle path or '-' for stdout."
    )
    selection = optimize_parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--preset",
        choices=[preset.value for preset in CircleOptimizationPreset],
        help=(
            "Run a built-in optimization pipeline with fixed pass ordering and "
            "scheduling."
        ),
    )
    selection.add_argument(
        "--passes",
        help=(
            "Comma-separated passes. Available values: "
            f"{', '.join(_PASS_REGISTRY)}. Defaults to {_DEFAULT_PASSES!r} when "
            "no preset is selected."
        ),
    )
    optimize_parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in CirclePassStrategy],
        help=(
            "Pass scheduling strategy for --passes. Use 'until-no-change' for "
            "complete rounds or 'restart' for legacy restart behavior. "
            "Presets own their scheduling and cannot be combined with this option."
        ),
    )

    optimization_options = optimize_parser.add_argument_group(
        "O1 and constant-folding optimization options"
    )
    optimization_options.add_argument(
        "--constant-folding-profile",
        choices=[profile.value for profile in ConstantFoldingProfile],
        default=ConstantFoldingProfile.BASIC.value,
        help=(
            "Select the evaluator family for O1 or a standalone " "fold-constants pass."
        ),
    )
    optimization_options.add_argument(
        "--fuse-transpose-conv-slice",
        action="store_true",
        help="Enable the optional static TransposeConv-Slice fusion during O1.",
    )
    legalization_options = optimize_parser.add_argument_group("O1 legalization options")
    legalization_options.add_argument(
        "--legalize-dynamic-fully-connected",
        action="store_true",
        help="Lower dynamic-weight FLOAT32 FullyConnected operators during O1.",
    )
    compatibility_options = optimize_parser.add_argument_group(
        "O1 legacy compatibility options"
    )
    compatibility_options.add_argument(
        "--resolve-legacy-custom-ops",
        action="store_true",
        help="Recover selected former TensorFlow custom operators during O1.",
    )
    compatibility_options.add_argument(
        "--fuse-legacy-fc-gelu-fc",
        action="store_true",
        help="Recognize the legacy FC-Erf GELU pattern during O1.",
    )
    optimize_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip internal-consistency checks after passes and at completion.",
    )
    optimize_parser.set_defaults(handler=_optimize_command)
    return parser


def _inspect_command(args: argparse.Namespace) -> int:
    """Inspect one Circle document."""

    document = CircleDocument.load(args.input)
    if args.subgraph is not None:
        document.subgraph(args.subgraph)
    if args.verify:
        report = document.verify(raise_on_error=False)
        print(report.format(), file=sys.stderr)
        if not report.ok:
            return 1
    if args.json:
        print(
            json.dumps(summarize_document(document).to_dict(), indent=2, sort_keys=True)
        )
    else:
        print(
            format_document(
                document,
                subgraph_index=args.subgraph,
                include_tensors=args.tensors,
                include_operators=args.operators,
            )
        )
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    """Verify one Circle document."""

    document = CircleDocument.load(args.input)
    report = document.verify(raise_on_error=False)
    failed_on_warnings = args.warnings_as_errors and bool(report.warnings)
    stream = sys.stderr if not report.ok or failed_on_warnings else sys.stdout
    print(report.format(), file=stream)
    if not report.ok or failed_on_warnings:
        return 1
    return 0


def _extract_command(args: argparse.Namespace) -> int:
    """Extract one operator region from a Circle document."""

    document = CircleDocument.load(args.input)
    policy = (
        SignaturePolicy.PRESERVE_COMPATIBLE
        if args.preserve_compatible_signatures
        else SignaturePolicy.DROP
    )
    common: dict[str, Any] = {
        "subgraph_index": args.subgraph,
        "keep_other_subgraphs": args.keep_other_subgraphs,
        "signature_policy": policy,
        "verify": not args.no_verify,
    }
    if args.ops is not None and (args.from_tensor or args.to_tensor):
        raise ValueError("--ops cannot be combined with --from-tensor or --to-tensor.")
    if args.ops is not None:
        result = extract_by_operator_indices(
            document,
            parse_operator_spec(args.ops),
            **common,
        )
    else:
        if not args.from_tensor and not args.to_tensor:
            raise ValueError(
                "Provide --ops or at least one --from-tensor/--to-tensor pattern."
            )
        result = extract_by_tensor_patterns(
            document,
            from_patterns=args.from_tensor,
            to_patterns=args.to_tensor,
            full_match=args.full_match,
            **common,
        )

    result.document.save(args.output)
    print(
        "Extracted operators "
        f"{list(result.selected_operator_indices)} with inputs "
        f"{list(result.boundary.inputs)} and outputs {list(result.boundary.outputs)}.",
        file=sys.stderr,
    )
    return 0


def _parse_passes(
    value: str,
    *,
    constant_folding_profile: ConstantFoldingProfile
    | str = (ConstantFoldingProfile.BASIC),
) -> list[CirclePass]:
    """Create Circle pass instances from canonical comma-separated names."""

    profile = ConstantFoldingProfile(constant_folding_profile)
    passes: list[CirclePass] = []
    for raw_name in value.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        replacement = _DEPRECATED_PASS_NAMES.get(name)
        if replacement is not None:
            raise ValueError(f"Circle pass {name!r} was removed; use {replacement}.")
        try:
            pass_type = _PASS_REGISTRY[name]
        except KeyError as error:
            raise ValueError(
                f"Unknown Circle pass {name!r}; available passes are "
                f"{sorted(_PASS_REGISTRY)}."
            ) from error
        if name == "fold-constants":
            passes.append(FoldConstantsPass(profile=profile))
        else:
            passes.append(pass_type())
    if not passes:
        raise ValueError("At least one Circle pass must be selected.")
    return passes


def _optimize_command(args: argparse.Namespace) -> int:
    """Run a configured Circle pass pipeline and save its output."""

    if args.preset is not None and args.strategy is not None:
        raise ValueError(
            "--strategy cannot be combined with --preset because presets own "
            "their pass scheduling."
        )

    profile = ConstantFoldingProfile(
        getattr(
            args,
            "constant_folding_profile",
            ConstantFoldingProfile.BASIC.value,
        )
    )
    resolve_custom_ops = bool(getattr(args, "resolve_legacy_custom_ops", False))
    legalize_dynamic_fc = bool(getattr(args, "legalize_dynamic_fully_connected", False))
    fuse_transpose_conv_slice = bool(getattr(args, "fuse_transpose_conv_slice", False))
    fuse_legacy_fc_gelu_fc = bool(getattr(args, "fuse_legacy_fc_gelu_fc", False))
    o1_only_options = any(
        (
            resolve_custom_ops,
            legalize_dynamic_fc,
            fuse_transpose_conv_slice,
            fuse_legacy_fc_gelu_fc,
        )
    )
    if o1_only_options and args.preset is None:
        raise ValueError(
            "O1 legalization and compatibility flags require --preset o1; "
            "use --passes for standalone transformations."
        )

    selected_passes = args.passes or _DEFAULT_PASSES
    if profile is ConstantFoldingProfile.HEAVY and args.preset is None:
        pass_names = {
            name.strip().lower() for name in selected_passes.split(",") if name.strip()
        }
        if "fold-constants" not in pass_names:
            raise ValueError(
                "--constant-folding-profile heavy requires --preset o1 or "
                "a fold-constants pass."
            )

    custom_passes: list[CirclePass] | None = None
    if args.preset is None:
        custom_passes = _parse_passes(
            selected_passes,
            constant_folding_profile=profile,
        )

    document = CircleDocument.load(args.input)
    context = CirclePassContext(verify_after_each_pass=not args.no_verify)
    if args.preset is not None:
        options = O1PipelineOptions(
            optimization=O1OptimizationOptions(
                constant_folding_profile=profile,
                fuse_transpose_conv_slice=fuse_transpose_conv_slice,
            ),
            legalization=O1LegalizationOptions(
                dynamic_fully_connected=legalize_dynamic_fc,
            ),
            compatibility=O1LegacyCompatibilityOptions(
                resolve_custom_ops=resolve_custom_ops,
                fuse_fc_gelu_fc=fuse_legacy_fc_gelu_fc,
            ),
        )
        result = create_optimization_preset(
            args.preset,
            options=options,
        ).run(document, context)
    else:
        if custom_passes is None:
            raise AssertionError("Custom pass selection was not initialized.")
        strategy = CirclePassStrategy(args.strategy or CirclePassStrategy.ONCE.value)
        result = CirclePassManager(
            custom_passes,
            strategy=strategy,
        ).run(document, context)
    if not args.no_verify:
        document.verify(raise_on_error=True)
    document.save(args.output)
    print(
        f"Circle optimization completed with {result.changes} reported changes.",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the tico-circle command-line interface."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    try:
        return int(args.handler(args))
    except (CircleError, IndexError, TypeError, ValueError) as error:
        LOGGER.debug("Circle command failed", exc_info=True)
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
