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

import argparse
from pathlib import Path

from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.static_gemma4_runtime import (
    run_static_gemma4_runtime,
    StaticGemma4RuntimeConfig,
)
from tico.quantization.recipes.debug.static_llama_runtime import (
    run_static_llama_runtime,
    StaticLlamaRuntimeConfig,
)
from tico.quantization.recipes.debug.tied_embedding import (
    run_tied_embedding_smoke,
    TiedEmbeddingSmokeConfig,
)
from tico.quantization.recipes.debug.trace import trace_ptq_parity
from tico.quantization.recipes.debug.wrapper_smoke import (
    list_cases,
    run_wrapper_smoke,
    run_wrapper_smoke_suite,
)
from tico.quantization.recipes.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect/debug quantization recipes.")
    parser.add_argument("--config", default=None, help="Recipe config.")
    parser.add_argument(
        "--mode",
        choices=[
            "trace",
            "static-llama-runtime",
            "static-gemma4-runtime",
            "tied-embedding-smoke",
            "wrapper-smoke",
        ],
        default="trace",
    )
    parser.add_argument("--model", default=None, help="Override model.name_or_path.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    parser.add_argument("--enable-quantization", action="store_true")
    parser.add_argument("--interesting-modules", nargs="*", default=[])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")

    parser.add_argument(
        "--case",
        default=None,
        help="Wrapper smoke case name. Use 'all' to run every registered case.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="List registered wrapper smoke cases and exit.",
    )
    parser.add_argument(
        "--export",
        choices=["none", "circle"],
        default="none",
        help="Optional wrapper-smoke export artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for wrapper-smoke plots, reports, and Circle files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with a non-zero exit when wrapper-smoke checks fail.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not print or save plot_two_outputs output for wrapper-smoke.",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="Optional JSON report path for a single wrapper-smoke case.",
    )
    parser.add_argument(
        "--calibration-iters",
        type=int,
        default=None,
        help="Limit wrapper-smoke calibration samples without editing YAML.",
    )
    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> list[str]:
    """Translate convenience CLI arguments into recipe config overrides."""
    overrides = list(args.set)
    if args.model:
        overrides.append(f"model.name_or_path={args.model}")
    if args.device:
        overrides.append(f"runtime.device={args.device}")
    if args.output_dir:
        overrides.append(f"debug.wrapper_smoke.output_dir={args.output_dir}")
    return overrides


def _load_cfg(args: argparse.Namespace, *, require_config: bool) -> dict:
    """Load a recipe config or return an empty wrapper-smoke config."""
    if args.config is None:
        if require_config:
            raise SystemExit("--config is required for this inspect mode")
        return {}
    return load_recipe_config(args.config, overrides=_build_overrides(args))


def _wrapper_smoke_case_group(case_name: str) -> str:
    """Return the display group for a wrapper smoke case name."""
    if case_name.startswith("nn_"):
        return "nn"
    if case_name.startswith("llama_"):
        return "llama"
    if case_name.startswith("qwen3_vl_"):
        return "qwen3_vl"
    if case_name.startswith("gemma4_"):
        return "gemma4"
    return "other"


def _print_wrapper_smoke_cases() -> None:
    """Print registered wrapper smoke case names grouped by family."""
    groups: dict[str, list[str]] = {
        "nn": [],
        "llama": [],
        "qwen3_vl": [],
        "gemma4": [],
        "other": [],
    }
    for case in list_cases():
        groups[_wrapper_smoke_case_group(case.name)].append(case.name)

    print("Available wrapper smoke cases:")
    for group_name in ("nn", "llama", "qwen3_vl", "gemma4", "other"):
        case_names = groups[group_name]
        if not case_names:
            continue
        print()
        for case_name in case_names:
            print(f"  {case_name}")


def _run_wrapper_smoke_mode(args: argparse.Namespace) -> None:
    """Dispatch wrapper-smoke mode."""
    if args.list_cases:
        _print_wrapper_smoke_cases()
        return
    if not args.case:
        raise SystemExit(
            "--case is required for --mode wrapper-smoke unless --list-cases is used"
        )

    cfg = _load_cfg(args, require_config=False)
    set_seed(cfg.get("runtime", {}).get("seed", 42))
    if args.case == "all":
        run_wrapper_smoke_suite(
            cfg=cfg,
            export=args.export,
            output_dir=args.output_dir,
            strict=args.strict,
            emit_plot=not args.no_plot,
            calibration_limit=args.calibration_iters,
        )
        return

    run_wrapper_smoke(
        args.case,
        cfg=cfg,
        export=args.export,
        output_dir=args.output_dir,
        strict=args.strict,
        emit_plot=not args.no_plot,
        report_json=Path(args.report_json) if args.report_json else None,
        calibration_limit=args.calibration_iters,
    )


def main() -> None:
    """Run the selected inspect/debug mode."""
    args = parse_args()

    if args.mode == "wrapper-smoke":
        _run_wrapper_smoke_mode(args)
        return

    cfg = _load_cfg(args, require_config=True)
    set_seed(cfg.get("runtime", {}).get("seed", 42))

    if args.mode == "tied-embedding-smoke":
        smoke_cfg = TiedEmbeddingSmokeConfig(
            **cfg.get("debug", {}).get("tied_embedding", {})
        )
        run_tied_embedding_smoke(smoke_cfg)
        return

    if args.mode == "static-llama-runtime":
        runtime_cfg = StaticLlamaRuntimeConfig(
            **cfg.get("debug", {}).get("static_llama_runtime", {})
        )
        run_static_llama_runtime(runtime_cfg)
        return

    if args.mode == "static-gemma4-runtime":
        gemma4_runtime_cfg = StaticGemma4RuntimeConfig(
            **cfg.get("debug", {}).get("static_gemma4_runtime", {})
        )
        run_static_gemma4_runtime(gemma4_runtime_cfg)
        return

    adapter = get_adapter(cfg["model"]["family"])
    ctx = RecipeContext(cfg=cfg, adapter=adapter)
    ctx = adapter.load_model(ctx)
    ctx.calibration_inputs = adapter.build_calibration_inputs(ctx)

    if args.mode == "trace":
        trace_ptq_parity(
            ctx,
            enable_quantization=args.enable_quantization,
            interesting_modules=args.interesting_modules,
        )


if __name__ == "__main__":
    main()
