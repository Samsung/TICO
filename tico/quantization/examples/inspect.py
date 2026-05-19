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

from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.debug.static_llama_runtime import (
    run_static_llama_runtime,
    StaticLlamaRuntimeConfig,
)
from tico.quantization.recipes.debug.tied_embedding import (
    run_tied_embedding_smoke,
    TiedEmbeddingSmokeConfig,
)
from tico.quantization.recipes.debug.trace import trace_ptq_parity
from tico.quantization.recipes.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect/debug quantization recipes.")
    parser.add_argument("--config", required=True, help="Recipe config.")
    parser.add_argument(
        "--mode",
        choices=["trace", "static-llama-runtime", "tied-embedding-smoke"],
        default="trace",
    )
    parser.add_argument("--model", default=None, help="Override model.name_or_path.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    parser.add_argument("--enable-quantization", action="store_true")
    parser.add_argument("--interesting-modules", nargs="*", default=[])
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if args.model:
        overrides.append(f"model.name_or_path={args.model}")
    if args.device:
        overrides.append(f"runtime.device={args.device}")

    cfg = load_recipe_config(args.config, overrides=overrides)
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
