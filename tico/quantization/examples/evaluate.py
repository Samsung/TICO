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
from collections.abc import MutableMapping

import torch

from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.evaluation.selection import (
    parse_evaluation_targets,
    validate_adapter_evaluation_config,
)
from tico.quantization.recipes.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an FP or fake-quant checkpoint."
    )
    parser.add_argument("--config", required=True, help="Base recipe config.")
    parser.add_argument(
        "--checkpoint", default=None, help="Optional torch checkpoint to evaluate."
    )
    parser.add_argument("--model", default=None, help="Override model.name_or_path.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    parser.add_argument(
        "--tasks",
        default=None,
        help=(
            "Comma-separated top-level evaluation targets. When set, only "
            "these targets run; benchmark details remain in the config."
        ),
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed evaluation sample logs.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm/progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    overrides.append("evaluation.enabled=true")
    if args.model:
        overrides.append(f"model.name_or_path={args.model}")
    if args.device:
        overrides.append(f"runtime.device={args.device}")
    if args.verbose:
        overrides.append("evaluation.verbose=true")
    if args.no_progress:
        overrides.append("runtime.show_progress=false")

    cfg = load_recipe_config(args.config, overrides=overrides)
    adapter = get_adapter(cfg["model"]["family"])

    if args.tasks is not None:
        eval_cfg = cfg.setdefault("evaluation", {})
        if not isinstance(eval_cfg, MutableMapping):
            raise TypeError("evaluation must be a mutable mapping.")
        eval_cfg["selected_tasks"] = parse_evaluation_targets(args.tasks)

    validate_adapter_evaluation_config(adapter, cfg)
    set_seed(cfg.get("runtime", {}).get("seed", 42))
    ctx = RecipeContext(cfg=cfg, adapter=adapter)
    ctx = adapter.load_model(ctx)

    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint,
            map_location=ctx.device,
            weights_only=False,
        )
        if hasattr(checkpoint, "to"):
            checkpoint = checkpoint.to(ctx.device)
        ctx.model = checkpoint.eval()

    adapter.evaluate(ctx)


if __name__ == "__main__":
    main()
