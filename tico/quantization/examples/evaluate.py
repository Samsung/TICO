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

import torch

from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.context import RecipeContext
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
        help="Override eval tasks. LLM: lm_eval_tasks, VLM: vlm_tasks.",
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    overrides.append("evaluation.enabled=true")
    if args.model:
        overrides.append(f"model.name_or_path={args.model}")
    if args.device:
        overrides.append(f"runtime.device={args.device}")

    cfg = load_recipe_config(args.config, overrides=overrides)
    set_seed(cfg.get("runtime", {}).get("seed", 42))
    adapter = get_adapter(cfg["model"]["family"])
    ctx = RecipeContext(cfg=cfg, adapter=adapter)
    ctx = adapter.load_model(ctx)

    if args.checkpoint:
        ctx.model = torch.load(args.checkpoint, weights_only=False).eval()

    if args.tasks:
        if adapter.family == "llama":
            cfg.setdefault("evaluation", {})["lm_eval_tasks"] = args.tasks
        else:
            cfg.setdefault("evaluation", {})["vlm_tasks"] = [
                t.strip() for t in args.tasks.split(",") if t.strip()
            ]

    adapter.evaluate(ctx)


if __name__ == "__main__":
    main()
