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

import torch

from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import load_recipe_config, save_effective_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.utils import set_seed, torch_dtype_from_name


def _parse_artifacts(value: str) -> str:
    artifacts = [item.strip() for item in value.split(",") if item.strip()]
    if not artifacts:
        raise argparse.ArgumentTypeError("--artifacts must contain at least one name")
    return "[" + ",".join(artifacts) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export artifacts from a saved fake-quant checkpoint."
    )
    parser.add_argument("--config", required=True, help="Base recipe config.")
    parser.add_argument(
        "--checkpoint", required=True, help="Torch checkpoint to export from."
    )
    parser.add_argument("--model", default=None, help="Override model.name_or_path.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    parser.add_argument(
        "--output-dir", default=None, help="Override export.output_dir."
    )
    parser.add_argument(
        "--artifacts",
        type=_parse_artifacts,
        default=None,
        help="Comma-separated export artifacts, for example circle_per_layer.",
    )
    parser.add_argument(
        "--no-prefill-decode",
        action="store_true",
        help=(
            "Disable paired prefill/decode layer export when the config enables "
            "export.prefill_decode."
        ),
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    overrides.append("export.enabled=true")
    if args.model:
        overrides.append(f"model.name_or_path={args.model}")
    if args.device:
        overrides.append(f"runtime.device={args.device}")
    if args.output_dir:
        overrides.append(f"export.output_dir={args.output_dir}")
    if args.artifacts:
        overrides.append(f"export.artifacts={args.artifacts}")
    if args.no_prefill_decode:
        overrides.append("export.prefill_decode=false")

    cfg = load_recipe_config(args.config, overrides=overrides)
    set_seed(cfg.get("runtime", {}).get("seed", 42))

    runtime_cfg = cfg.get("runtime", {})
    device = torch.device(runtime_cfg.get("device", "cpu"))
    dtype = torch_dtype_from_name(runtime_cfg.get("dtype", "float32"))

    adapter = get_adapter(cfg["model"]["family"])
    ctx = RecipeContext(cfg=cfg, adapter=adapter, device=device, dtype=dtype)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )
    if hasattr(checkpoint, "to"):
        checkpoint = checkpoint.to(device)
    ctx.model = checkpoint.eval()

    adapter.export(ctx)

    output_dir = cfg.get("export", {}).get("output_dir")
    if output_dir:
        save_effective_config(Path(output_dir) / "effective_config.yaml", cfg)

    print("Done.")


if __name__ == "__main__":
    main()
