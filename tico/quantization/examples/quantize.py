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

from tico.quantization.recipes.config import load_recipe_config
from tico.quantization.recipes.runner import QuantizationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a config-driven quantization recipe."
    )
    parser.add_argument(
        "--config", required=True, help="Path to a recipe YAML/JSON config."
    )
    parser.add_argument("--model", default=None, help="Override model.name_or_path.")
    parser.add_argument("--device", default=None, help="Override runtime.device.")
    parser.add_argument(
        "--output-dir", default=None, help="Override export.output_dir."
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values using dotted paths. Example: --set pipeline.0.enabled=false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = list(args.set)
    if args.model:
        overrides.append(f"model.name_or_path={args.model}")
    if args.device:
        overrides.append(f"runtime.device={args.device}")
    if args.output_dir:
        overrides.append(f"export.output_dir={args.output_dir}")

    cfg = load_recipe_config(args.config, overrides=overrides)
    QuantizationRunner().run(cfg)


if __name__ == "__main__":
    main()
