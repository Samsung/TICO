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

from typing import Any, Mapping

from tico.quantization import convert, prepare
from tico.quantization.config.cle import CLEConfig
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.base import Stage


def parse_cle_pairs(raw_pairs: list[str] | None) -> list[tuple[str, str]]:
    if not raw_pairs:
        return []
    pairs: list[tuple[str, str]] = []
    for raw_pair in raw_pairs:
        if ":" not in raw_pair:
            raise ValueError(
                f"CLE pair must be formatted as first_layer:second_layer. got: {raw_pair}"
            )
        first, second = [part.strip() for part in raw_pair.split(":", 1)]
        if not first or not second:
            raise ValueError(f"Invalid CLE pair: {raw_pair}")
        pairs.append((first, second))
    return pairs


class CLEStage(Stage):
    name = "cle"

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        pairs = parse_cle_pairs(stage_cfg.get("pairs"))
        if not pairs:
            print("Skipping CLE: no layer pairs were provided.")
            return ctx

        print("Applying Cross-Layer Equalization preprocessing …")
        cle_config = CLEConfig(
            pairs=pairs,
            method=stage_cfg.get("method", "absmax"),
            max_iter=int(stage_cfg.get("max_iter", 1)),
            show_progress=bool(stage_cfg.get("show_progress", True)),
        )
        q_model = prepare(ctx.require_model(), cle_config)
        ctx.model = convert(q_model)
        return ctx
