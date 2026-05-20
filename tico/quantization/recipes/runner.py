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

from pathlib import Path
from typing import Any

from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import save_effective_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages import get_stage
from tico.quantization.recipes.utils import set_seed


class QuantizationRunner:
    """Pipeline runner for model-family adapters and algorithm stages."""

    def run(self, cfg: dict[str, Any]) -> RecipeContext:
        model_cfg = cfg.get("model", {})
        if "family" not in model_cfg:
            raise KeyError("Recipe config requires model.family.")
        if "name_or_path" not in model_cfg:
            raise KeyError("Recipe config requires model.name_or_path.")

        set_seed(cfg.get("runtime", {}).get("seed", 42))

        adapter = get_adapter(model_cfg["family"])
        ctx = RecipeContext(cfg=cfg, adapter=adapter)

        print("=== Loading model ===")
        ctx = adapter.load_model(ctx)

        print("=== Building calibration inputs ===")
        ctx.calibration_inputs = adapter.build_calibration_inputs(ctx)

        print("=== Running quantization pipeline ===")
        for stage_cfg in cfg.get("pipeline", []):
            if not stage_cfg.get("enabled", True):
                print(f"Skipping {stage_cfg.get('name')} …")
                continue
            stage = get_stage(stage_cfg["name"])
            ctx = stage.run(ctx, stage_cfg)

        print("=== Evaluation ===")
        adapter.evaluate(ctx)

        print("=== Export ===")
        adapter.export(ctx)

        output_dir = cfg.get("export", {}).get("output_dir")
        if output_dir:
            save_effective_config(Path(output_dir) / "effective_config.yaml", cfg)

        print("Done.")
        return ctx
