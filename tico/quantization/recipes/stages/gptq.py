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

import torch

from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.base import Stage
from tico.quantization.recipes.utils import filter_dataclass_kwargs, stage_payload


class GPTQStage(Stage):
    name = "gptq"

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        payload = stage_payload(stage_cfg)
        if payload.get("mse") == "smse":
            sensitivity_path = payload.get("sensitivity_path")
            if sensitivity_path:
                payload["sensitivity"] = torch.load(
                    sensitivity_path, map_location="cpu"
                )
            else:
                calibrator = SensitivityCalibrator(
                    ctx.require_model(), ctx.calibration_inputs
                )
                payload["sensitivity"] = calibrator.compute_sensitivity_info()

        # Generic GPTQ now has LLaMA-specific safety switches such as
        # quantize_lm_head=False by default and use_orig_model_inference.
        config_cls = (
            Qwen3VLGPTQConfig if ctx.adapter.family == "qwen3_vl" else GPTQConfig
        )
        gptq_config = config_cls(**filter_dataclass_kwargs(config_cls, payload))

        print(f"Applying {gptq_config.name} …")
        q_model = prepare(ctx.require_model(), gptq_config, inplace=True)

        ctx.adapter.forward_calibration(
            ctx,
            q_model,
            ctx.calibration_inputs,
            desc="GPTQ calibration",
        )

        ctx.model = convert(q_model, inplace=True)
        return ctx
