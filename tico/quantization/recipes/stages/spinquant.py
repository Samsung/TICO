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
from tico.quantization.config.spinquant import SpinQuantConfig
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.base import Stage


class SpinQuantStage(Stage):
    name = "spinquant"

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        # Model-family-specific SpinQuant should live in the adapter. Qwen3-VL,
        # for example, uses Qwen3VLSpinQuantConfig rather than generic
        # SpinQuantConfig.
        apply_spinquant = getattr(ctx.adapter, "apply_spinquant", None)
        if callable(apply_spinquant):
            ctx.model = apply_spinquant(ctx, stage_cfg)
            return ctx

        print("Applying SpinQuant preprocessing …")
        q_model = prepare(ctx.require_model(), SpinQuantConfig())
        ctx.model = convert(q_model)
        return ctx
