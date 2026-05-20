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
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.qparams import (
    clear_gptq_quantizers,
    find_gptq_quantizers,
    inject_gptq_qparams,
)
from tico.quantization.recipes.stages.base import Stage


class PTQStage(Stage):
    name = "ptq"

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        print("Wrapping model with PTQ wrappers …")
        ptq_config = ctx.adapter.build_ptq_config(ctx, stage_cfg)
        q_model = prepare(ctx.require_model(), ptq_config)

        owner, quantizers = find_gptq_quantizers(q_model)
        if quantizers:
            inject_gptq_qparams(
                q_model if owner is q_model else owner,
                quantizers,
                verbose=bool(stage_cfg.get("verbose", False)),
            )
            clear_gptq_quantizers(q_model)
        else:
            print(
                "[Warn] GPTQ quantizers were not found; PTQ weight observers will use PTQ statistics."
            )

        ctx.adapter.calibrate_prepared_model(ctx, q_model, stage_cfg)

        ctx.model = convert(q_model)
        return ctx
