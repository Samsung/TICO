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
from tico.quantization.recipes.override_policies import apply_ptq_override_policies
from tico.quantization.recipes.qparams import (
    clear_gptq_quantizers,
    find_gptq_quantizers,
    inject_gptq_qparams,
)
from tico.quantization.recipes.stages.base import Stage


def _find_enabled_stage(
    cfg: Mapping[str, Any],
    stage_name: str,
) -> Mapping[str, Any] | None:
    """Return an enabled stage from the recipe pipeline."""
    for stage_cfg in cfg.get("pipeline", []):
        if not isinstance(stage_cfg, Mapping):
            continue
        if stage_cfg.get("name") != stage_name:
            continue
        if not stage_cfg.get("enabled", True):
            continue
        return stage_cfg
    return None


def _qparam_injection_verbose(
    ctx: RecipeContext,
    stage_cfg: Mapping[str, Any],
) -> bool:
    """Return whether GPTQ-to-PTQ qparam injection should print a summary."""
    if "verbose" in stage_cfg:
        return bool(stage_cfg.get("verbose"))

    runtime_cfg = ctx.cfg.get("runtime", {})
    if isinstance(runtime_cfg, Mapping) and "verbose" in runtime_cfg:
        return bool(runtime_cfg.get("verbose"))

    gptq_stage = _find_enabled_stage(ctx.cfg, "gptq")
    return bool(gptq_stage and gptq_stage.get("verbose", False))


class PTQStage(Stage):
    name = "ptq"

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        print("Wrapping model with PTQ wrappers …")
        ptq_config = ctx.adapter.build_ptq_config(ctx, stage_cfg)
        ptq_config = apply_ptq_override_policies(
            ptq_config,
            stage_cfg,
            family=ctx.adapter.family,
            model=ctx.require_model(),
        )
        q_model = prepare(ctx.require_model(), ptq_config)

        owner, quantizers = find_gptq_quantizers(q_model)
        if quantizers:
            inject_gptq_qparams(
                q_model if owner is q_model else owner,
                quantizers,
                verbose=_qparam_injection_verbose(ctx, stage_cfg),
            )
            clear_gptq_quantizers(q_model)
        else:
            print(
                "[Warn] GPTQ quantizers were not found; "
                "PTQ weight observers will use PTQ statistics."
            )

        ctx.adapter.calibrate_prepared_model(ctx, q_model, stage_cfg)

        ctx.model = convert(q_model)
        return ctx
