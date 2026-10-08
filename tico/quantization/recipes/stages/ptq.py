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
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.override_policies import apply_ptq_override_policies
from tico.quantization.recipes.qparams import (
    clear_gptq_quantizers,
    find_gptq_quantizers,
    inject_gptq_qparams,
)
from tico.quantization.recipes.stages.base import Stage


_INTERNAL_OVERRIDE_FIELDS = frozenset({"__quant_spec_replace_role__"})
_OVERRIDE_FIELD_ORDER = (
    "observer",
    "dtype",
    "qscheme",
    "elem_format",
    "axis",
    "shared_exp_method",
    "round",
)


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


def _qparam_reuse_verbose(
    ctx: RecipeContext,
    stage_cfg: Mapping[str, Any],
) -> bool:
    """Return whether GPTQ-to-PTQ qparam reuse should print a summary."""
    if "verbose" in stage_cfg:
        return bool(stage_cfg.get("verbose"))

    runtime_cfg = ctx.cfg.get("runtime", {})
    if isinstance(runtime_cfg, Mapping) and "verbose" in runtime_cfg:
        return bool(runtime_cfg.get("verbose"))

    gptq_stage = _find_enabled_stage(ctx.cfg, "gptq")
    return bool(gptq_stage and gptq_stage.get("verbose", False))


def _reuse_gptq_qparams(stage_cfg: Mapping[str, Any]) -> bool:
    """Return whether PTQ should reuse qparams produced by GPTQ."""
    value = stage_cfg.get("reuse_gptq_qparams", True)
    if not isinstance(value, bool):
        raise TypeError(
            "ptq.reuse_gptq_qparams must be a boolean. " f"got {type(value).__name__}"
        )
    return value


def _get_effective_override_fields(
    overrides: Mapping[str, Any],
    path: tuple[str, ...],
) -> dict[str, Any]:
    """Read normalized override fields at one exact observer path."""
    current: Any = overrides
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            dotted_path = ".".join(path)
            raise KeyError(f"Effective PTQ override path is missing: {dotted_path}")
        current = current[key]

    if not isinstance(current, Mapping):
        dotted_path = ".".join(path)
        raise TypeError(
            "Effective PTQ override must resolve to a mapping at "
            f"{dotted_path}, got {type(current).__name__}."
        )

    return {
        str(key): value
        for key, value in current.items()
        if str(key) not in _INTERNAL_OVERRIDE_FIELDS
    }


def _format_override_value(value: Any) -> str:
    """Return a compact display value for a normalized override field."""
    name = getattr(value, "__name__", None)
    if isinstance(name, str):
        return name
    return str(value)


def _format_effective_override(fields: Mapping[str, Any]) -> str:
    """Format one normalized effective override."""
    ordered_keys = [key for key in _OVERRIDE_FIELD_ORDER if key in fields]
    ordered_keys.extend(sorted(key for key in fields if key not in ordered_keys))
    return ", ".join(
        f"{key}={_format_override_value(fields[key])}" for key in ordered_keys
    )


def _print_effective_overrides(
    ptq_config: PTQConfig,
    override_paths: tuple[tuple[str, ...], ...],
) -> None:
    """Print final values only for observer paths targeted by recipe overrides."""
    print("=== Effective PTQ overrides ===")
    if not override_paths:
        print("  <none>")
    else:
        for path in override_paths:
            fields = _get_effective_override_fields(ptq_config.overrides, path)
            print(f"  {'.'.join(path)}: {_format_effective_override(fields)}")
    print()


class PTQStage(Stage):
    name = "ptq"
    requires_calibration_inputs = True

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        print("Wrapping model with PTQ wrappers …")
        reuse_gptq_qparams = _reuse_gptq_qparams(stage_cfg)
        ptq_config = ctx.adapter.build_ptq_config(ctx, stage_cfg)
        # Collect the exact paths applied by recipe overrides only when they
        # will be printed, so the diagnostic reuses the single policy
        # resolution instead of re-interpreting the stage configuration.
        applied_override_paths: set[tuple[str, ...]] | None = (
            set() if bool(stage_cfg.get("print_overrides", False)) else None
        )
        ptq_config = apply_ptq_override_policies(
            ptq_config,
            stage_cfg,
            family=ctx.adapter.family,
            model=ctx.require_model(),
            applied_paths=applied_override_paths,
        )

        if applied_override_paths is not None:
            _print_effective_overrides(
                ptq_config,
                tuple(sorted(applied_override_paths)),
            )

        q_model = prepare(ctx.require_model(), ptq_config)

        _, quantizers = find_gptq_quantizers(q_model)

        if not reuse_gptq_qparams:
            if quantizers is not None:
                clear_gptq_quantizers(q_model)
            print(
                "[Info] GPTQ qparam reuse disabled by config "
                "(reuse_gptq_qparams=false); PTQ weight observers will compute "
                "qparams from the current weights."
            )
        elif not quantizers:
            if _find_enabled_stage(ctx.cfg, "gptq") is not None:
                raise RuntimeError(
                    "GPTQ qparam reuse was requested, but no GPTQ quantizers "
                    "were found after an enabled GPTQ stage."
                )
            print(
                "[Warn] GPTQ quantizers were not found; "
                "PTQ weight observers will use PTQ statistics."
            )
        else:
            # Quantizers may live on the original FP owner, but weight observers
            # always live in the prepared PTQ tree.
            stats = inject_gptq_qparams(
                q_model,
                quantizers,
                verbose=_qparam_reuse_verbose(ctx, stage_cfg),
            )
            clear_gptq_quantizers(q_model)

            if stats["matched"] == 0:
                raise RuntimeError(
                    "GPTQ quantizers were found, but no PTQ weight observer "
                    "reused their qparams. Check wrapper fp_name mappings and "
                    "GPTQ/PTQ weight quantization policies."
                )

            print(
                f"[Info] Reused GPTQ qparams for {stats['matched']} "
                "PTQ weight observer(s)."
            )

        ctx.adapter.calibrate_prepared_model(ctx, q_model, stage_cfg)

        ctx.model = convert(q_model)
        if bool(stage_cfg.get("print_model", False)):
            print("=== Model after PTQ ===")
            print(ctx.require_model())

        return ctx
