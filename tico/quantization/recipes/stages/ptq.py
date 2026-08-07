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
from tico.quantization.recipes.override_policies import (
    apply_ptq_override_policies,
    build_quant_target_resolver_context,
    compile_override_policies,
    compile_raw_overrides,
    parse_named_specs,
)
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


def _resolve_configured_override_paths(
    stage_cfg: Mapping[str, Any],
    *,
    family: str,
    model: Any,
) -> tuple[tuple[str, ...], ...]:
    """Resolve exact observer paths targeted by selector and raw overrides."""
    policies = stage_cfg.get("override_policies")
    raw_overrides = stage_cfg.get("raw_overrides")
    if not policies and not raw_overrides:
        return ()

    specs = parse_named_specs(stage_cfg.get("specs", {}))
    paths: set[tuple[str, ...]] = set()

    if policies:
        context = build_quant_target_resolver_context(family=family, model=model)
        paths.update(
            override.path
            for override in compile_override_policies(
                policies,
                specs=specs,
                context=context,
            )
        )

    paths.update(
        path
        for path, _ in compile_raw_overrides(
            raw_overrides,
            specs=specs,
        )
    )
    return tuple(sorted(paths))


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

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        print("Wrapping model with PTQ wrappers …")
        ptq_config = ctx.adapter.build_ptq_config(ctx, stage_cfg)
        ptq_config = apply_ptq_override_policies(
            ptq_config,
            stage_cfg,
            family=ctx.adapter.family,
            model=ctx.require_model(),
        )

        if bool(stage_cfg.get("print_overrides", False)):
            override_paths = _resolve_configured_override_paths(
                stage_cfg,
                family=ctx.adapter.family,
                model=ctx.require_model(),
            )
            _print_effective_overrides(ptq_config, override_paths)

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
        if bool(stage_cfg.get("print_model", False)):
            print("=== Model after PTQ ===")
            print(ctx.require_model())

        return ctx
