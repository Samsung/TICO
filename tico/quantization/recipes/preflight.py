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

"""Static GPTQ -> PTQ weight-qparam compatibility preflight.

When a pipeline enables both a GPTQ stage and a PTQ stage with
``reuse_gptq_qparams`` (the default), the PTQ stage injects the GPTQ grid
into its weight observers. That handoff requires the GPTQ bit width and the
PTQ weight dtype to describe exactly the same unsigned integer range (see
``tico.quantization.recipes.qparams._validated_gptq_qparams``). A mismatch
is only detected after the full GPTQ pass, wasting the entire calibration
and quantization compute.

Both sides of that contract are known before anything runs:

* GPTQ bit widths come from ``weight_bits`` / ``weight_bits_overrides``
  resolved per module with the quantizers' own enumeration and key-matching
  policy.
* PTQ weight dtypes come from the adapter's PTQ config builder plus the
  recipe override policies, materialized by ``prepare()``.

This module compares the two right after model loading and aborts the run
in seconds on a mismatch. It also warns about ``weight_bits_overrides``
keys that match no quantizable module (silently ignored by the quantizers,
usually a sign of a key written for a different model family).

Only config-decidable properties are checked here (bit width, signedness,
qscheme, channel axis). Checks that need the actual GPTQ statistics (scale
shape/finiteness, zero-point range) remain guarded at injection time.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

from tico.quantization import prepare
from tico.quantization.algorithm.gptq.utils import resolve_weight_bits
from tico.quantization.config.base import BaseConfig
from tico.quantization.config.gptq import GPTQConfig, UniversalGPTQConfig
from tico.quantization.recipes.config import stage_name_matches
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.override_policies import apply_ptq_override_policies
from tico.quantization.recipes.utils import filter_dataclass_kwargs, stage_payload
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class PreflightError(ValueError):
    """GPTQ and PTQ weight quantization configs are statically incompatible."""


@dataclass(frozen=True)
class GptqWeightPlanEntry:
    """Resolved GPTQ bit-width plan for one quantizable module."""

    full_name: str
    bits: int
    source: str
    module_kind: str


@dataclass(frozen=True)
class PtqWeightSpec:
    """PTQ weight-observer spec for one wrapped module."""

    dtype: Any
    qscheme: Any
    channel_axis: Optional[int]


# Output-channel axis the GPTQ quantizers use per module kind; mirrors
# ``tico.quantization.recipes.qparams._validated_gptq_qparams``.
_CHANNEL_AXIS_BY_KIND = {
    "Linear": 0,
    "Conv1d": 0,
    "Conv2d": 0,
    "Conv3d": 0,
    "ConvTranspose2d": 1,
}


def _find_enabled_stage(
    cfg: Mapping[str, Any],
    stage_name: str,
) -> Mapping[str, Any] | None:
    """Return the first enabled pipeline stage with the requested name."""
    for stage_cfg in cfg.get("pipeline", []):
        if not isinstance(stage_cfg, Mapping):
            continue
        if not stage_name_matches(stage_cfg.get("name"), stage_name):
            continue
        if not stage_cfg.get("enabled", True):
            continue
        return stage_cfg
    return None


def _reuse_gptq_qparams(stage_cfg: Mapping[str, Any]) -> bool:
    """Mirror the PTQ stage's reuse flag resolution (default: enabled)."""
    value = stage_cfg.get("reuse_gptq_qparams", True)
    if not isinstance(value, bool):
        raise TypeError(
            "ptq.reuse_gptq_qparams must be a boolean. " f"got {type(value).__name__}"
        )
    return value


def _build_gptq_config(
    adapter: Any,
    gptq_stage_cfg: Mapping[str, Any],
) -> BaseConfig:
    """Instantiate the same GPTQ config class the GPTQ stage would use."""
    payload = stage_payload(gptq_stage_cfg)
    variant = str(payload.pop("variant", "default")).strip().lower()
    if variant not in {"default", "universal"}:
        raise ValueError(
            f"Unsupported GPTQ variant {variant!r}. "
            "Supported variants: default, universal."
        )
    if variant == "universal":
        config_cls: type[BaseConfig] = UniversalGPTQConfig
    else:
        # Imported lazily so this module stays importable without the
        # model-family dependencies the built-in adapters pull in (mirrors
        # the GPTQ stage).
        from tico.quantization.recipes.adapters import resolve_gptq_config_class

        config_cls = resolve_gptq_config_class(adapter) or GPTQConfig
    return config_cls(**filter_dataclass_kwargs(config_cls, payload))


def plan_gptq_weight_bits(
    model: torch.nn.Module,
    gptq_config: BaseConfig,
    *,
    target_roots: Sequence[str] | None = None,
) -> tuple[dict[str, GptqWeightPlanEntry], list[str]]:
    """Resolve the effective GPTQ bit width of every quantizable module.

    Uses the same layer enumeration (``get_quantizable_layers``) and the
    same override key-matching policy (``resolve_weight_bits``) as the
    GPTQ-family quantizers, so the plan matches what the GPTQ stage will
    actually apply.

    Args:
        model: Loaded model.
        gptq_config: GPTQ stage configuration.
        target_roots: Optional module-name prefixes restricting the plan to
            the components the family quantizer processes (see
            ``ModelAdapter.gptq_target_roots``); ``None`` plans the whole
            model.

    Returns:
        A pair of the per-module plan (keyed by full module name) and the
        override keys that matched no module.
    """
    # Shared by every family-specific GPTQ quantizer; imported lazily to
    # keep module import light.
    from tico.quantization.algorithm.qwen3_vl_gptq.utils import get_quantizable_layers

    overrides: Mapping[str, int] = (
        getattr(gptq_config, "weight_bits_overrides", None) or {}
    )
    default_bits = int(getattr(gptq_config, "weight_bits", 8))
    quantize_lm_head = bool(getattr(gptq_config, "quantize_lm_head", False))

    plan: dict[str, GptqWeightPlanEntry] = {}
    used_keys: set[str] = set()
    for full_name, module in get_quantizable_layers(model).items():
        if target_roots is not None and not any(
            full_name == root or full_name.startswith(f"{root}.")
            for root in target_roots
        ):
            continue
        if not quantize_lm_head and (
            full_name == "lm_head" or full_name.endswith(".lm_head")
        ):
            continue
        # The quantizers resolve overrides with the stage-local module name;
        # its trailing component is identical for the suffix policy, so the
        # whole-model enumeration resolves to the same bits.
        local_name = full_name.rsplit(".", 1)[-1]
        bits = int(
            resolve_weight_bits(
                default_bits=default_bits,
                weight_bits_overrides=overrides,
                full_module_name=full_name,
                local_module_name=local_name,
            )
        )
        if full_name in overrides:
            used_keys.add(full_name)
            source = f"weight_bits_overrides[{full_name!r}]"
        elif local_name in overrides:
            used_keys.add(local_name)
            source = f"weight_bits_overrides[{local_name!r}]"
        else:
            suffix_keys = [key for key in overrides if full_name.endswith(f".{key}")]
            if suffix_keys:
                # resolve_weight_bits keeps the last suffix match.
                used_keys.add(suffix_keys[-1])
                source = f"weight_bits_overrides[{suffix_keys[-1]!r}] (suffix match)"
            else:
                source = "weight_bits default"
        plan[full_name] = GptqWeightPlanEntry(
            full_name=full_name,
            bits=bits,
            source=source,
            module_kind=type(module).__name__,
        )

    unused_keys = [key for key in overrides if key not in used_keys]
    return plan, unused_keys


def collect_ptq_weight_specs(
    ctx: RecipeContext,
    ptq_stage_cfg: Mapping[str, Any],
) -> dict[str, PtqWeightSpec]:
    """Resolve the PTQ weight-observer spec of every wrapped module.

    Builds the PTQ config exactly like the PTQ stage does, then applies it
    to a meta-device twin of the loaded model so the real ``prepare()``
    wrapper/observer resolution produces the specs without touching (or
    copying) the trained weights.
    """
    model = ctx.require_model()
    ptq_config = ctx.adapter.build_ptq_config(ctx, ptq_stage_cfg)
    ptq_config = apply_ptq_override_policies(
        ptq_config,
        ptq_stage_cfg,
        family=ctx.adapter.family,
        model=model,
        adapter=ctx.adapter,
    )
    twin = ctx.adapter.build_meta_model(ctx)
    q_model = prepare(twin, ptq_config)

    specs: dict[str, PtqWeightSpec] = {}
    for module in q_model.modules():
        if not isinstance(module, QuantModuleBase):
            continue
        fp_name = module.fp_name
        if not fp_name:
            continue
        obs = module.get_observer("weight")
        if obs is None:
            continue
        specs[fp_name] = PtqWeightSpec(
            dtype=getattr(obs, "dtype", None),
            qscheme=getattr(obs, "qscheme", None),
            channel_axis=getattr(obs, "channel_axis", None),
        )
    return specs


def check_gptq_ptq_compat(
    plan: Mapping[str, GptqWeightPlanEntry],
    specs: Mapping[str, PtqWeightSpec],
    gptq_config: BaseConfig,
) -> tuple[list[str], list[str]]:
    """Compare the static GPTQ plan against the PTQ weight specs.

    Mirrors the config-decidable checks of
    ``tico.quantization.recipes.qparams._validated_gptq_qparams``.

    Returns:
        A pair of (errors, warnings). Errors abort the run; warnings
        describe modules whose GPTQ qparams cannot be reused but do not
        fail the handoff.
    """
    errors: list[str] = []
    warnings: list[str] = []

    groupsize = int(getattr(gptq_config, "groupsize", -1))
    if groupsize != -1:
        errors.append(
            f"  - GPTQ groupsize={groupsize} produces grouped grids whose "
            "scale layout cannot be mapped onto PTQ weight observers; qparam "
            "reuse only supports groupsize=-1. Set pipeline.gptq.groupsize=-1 "
            "or pipeline.ptq.reuse_gptq_qparams=false."
        )

    for full_name, entry in plan.items():
        spec = specs.get(full_name)
        if spec is None:
            warnings.append(
                f"{full_name!r}: GPTQ will quantize this module, but PTQ "
                "wraps it without a weight observer; its GPTQ qparams cannot "
                "be reused."
            )
            continue
        dtype = spec.dtype
        if not isinstance(dtype, DType):
            errors.append(
                f"  - {full_name!r}: PTQ weight observer dtype {dtype!r} is "
                "not an affine integer dtype; qparam reuse requires an "
                "affine integer weight dtype."
            )
            continue
        expected_maxq = (1 << entry.bits) - 1
        problems: list[str] = []
        if dtype.signed or dtype.qmin != 0:
            problems.append("PTQ weight dtype is signed")
        if dtype.qmax != expected_maxq:
            problems.append("weight bit width differs")
        if not (
            isinstance(spec.qscheme, QScheme)
            and spec.qscheme in {QScheme.PER_CHANNEL_ASYMM, QScheme.PER_TENSOR_ASYMM}
        ):
            problems.append(f"qscheme={spec.qscheme} is not asymmetric affine")
        elif spec.qscheme.is_per_channel():
            expected_axis = _CHANNEL_AXIS_BY_KIND.get(entry.module_kind)
            if expected_axis is not None and spec.channel_axis != expected_axis:
                problems.append(f"channel axis {spec.channel_axis} != {expected_axis}")
        if problems:
            hint_key = ".".join(full_name.split(".")[-2:])
            errors.append(
                f"  - {full_name!r}: GPTQ {entry.bits}-bit unsigned "
                f"(maxq={expected_maxq}, {entry.source}) vs PTQ {dtype} "
                f"range=[{dtype.qmin}, {dtype.qmax}], "
                f"qscheme={spec.qscheme}, axis={spec.channel_axis}: "
                f"{'; '.join(problems)}.\n"
                "    Align the configs, e.g. --set "
                f'\'pipeline.gptq.weight_bits_overrides={{"{hint_key}": '
                f"{dtype.bits}}}' or set the PTQ weight spec for this module "
                f"to uint{entry.bits}."
            )
    return errors, warnings


def run_gptq_ptq_preflight(ctx: RecipeContext) -> None:
    """Validate GPTQ<->PTQ weight-qparam compatibility before any stage runs.

    No-op unless the pipeline enables a GPTQ stage. Override keys that match
    no quantizable module always produce a warning. When a PTQ stage with
    ``reuse_gptq_qparams`` (the default) is also enabled, a static
    incompatibility raises :class:`PreflightError` before any calibration
    compute. Set ``pipeline.ptq.preflight: false`` to skip the check.
    """
    gptq_stage = _find_enabled_stage(ctx.cfg, "gptq")
    if gptq_stage is None:
        return
    model = ctx.require_model()
    if not isinstance(model, torch.nn.Module):
        # Duck-typed test doubles and non-module fakes have no module tree to
        # check; defer to the runtime validation in the stages.
        return
    gptq_config = _build_gptq_config(ctx.adapter, gptq_stage)
    hook = getattr(ctx.adapter, "gptq_target_roots", None)
    try:
        target_roots = hook(model, gptq_config) if hook is not None else None
    except Exception as exc:  # filtering must not block the check itself
        print(
            "[Preflight] WARNING: could not resolve GPTQ target components "
            f"({type(exc).__name__}: {exc}); checking all quantizable modules."
        )
        target_roots = None
    plan, unused_keys = plan_gptq_weight_bits(
        model, gptq_config, target_roots=target_roots
    )
    if unused_keys and bool(stage_payload(gptq_stage).get("strict_overrides", True)):
        keys = ", ".join(repr(key) for key in unused_keys)
        raise PreflightError(
            "[Preflight] pipeline.gptq.weight_bits_overrides key(s) "
            f"{keys} matched no quantizable module and will be ignored. "
            "Fix the key(s), or set pipeline.gptq.strict_overrides=false "
            "to demote this to a warning (strict_overrides defaults to true)."
        )
    for key in unused_keys:
        print(
            f"[Preflight] WARNING: pipeline.gptq.weight_bits_overrides key "
            f"{key!r} matched no quantizable module and will be ignored."
        )

    ptq_stage = _find_enabled_stage(ctx.cfg, "ptq")
    if ptq_stage is None or not _reuse_gptq_qparams(ptq_stage):
        return
    if not bool(ptq_stage.get("preflight", True)):
        return

    try:
        specs = collect_ptq_weight_specs(ctx, ptq_stage)
    except Exception as exc:  # tooling failure must not block a valid run
        print(
            "[Preflight] WARNING: could not resolve PTQ weight specs "
            f"({type(exc).__name__}: {exc}); skipping the GPTQ->PTQ "
            "compatibility check."
        )
        return

    errors, warnings = check_gptq_ptq_compat(plan, specs, gptq_config)
    for warning in warnings:
        print(f"[Preflight] WARNING: {warning}")
    if errors:
        raise PreflightError(
            "[Preflight] GPTQ and PTQ weight quantization configs are "
            f"incompatible ({len(errors)} problem(s)); the run was aborted "
            "before quantization:\n" + "\n".join(errors)
        )
    matched = sum(1 for name in plan if name in specs)
    print(
        f"[Preflight] GPTQ->PTQ weight qparams compatible for " f"{matched} module(s)."
    )
