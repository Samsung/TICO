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

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.config import save_effective_config
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages import get_stage
from tico.quantization.recipes.utils import set_seed


def _find_stage(
    cfg: Mapping[str, Any],
    stage_name: str,
) -> Mapping[str, Any] | None:
    """Return the first pipeline stage with the requested name."""
    for stage_cfg in cfg.get("pipeline", []):
        if isinstance(stage_cfg, Mapping) and stage_cfg.get("name") == stage_name:
            return stage_cfg
    return None


def _is_stage_enabled(stage_cfg: Mapping[str, Any] | None) -> bool:
    """Return whether a pipeline stage exists and is enabled."""
    return stage_cfg is not None and bool(stage_cfg.get("enabled", True))


def _stage_value(
    stage_cfg: Mapping[str, Any] | None,
    key: str,
    default: Any = "not set",
) -> Any:
    """Read a scalar value from a stage config."""
    if stage_cfg is None:
        return default
    value = stage_cfg.get(key, default)
    return default if value is None else value


def _first_value(*values: Any, default: Any = "not set") -> Any:
    """Return the first non-None value."""
    for value in values:
        if value is not None:
            return value
    return default


def _format_quant_spec(value: Any, default: Any = "not set") -> Any:
    """Return a compact, human-readable quantization spec."""
    if value is None:
        return default

    if isinstance(value, Mapping):
        kind = str(value.get("kind", value.get("type", "affine"))).strip().lower()
        if kind == "mx":
            elem_format = value.get("elem_format", "fp8_e4m3")
            axis = value.get("axis", -1)
            shared_exp_method = value.get("shared_exp_method")
            rounding = value.get("round")
            parts = [f"elem_format={elem_format}", f"axis={axis}"]
            if shared_exp_method is not None:
                parts.append(f"shared_exp_method={shared_exp_method}")
            if rounding is not None:
                parts.append(f"round={rounding}")
            return f"mx({', '.join(parts)})"

        if kind == "affine":
            dtype = value.get("dtype", default)
            qscheme = value.get("qscheme")
            observer = value.get("observer")
            parts = [str(dtype)]
            if qscheme is not None:
                parts.append(f"qscheme={qscheme}")
            if observer is not None:
                parts.append(f"observer={observer}")
            return ", ".join(parts)

        return dict(value)

    return value


def _stage_quant_spec(
    stage_cfg: Mapping[str, Any] | None,
    key: str,
    default: Any = "not set",
) -> Any:
    """Read and format a quantization spec from a stage config."""
    return _format_quant_spec(_stage_value(stage_cfg, key, None), default)


def _format_override_policy_spec(
    policy: Mapping[str, Any],
    specs: Mapping[str, Any],
) -> str:
    """Format a policy spec and expand a referenced named spec."""
    spec = policy.get("spec", "not set")
    if isinstance(spec, str) and spec in specs:
        return f"{spec} [{_format_quant_spec(specs[spec])}]"
    return str(_format_quant_spec(spec))


def _format_override_policy(
    policy: Mapping[str, Any],
    specs: Mapping[str, Any],
) -> str:
    """Format one PTQ override policy for the config summary."""
    name = str(policy.get("name", "<unnamed>"))
    status = "" if bool(policy.get("enabled", True)) else " (disabled)"

    target = policy.get("target", {})
    if not isinstance(target, Mapping):
        target = {}

    fields = [
        f"spec={_format_override_policy_spec(policy, specs)}",
        f"component={target.get('component', 'all')}",
        f"layers={target.get('layers', 'all')}",
    ]

    for key in ("module", "modules", "op_type"):
        value = target.get(key)
        if value is not None:
            fields.append(f"{key}={value}")

    for key in ("observers", "observer_role"):
        value = target.get(key)
        if value is not None:
            fields.append(f"{key}={value}")

    if bool(policy.get("allow_empty", False)):
        fields.append("allow_empty=True")

    return f"{name}{status} -> {', '.join(fields)}"


def _print_override_policy_summary(
    ptq_stage: Mapping[str, Any] | None,
) -> None:
    """Print PTQ override policies as part of the config summary."""
    if ptq_stage is None:
        return

    policies = ptq_stage.get("override_policies")
    if (
        not isinstance(policies, Sequence)
        or isinstance(policies, (str, bytes))
        or not policies
    ):
        return

    specs = ptq_stage.get("specs", {})
    if not isinstance(specs, Mapping):
        specs = {}

    formatted = [
        _format_override_policy(policy, specs)
        for policy in policies
        if isinstance(policy, Mapping)
    ]
    if not formatted:
        return

    print()
    print("=== PTQ override policies ===")
    for policy in formatted:
        print(f"  {policy}")


def _gptq_weight_bits(gptq_stage: Mapping[str, Any] | None) -> Any:
    """Return a readable GPTQ weight bit-width fallback for summaries."""
    bits = _stage_value(gptq_stage, "weight_bits", None)
    if bits is None:
        return None
    return f"gptq:{bits}-bit"


def _calibration_sample_count(calibration_cfg: Mapping[str, Any]) -> Any:
    """Return a readable calibration sample count."""
    datasets = calibration_cfg.get("datasets")
    if isinstance(datasets, list):
        total = 0
        found = False
        for dataset_cfg in datasets:
            if not isinstance(dataset_cfg, Mapping):
                continue
            n_samples = dataset_cfg.get("n_samples")
            if n_samples is None:
                continue
            try:
                total += int(n_samples)
                found = True
            except (TypeError, ValueError):
                return "mixed"
        if found:
            return total

    return calibration_cfg.get("n_samples", "not set")


def _max_seq_len(cfg: Mapping[str, Any]) -> Any:
    """Return the most relevant max sequence length."""
    evaluation_cfg = cfg.get("evaluation", {})
    export_cfg = cfg.get("export", {})
    calibration_cfg = cfg.get("calibration", {})

    if isinstance(evaluation_cfg, Mapping) and evaluation_cfg.get("max_seq_len"):
        return evaluation_cfg["max_seq_len"]
    if isinstance(export_cfg, Mapping) and export_cfg.get("max_seq_len"):
        return export_cfg["max_seq_len"]
    if isinstance(calibration_cfg, Mapping) and calibration_cfg.get("seq_len"):
        return calibration_cfg["seq_len"]
    return "not set"


def _print_config_row(label: str, value: Any) -> None:
    """Print one aligned config summary row."""
    print(f"{label:<22}: {value}")


def _print_config_summary(cfg: Mapping[str, Any]) -> None:
    """Print the high-level quantization recipe configuration."""
    runtime_cfg = cfg.get("runtime", {})
    if isinstance(runtime_cfg, Mapping) and not bool(
        runtime_cfg.get("print_config", True)
    ):
        return

    model_cfg = cfg.get("model", {})
    calibration_cfg = cfg.get("calibration", {})
    model_args = cfg.get("model_args", {})

    if not isinstance(model_cfg, Mapping):
        model_cfg = {}
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}
    if not isinstance(calibration_cfg, Mapping):
        calibration_cfg = {}
    if not isinstance(model_args, Mapping):
        model_args = {}

    spinquant_stage = _find_stage(cfg, "spinquant")
    cle_stage = _find_stage(cfg, "cle")
    gptq_stage = _find_stage(cfg, "gptq")
    ptq_stage = _find_stage(cfg, "ptq")

    spinquant_enabled = _is_stage_enabled(spinquant_stage)
    gptq_enabled = _is_stage_enabled(gptq_stage)
    ptq_enabled = _is_stage_enabled(ptq_stage)

    linear_weight = _first_value(
        _stage_quant_spec(ptq_stage, "linear_weight", None),
        _gptq_weight_bits(gptq_stage),
    )
    vision_patch_embed_weight = _stage_quant_spec(
        ptq_stage, "vision_patch_embed_weight", None
    )
    norm = _stage_quant_spec(ptq_stage, "norm", None)
    norm_weight = _stage_quant_spec(ptq_stage, "norm_weight", None)
    spin_rotation_weight = (
        _stage_quant_spec(ptq_stage, "spin_rotation_weight")
        if spinquant_enabled
        else "disabled"
    )
    profile = _first_value(
        _stage_value(ptq_stage, "profile", None),
        model_args.get("profile"),
    )

    print("=== Config ===")
    _print_config_row("Model", model_cfg.get("name_or_path", "not set"))
    _print_config_row("Device", runtime_cfg.get("device", "auto"))
    _print_config_row("DType", runtime_cfg.get("dtype", "float32"))
    _print_config_row("Seed", runtime_cfg.get("seed", 42))
    _print_config_row("GPTQ enabled", gptq_enabled)
    _print_config_row(
        "GPTQ lm_head enabled",
        bool(_stage_value(gptq_stage, "quantize_lm_head", False)),
    )
    _print_config_row("PTQ enabled", ptq_enabled)
    _print_config_row("SpinQuant enabled", spinquant_enabled)
    _print_config_row("CLE enabled", _is_stage_enabled(cle_stage))
    _print_config_row("Activation", _stage_quant_spec(ptq_stage, "activation"))
    _print_config_row("Linear weight", linear_weight)
    if vision_patch_embed_weight is not None:
        _print_config_row("Vision patch weight", vision_patch_embed_weight)
    _print_config_row(
        "Embedding weight",
        _stage_quant_spec(ptq_stage, "embedding_weight"),
    )
    _print_config_row(
        "LM head weight",
        _stage_quant_spec(ptq_stage, "lm_head_weight"),
    )
    if norm is not None:
        _print_config_row("Norm", norm)
    if norm_weight is not None:
        _print_config_row("Norm weight", norm_weight)
    _print_config_row("Spin rotation weight", spin_rotation_weight)
    _print_config_row("Calibration samples", _calibration_sample_count(calibration_cfg))
    _print_config_row(
        "Calibration seq length",
        calibration_cfg.get("seq_len", "not set"),
    )
    _print_config_row("Max seq length", _max_seq_len(cfg))
    _print_config_row("Profile", profile)
    _print_override_policy_summary(ptq_stage)
    print()


class QuantizationRunner:
    """Pipeline runner for model-family adapters and algorithm stages."""

    def run(self, cfg: dict[str, Any]) -> RecipeContext:
        model_cfg = cfg.get("model", {})
        if "family" not in model_cfg:
            raise KeyError("Recipe config requires model.family.")
        if "name_or_path" not in model_cfg:
            raise KeyError("Recipe config requires model.name_or_path.")

        set_seed(cfg.get("runtime", {}).get("seed", 42))
        _print_config_summary(cfg)

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
