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

from dataclasses import dataclass, fields, replace
from typing import Any, cast, Literal, Mapping, Optional

from tico.quantization.config.ptq import PTQConfig


ExecutionProfile = Literal["reference_eval", "npu_export"]
ScaleFusion = Literal["none", "q_proj", "k_proj"]
RopeConvention = Literal["hf", "pre_negated_sin"]
AttentionLayout = Literal["batched", "unrolled"]

DEFAULT_EXECUTION_PROFILE: ExecutionProfile = "npu_export"
SUPPORTED_EXECUTION_PROFILES: tuple[ExecutionProfile, ...] = (
    "reference_eval",
    "npu_export",
)


@dataclass(frozen=True)
class LlamaAttentionOptions:
    """
    Execution options for quantized Llama attention wrappers.

    These options describe graph-level implementation choices, not quantization
    policy. They are intentionally read from `PTQConfig.model_args` instead of
    `PTQConfig.overrides`.

    Attributes
    ----------
    scale_fusion : ScaleFusion
        Where to apply the attention scale `1 / sqrt(head_dim)`.
        "none" applies it to logits at runtime, while "q_proj" and
        "k_proj" fold it into the corresponding projection weights.
    rope : RopeConvention
        Rotary embedding sign convention. "hf" uses `rotate_half` as
        `(-x2, x1)` with normal sine values. "pre_negated_sin" expects the
        first half of sine values to be pre-negated and uses `(x2, x1)` in the
        rotate-half operation.
    layout : AttentionLayout
        Attention implementation layout. "batched" is closer to the
        Hugging Face implementation and is preferable for GPU evaluation.
        "unrolled" preserves the NPU-export-friendly per-head loop.
    """

    scale_fusion: ScaleFusion = "k_proj"
    rope: RopeConvention = "pre_negated_sin"
    layout: AttentionLayout = "unrolled"


_PRESETS: dict[ExecutionProfile, LlamaAttentionOptions] = {
    "reference_eval": LlamaAttentionOptions(
        scale_fusion="none",
        rope="hf",
        layout="batched",
    ),
    "npu_export": LlamaAttentionOptions(
        scale_fusion="k_proj",
        rope="pre_negated_sin",
        layout="unrolled",
    ),
}


def normalize_execution_profile(
    profile: Any,
    *,
    context: str = "profile",
) -> ExecutionProfile:
    """
    Validate and return an execution profile string.

    Parameters
    ----------
    profile : Any
        User-provided profile value.
    context : str
        Human-readable location used in error messages.

    Returns
    -------
    ExecutionProfile
        Validated profile value.

    Raises
    ------
    TypeError
        If the profile value is not a string.
    ValueError
        If the profile string is not supported.
    """
    if not isinstance(profile, str):
        raise TypeError(f"{context} must be a string, got {type(profile).__name__}.")
    if profile not in SUPPORTED_EXECUTION_PROFILES:
        raise ValueError(
            f"Unsupported execution profile at {context}: {profile!r}. "
            f"Supported profiles: {list(SUPPORTED_EXECUTION_PROFILES)}."
        )
    return cast(ExecutionProfile, profile)


def get_llama_attention_options(
    qcfg: Optional[PTQConfig],
) -> LlamaAttentionOptions:
    """
    Resolve Llama attention implementation options from a PTQConfig.

    The root-level `model_args["profile"]` selects the default execution
    profile for all profile-aware wrappers. The attention wrapper may override
    that default through `model_args["attention"]`.

    Supported examples are::

        PTQConfig(..., model_args={"profile": "reference_eval"})

    and::

        PTQConfig(
            ...,
            model_args={
                "profile": "reference_eval",
                "attention": {
                    "layout": "unrolled",
                },
            },
        )

    `model_args["attention"]` may also be a plain profile string, for example
    "npu_export". When no option is provided, the default profile is
    "npu_export" to preserve the existing export-oriented graph.

    Parameters
    ----------
    qcfg : Optional[PTQConfig]
        PTQ configuration associated with the wrapper.

    Returns
    -------
    LlamaAttentionOptions
        Validated execution options.
    """
    if qcfg is None:
        return _PRESETS[DEFAULT_EXECUTION_PROFILE]

    root_profile = normalize_execution_profile(
        qcfg.get_model_arg("profile", DEFAULT_EXECUTION_PROFILE),
        context="PTQConfig.model_args['profile']",
    )

    raw_attention = qcfg.get_model_arg("attention", {})
    if raw_attention is None:
        raw_attention = {}
    if isinstance(raw_attention, str):
        raw_attention = {"profile": raw_attention}
    if not isinstance(raw_attention, Mapping):
        raise TypeError(
            "PTQConfig.model_args['attention'] must be a mapping, a string, or None."
        )

    raw = dict(raw_attention)
    profile = normalize_execution_profile(
        raw.pop("profile", root_profile),
        context="PTQConfig.model_args['attention']['profile']",
    )

    valid_keys = {field.name for field in fields(LlamaAttentionOptions)}
    unknown_keys = sorted(set(raw) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown Llama attention option(s): {unknown_keys}.")

    options = replace(_PRESETS[profile], **raw)
    _validate_llama_attention_options(options)
    return options


def is_npu_export_attention_options(options: LlamaAttentionOptions) -> bool:
    """
    Return whether the options match the NPU-export-friendly attention graph.
    """
    return (
        options.scale_fusion == "k_proj"
        and options.rope == "pre_negated_sin"
        and options.layout == "unrolled"
    )


def _validate_llama_attention_options(options: LlamaAttentionOptions) -> None:
    """
    Validate a fully resolved LlamaAttentionOptions instance.
    """
    if options.scale_fusion not in ("none", "q_proj", "k_proj"):
        raise ValueError(f"Unsupported scale_fusion: {options.scale_fusion!r}.")
    if options.rope not in ("hf", "pre_negated_sin"):
        raise ValueError(f"Unsupported rope convention: {options.rope!r}.")
    if options.layout not in ("batched", "unrolled"):
        raise ValueError(f"Unsupported attention layout: {options.layout!r}.")
