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
AttentionLayout = Literal["batched", "unrolled"]

DEFAULT_EXECUTION_PROFILE: ExecutionProfile = "npu_export"
SUPPORTED_EXECUTION_PROFILES: tuple[ExecutionProfile, ...] = (
    "reference_eval",
    "npu_export",
)


@dataclass(frozen=True)
class Gemma4TextAttentionOptions:
    """Execution options for quantized Gemma4 text attention.

    These options describe graph-level implementation choices rather than
    quantization policy. They are resolved from ``PTQConfig.model_args`` so the
    same quantization overrides can be used with either a reference-friendly or
    an NPU-export-friendly attention graph.

    Attributes
    ----------
    layout : AttentionLayout
        ``"batched"`` uses a Hugging Face-like batched GQA graph and is useful
        for reference evaluation. ``"unrolled"`` emits one rank-3 attention
        matmul per query head and avoids KV-head broadcasting for NPU export.
    """

    layout: AttentionLayout = "unrolled"


_PRESETS: dict[ExecutionProfile, Gemma4TextAttentionOptions] = {
    "reference_eval": Gemma4TextAttentionOptions(layout="batched"),
    "npu_export": Gemma4TextAttentionOptions(layout="unrolled"),
}


def normalize_execution_profile(
    profile: Any,
    *,
    context: str = "profile",
) -> ExecutionProfile:
    """Validate and return a Gemma4 execution profile string.

    Parameters
    ----------
    profile : Any
        User-provided profile value.
    context : str
        Human-readable configuration location used in error messages.

    Returns
    -------
    ExecutionProfile
        The validated execution profile.

    Raises
    ------
    TypeError
        If ``profile`` is not a string.
    ValueError
        If the string is not a supported profile.
    """
    if not isinstance(profile, str):
        raise TypeError(f"{context} must be a string, got {type(profile).__name__}.")
    if profile not in SUPPORTED_EXECUTION_PROFILES:
        raise ValueError(
            f"Unsupported execution profile at {context}: {profile!r}. "
            f"Supported profiles: {list(SUPPORTED_EXECUTION_PROFILES)}."
        )
    return cast(ExecutionProfile, profile)


def get_gemma4_text_attention_options(
    qcfg: Optional[PTQConfig],
) -> Gemma4TextAttentionOptions:
    """Resolve Gemma4 text-attention options from a PTQ configuration.

    The root-level ``model_args["profile"]`` selects the default execution
    profile. ``model_args["attention"]`` may override that profile or individual
    attention fields. The attention override accepts either a profile string or
    a mapping.

    Examples
    --------
    Select the reference graph for the whole model::

        PTQConfig(model_args={"profile": "reference_eval"})

    Override only the attention layout::

        PTQConfig(
            model_args={
                "profile": "reference_eval",
                "attention": {"layout": "unrolled"},
            }
        )

    Parameters
    ----------
    qcfg : Optional[PTQConfig]
        PTQ configuration associated with the wrapper.

    Returns
    -------
    Gemma4TextAttentionOptions
        Fully resolved and validated attention options.
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

    valid_keys = {field.name for field in fields(Gemma4TextAttentionOptions)}
    unknown_keys = sorted(set(raw) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unknown Gemma4 text attention option(s): {unknown_keys}.")

    options = replace(_PRESETS[profile], **raw)
    _validate_gemma4_text_attention_options(options)
    return options


def is_npu_export_text_attention_options(
    options: Gemma4TextAttentionOptions,
) -> bool:
    """Return whether the options match the NPU-export attention graph."""
    return options.layout == "unrolled"


def _validate_gemma4_text_attention_options(
    options: Gemma4TextAttentionOptions,
) -> None:
    """Validate a fully resolved Gemma4 text-attention option set."""
    if options.layout not in ("batched", "unrolled"):
        raise ValueError(f"Unsupported attention layout: {options.layout!r}.")
