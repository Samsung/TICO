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

"""Smoke cases for LLaMA wrapper checks."""

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.utils import (
    clone_module,
    smoke_section,
)


_LLAMA_SIZE_PROFILE_TINY = "tiny"
_LLAMA_SIZE_PROFILE_LLAMA3_2_3B_DIMS = "llama3_2_3b_dims"
_LLAMA_SIZE_PROFILE_LLAMA3_2_3B_STATIC_RUNTIME = "llama3_2_3b_static_runtime"
_LLAMA_SIZE_PROFILES = frozenset(
    {
        _LLAMA_SIZE_PROFILE_TINY,
        _LLAMA_SIZE_PROFILE_LLAMA3_2_3B_DIMS,
        _LLAMA_SIZE_PROFILE_LLAMA3_2_3B_STATIC_RUNTIME,
    }
)
_LLAMA3_2_3B_WIDTH_PROFILES = frozenset(
    {
        _LLAMA_SIZE_PROFILE_LLAMA3_2_3B_DIMS,
        _LLAMA_SIZE_PROFILE_LLAMA3_2_3B_STATIC_RUNTIME,
    }
)
_LLAMA3_2_3B_STATIC_MAX_SEQ = 2_048


@dataclass(frozen=True)
class LlamaStaticRuntimeShape:
    """Fixed input-shape contract used by Llama 3.2-3B smoke exports."""

    max_seq: int = _LLAMA3_2_3B_STATIC_MAX_SEQ

    def __post_init__(self) -> None:
        if self.max_seq < 2:
            raise ValueError(
                f"Llama static max_seq must be at least 2, got {self.max_seq}."
            )


def _has_llama() -> CaseAvailability:
    """Return availability for Hugging Face Llama modules."""
    try:
        from tico.quantization.wrapq.utils.version import has_transformers_for

        if not has_transformers_for("llama"):
            return CaseAvailability(
                False, "required transformers Llama modules are unavailable"
            )
        return CaseAvailability(True)
    except Exception as exc:
        return CaseAvailability(False, f"failed to check Llama availability: {exc}")


def _llama_options(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the Llama-specific wrapper-smoke configuration mapping."""
    section = smoke_section(cfg)
    llama_cfg = section.get("llama", {})
    if not isinstance(llama_cfg, Mapping):
        raise ValueError("debug.wrapper_smoke.llama must be a mapping.")
    return llama_cfg


def _llama_size_profile(cfg: Mapping[str, Any]) -> str:
    """Return and validate the requested Llama smoke size profile."""
    llama_cfg = _llama_options(cfg)
    profile = (
        str(llama_cfg.get("size_profile", _LLAMA_SIZE_PROFILE_TINY)).strip().lower()
    )
    if profile not in _LLAMA_SIZE_PROFILES:
        choices = ", ".join(sorted(_LLAMA_SIZE_PROFILES))
        raise ValueError(
            f"Unsupported Llama wrapper-smoke size profile '{profile}'. "
            f"Expected one of: {choices}."
        )
    return profile


def _llama_static_runtime_shape(cfg: Mapping[str, Any]) -> LlamaStaticRuntimeShape:
    """Parse and validate the Llama 3.2-3B static-runtime options."""
    llama_cfg = _llama_options(cfg)
    static_cfg = llama_cfg.get("static_runtime", {})
    if not isinstance(static_cfg, Mapping):
        raise ValueError("debug.wrapper_smoke.llama.static_runtime must be a mapping.")
    return LlamaStaticRuntimeShape(
        max_seq=int(static_cfg.get("max_seq", _LLAMA3_2_3B_STATIC_MAX_SEQ))
    )


def _build_llama_config(*, size_profile: str, max_seq: int) -> Any:
    """Create a tiny or Llama 3.2-3B-width eager config.

    The large profiles copy the target model's channel, MLP, and attention-head
    dimensions while retaining one synthetic layer. ``max_position_embeddings``
    intentionally follows the smoke/runtime capacity instead of the checkpoint's
    131,072-token training limit because the quantized wrapper materializes a
    static causal-mask template from this value.
    """
    from transformers.models.llama.configuration_llama import LlamaConfig

    if size_profile == _LLAMA_SIZE_PROFILE_TINY:
        params: dict[str, Any] = {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": max_seq,
        }
    elif size_profile in _LLAMA3_2_3B_WIDTH_PROFILES:
        params = {
            "vocab_size": 128_256,
            "hidden_size": 3_072,
            "intermediate_size": 8_192,
            "num_hidden_layers": 1,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_act": "silu",
            "max_position_embeddings": max_seq,
            "rms_norm_eps": 1e-5,
            "rope_theta": 500_000.0,
            # RoPE tables are explicit smoke inputs, so the Llama-3 scaling
            # policy is not needed here. Omitting it keeps the profile usable
            # with TICO's minimum supported Transformers versions.
            "mlp_bias": False,
        }
    else:
        raise AssertionError(f"Unhandled Llama size profile: {size_profile}")

    return LlamaConfig(
        **params,
        attention_bias=False,
        attention_dropout=0.0,
        attn_implementation="eager",
    )


def _rand_rope(
    batch_size: int, seq_len: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic Hugging Face-style RoPE tensors."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


class LlamaBaseCase(WrapperSmokeCase):
    """Base class for Llama wrapper smoke cases with size profiles."""

    tags: tuple[str, ...] = ("llama",)

    def availability(self) -> CaseAvailability:
        """Return whether this case can import Llama modules."""
        return _has_llama()

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        """Validate and cache the active Llama profile before model creation."""
        self._validated_size_profile(cfg)

    def _validated_size_profile(self, cfg: Mapping[str, Any]) -> str:
        """Return the active profile and parse its fixed shape when needed."""
        profile = _llama_size_profile(cfg)
        self._active_size_profile = profile
        self._active_static_runtime_shape = (
            _llama_static_runtime_shape(cfg)
            if profile == _LLAMA_SIZE_PROFILE_LLAMA3_2_3B_STATIC_RUNTIME
            else None
        )
        return profile

    def _static_runtime_shape(self) -> LlamaStaticRuntimeShape | None:
        """Return the active static-runtime shape after validation."""
        return getattr(self, "_active_static_runtime_shape", None)

    def _is_wide_profile(self, cfg: Mapping[str, Any]) -> bool:
        """Return whether the selected profile uses Llama 3.2-3B widths."""
        return self._validated_size_profile(cfg) in _LLAMA3_2_3B_WIDTH_PROFILES

    def _prefill_seq_len(self, default: int) -> int:
        """Return the prefill length for the active profile."""
        shape = self._static_runtime_shape()
        return shape.max_seq if shape is not None else int(default)

    def _decode_max_seq(self, default: int) -> int:
        """Return the fixed decode cache capacity for the active profile."""
        shape = self._static_runtime_shape()
        return shape.max_seq if shape is not None else int(default)

    def _batch_size(self, default: int) -> int:
        """Use batch one for real-width profiles and preserve tiny behavior."""
        profile = getattr(self, "_active_size_profile", _LLAMA_SIZE_PROFILE_TINY)
        return 1 if profile in _LLAMA3_2_3B_WIDTH_PROFILES else int(default)

    def _calibration_sample_count(self, cfg: Mapping[str, Any], *, default: int) -> int:
        """Avoid retaining multiple full static-runtime samples in memory."""
        profile = self._validated_size_profile(cfg)
        return (
            1 if profile == _LLAMA_SIZE_PROFILE_LLAMA3_2_3B_STATIC_RUNTIME else default
        )

    def _make_config(self, cfg: Mapping[str, Any], *, tiny_max_seq: int) -> Any:
        """Create a profile-aware one-layer Llama configuration."""
        profile = self._validated_size_profile(cfg)
        shape = self._static_runtime_shape()
        max_seq = shape.max_seq if shape is not None else int(tiny_max_seq)
        return _build_llama_config(size_profile=profile, max_seq=max_seq)

    def prepare_model(
        self, model: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Prepare real-width modules in place to limit peak host memory."""
        from tico.quantization import prepare

        inplace = self.inplace_prepare or self._is_wide_profile(cfg)
        return prepare(model, self.ptq_config(cfg), inplace=inplace)

    def convert_model(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Convert real-width modules in place to limit peak host memory."""
        from tico.quantization import convert

        inplace = self.inplace_convert or self._is_wide_profile(cfg)
        return convert(prepared, inplace=inplace)

    def export_filename(self, cfg: Mapping[str, Any]) -> str:
        """Include non-default profiles in generated Circle filenames."""
        profile = self._validated_size_profile(cfg)
        if profile == _LLAMA_SIZE_PROFILE_TINY:
            return super().export_filename(cfg)
        return f"{self.name}.{profile}.q.circle"


class LlamaMLPCase(LlamaBaseCase):
    """Smoke case for the LLaMA MLP wrapper."""

    name = "llama_mlp"
    description = "Quantize one LlamaMLP module with the INT16 policy."
    tags = ("llama", "mlp")
    max_mean_abs_diff = 1.0

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the INT16 PTQ config used by the LLaMA MLP smoke check."""
        from tico.quantization.config.ptq import PTQConfig
        from tico.quantization.config.specs import affine
        from tico.quantization.wrapq.dtypes import INT16
        from tico.quantization.wrapq.qscheme import QScheme

        int16_spec = affine(INT16, qscheme=QScheme.PER_TENSOR_SYMM)
        return PTQConfig(activation=int16_spec, weight=int16_spec)

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a profile-aware LlamaMLP and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        torch.manual_seed(123)
        self.config = self._make_config(cfg, tiny_max_seq=16)
        self.seq_len = self._prefill_seq_len(5)
        self.batch_size = self._batch_size(2)
        module = LlamaMLP(self.config).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one profile-aware MLP input."""
        return ForwardInput(
            (
                torch.randn(
                    self.batch_size,
                    self.seq_len,
                    self.config.hidden_size,
                ),
            )
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic MLP calibration inputs."""
        count = self._calibration_sample_count(cfg, default=4)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the MLP evaluation input."""
        return self._sample()


class LlamaAttentionPrefillCase(LlamaBaseCase):
    """Smoke case for the LLaMA attention prefill wrapper path."""

    name = "llama_attention_prefill"
    description = "Quantize one LlamaAttention module in prefill mode."
    tags = ("llama", "attention", "prefill")
    min_mean_abs_diff = 0.0
    max_mean_abs_diff = 0.8

    def __init__(self) -> None:
        """Initialize shape metadata without importing Transformers."""
        self.config = None
        self.seq_len = 6
        self.batch_size = 2

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by reference-eval Llama attention tests."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"profile": "reference_eval"})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a profile-aware LlamaAttention module and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaAttention

        torch.manual_seed(0)
        self.config = self._make_config(cfg, tiny_max_seq=16)
        self.seq_len = self._prefill_seq_len(6)
        self.batch_size = self._batch_size(2)
        module = LlamaAttention(self.config, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic prefill attention sample."""
        assert self.config is not None
        hidden = torch.randn(self.batch_size, self.seq_len, self.config.hidden_size)
        rope = _rand_rope(self.batch_size, self.seq_len, self.config.head_dim)
        mask = torch.zeros(self.batch_size, self.seq_len, self.seq_len)
        return ForwardInput((hidden, rope), {"attention_mask": mask})

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic prefill attention calibration inputs."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the synthetic prefill attention evaluation input."""
        return self._sample()

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original LlamaAttention signature for a prefill sample."""
        hidden, rope = sample.args
        mask = sample.kwargs.get("attention_mask")
        if not isinstance(mask, torch.Tensor):
            raise TypeError("Llama attention prefill requires a tensor attention mask.")
        return reference(
            hidden, position_embeddings=rope, attention_mask=mask.unsqueeze(1)
        )[0]

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped attention module in prefill mode when available."""
        wrapped = getattr(quantized, "wrapped", quantized)
        return (
            wrapped.as_export_module("prefill").eval()
            if hasattr(wrapped, "as_export_module")
            else quantized
        )

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create fixed prefill export inputs without an attention mask."""
        assert self.config is not None
        export_seq_len = int(self.config.max_position_embeddings)
        hidden = torch.randn(1, export_seq_len, self.config.hidden_size)
        rope = _rand_rope(1, export_seq_len, self.config.head_dim)
        return ForwardInput((hidden, rope))


class LlamaAttentionDecodeCase(LlamaBaseCase):
    """Smoke case for the LLaMA attention decode wrapper path."""

    name = "llama_attention_decode"
    description = "Quantize one LlamaAttention module in static decode mode."
    tags = ("llama", "attention", "decode")
    compare_reference_source = "prepared"
    min_mean_abs_diff = 0.0
    max_mean_abs_diff = 2.0

    def __init__(self) -> None:
        """Initialize static decode shape metadata."""
        self.max_seq = 16
        self.config = None

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a profile-aware LlamaAttention module and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaAttention

        torch.manual_seed(123)
        self.max_seq = self._decode_max_seq(16)
        self.config = self._make_config(cfg, tiny_max_seq=self.max_seq)
        module = LlamaAttention(self.config, layer_idx=0).eval()
        return module, clone_module(module)

    def _decode_sample(self, prepared: torch.nn.Module | None = None) -> ForwardInput:
        """Create one static decode input sample."""
        assert self.config is not None
        hidden = torch.randn(1, 1, self.config.hidden_size)
        cos = torch.randn(1, 1, self.config.head_dim)
        sin = torch.randn(1, 1, self.config.head_dim)
        wrapped = getattr(prepared, "wrapped", None)
        attn_options = getattr(wrapped, "attn_options", None)
        if getattr(attn_options, "rope", None) == "pre_negated_sin":
            sin = sin.clone()
            sin[..., : self.config.head_dim // 2] = -sin[
                ..., : self.config.head_dim // 2
            ]
        mask = torch.zeros(1, 1, self.max_seq)
        past = (
            torch.randn(
                1,
                self.config.num_key_value_heads,
                self.max_seq - 1,
                self.config.head_dim,
            ),
            torch.randn(
                1,
                self.config.num_key_value_heads,
                self.max_seq - 1,
                self.config.head_dim,
            ),
        )
        return ForwardInput(
            (hidden, (cos, sin)),
            {"attention_mask": mask, "past_key_value": past, "use_cache": True},
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create static decode calibration samples."""
        count = self._calibration_sample_count(cfg, default=4)
        return [self._decode_sample(prepared) for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the static decode evaluation sample."""
        return self._decode_sample(prepared)

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped attention module in decode mode when available."""
        wrapped = getattr(quantized, "wrapped", quantized)
        return (
            wrapped.as_export_module("decode").eval()
            if hasattr(wrapped, "as_export_module")
            else quantized
        )

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create positional static decode inputs expected by the export adapter."""
        hidden, pos = eval_sample.args
        mask = eval_sample.kwargs["attention_mask"]
        past = eval_sample.kwargs["past_key_value"]
        return ForwardInput((hidden, pos, mask, past))


class LlamaDecoderLayerPrefillCase(LlamaBaseCase):
    """Smoke case for the LLaMA decoder-layer prefill wrapper path."""

    name = "llama_decoder_layer_prefill"
    description = "Quantize one LlamaDecoderLayer module in prefill mode."
    tags = ("llama", "decoder_layer", "prefill")
    max_mean_abs_diff = 1.2

    def __init__(self) -> None:
        """Initialize prefill shape metadata."""
        self.max_seq = 16
        self.batch_size = 2
        self.config = None

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the reference-eval PTQ config used by the decoder prefill test."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"profile": "reference_eval"})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a profile-aware LlamaDecoderLayer and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        torch.manual_seed(0)
        self.max_seq = self._prefill_seq_len(16)
        self.batch_size = self._batch_size(2)
        self.config = self._make_config(cfg, tiny_max_seq=self.max_seq)
        module = LlamaDecoderLayer(self.config, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic decoder-layer prefill sample."""
        assert self.config is not None
        hidden = torch.randn(self.batch_size, self.max_seq, self.config.hidden_size)
        rope = _rand_rope(self.batch_size, self.max_seq, self.config.head_dim)
        mask = torch.ones(self.batch_size, self.max_seq, self.max_seq, dtype=torch.bool)
        return ForwardInput(
            (hidden,), {"attention_mask": mask, "position_embeddings": rope}
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create decoder-layer prefill calibration samples."""
        count = self._calibration_sample_count(cfg, default=3)
        return [self._sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the decoder-layer prefill evaluation sample."""
        return self._sample()

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original LlamaDecoderLayer signature for prefill."""
        hidden = sample.args[0]
        mask = sample.kwargs["attention_mask"]
        rope = sample.kwargs["position_embeddings"]
        out = reference(
            hidden, attention_mask=mask.unsqueeze(1), position_embeddings=rope
        )
        return out[0] if isinstance(out, tuple) else out

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static decoder-layer prefill export input."""
        assert self.config is not None
        return ForwardInput((torch.randn(1, self.max_seq, self.config.hidden_size),))


class LlamaDecoderLayerDecodeCase(LlamaBaseCase):
    """Smoke case for the LLaMA decoder-layer decode wrapper path."""

    name = "llama_decoder_layer_decode"
    description = "Quantize one LlamaDecoderLayer module in static decode mode."
    tags = ("llama", "decoder_layer", "decode")
    compare_reference_source = "prepared"
    max_mean_abs_diff = 2.0

    def __init__(self) -> None:
        """Initialize static decode shape metadata."""
        self.max_seq = 16
        self.config = None

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a profile-aware LlamaDecoderLayer and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        torch.manual_seed(123)
        self.max_seq = self._decode_max_seq(16)
        self.config = self._make_config(cfg, tiny_max_seq=self.max_seq)
        module = LlamaDecoderLayer(self.config, layer_idx=0).eval()
        return module, clone_module(module)

    def after_prepare(self, prepared: torch.nn.Module, cfg: Mapping[str, Any]) -> None:
        """Force tuple return so hidden and cache deltas are available."""
        if hasattr(prepared, "wrapped"):
            prepared.wrapped.return_type = "tuple"

    def _decode_sample(self) -> ForwardInput:
        """Create one static decode input sample for a decoder layer."""
        assert self.config is not None
        hidden = torch.randn(1, 1, self.config.hidden_size)
        pos = (
            torch.randn(1, 1, self.config.head_dim),
            torch.randn(1, 1, self.config.head_dim),
        )
        mask = torch.zeros(1, 1, self.max_seq)
        past = (
            torch.randn(
                1,
                self.config.num_key_value_heads,
                self.max_seq - 1,
                self.config.head_dim,
            ),
            torch.randn(
                1,
                self.config.num_key_value_heads,
                self.max_seq - 1,
                self.config.head_dim,
            ),
        )
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "attention_mask": mask,
                "past_key_value": past,
                "position_embeddings": pos,
                "use_cache": True,
            },
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create decoder-layer decode calibration samples."""
        count = self._calibration_sample_count(cfg, default=4)
        return [self._decode_sample() for _ in range(count)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the decoder-layer decode evaluation sample."""
        return self._decode_sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the decoder layer in decode mode when supported."""
        return (
            quantized.as_export_module("decode").eval()
            if hasattr(quantized, "as_export_module")
            else quantized
        )

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static decode inputs expected by the decoder-layer export adapter."""
        hidden = eval_sample.kwargs["hidden_states"]
        mask = eval_sample.kwargs["attention_mask"]
        past = eval_sample.kwargs["past_key_value"]
        pos = eval_sample.kwargs["position_embeddings"]
        return ForwardInput(
            (hidden, mask), {"past_key_value": past, "position_embeddings": pos}
        )


LLAMA_CASES: tuple[WrapperSmokeCase, ...] = (
    LlamaMLPCase(),
    LlamaAttentionPrefillCase(),
    LlamaAttentionDecodeCase(),
    LlamaDecoderLayerPrefillCase(),
    LlamaDecoderLayerDecodeCase(),
)
