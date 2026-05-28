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

from typing import Any, Mapping

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.utils import clone_module


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


def _make_llama_config(max_seq: int = 16):
    """Create a tiny eager Llama config for synthetic smoke tests."""
    from transformers.models.llama.configuration_llama import LlamaConfig

    return LlamaConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_bias=False,
        attention_dropout=0.0,
        attn_implementation="eager",
        max_position_embeddings=max_seq,
    )


def _rand_rope(
    batch_size: int, seq_len: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic Hugging Face-style RoPE tensors."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


class LlamaMLPCase(WrapperSmokeCase):
    """Smoke case for the LLaMA MLP wrapper."""

    name = "llama_mlp"
    description = "Quantize one tiny LlamaMLP module with the INT16 policy."
    tags = ("llama", "mlp")
    max_mean_abs_diff = 1.0

    def availability(self) -> CaseAvailability:
        """Return whether this case can import Llama modules."""
        return _has_llama()

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the INT16 PTQ config used by the LLaMA MLP smoke check."""
        from tico.quantization.config.ptq import PTQConfig
        from tico.quantization.wrapq.dtypes import INT16
        from tico.quantization.wrapq.qscheme import QScheme

        return PTQConfig(default_dtype=INT16, default_qscheme=QScheme.PER_TENSOR_SYMM)

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny LlamaMLP and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaMLP

        torch.manual_seed(123)
        config = _make_llama_config()
        module = LlamaMLP(config).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic MLP calibration inputs."""
        return [ForwardInput((torch.randn(2, 5, 16),)) for _ in range(4)]


class LlamaAttentionPrefillCase(WrapperSmokeCase):
    """Smoke case for the LLaMA attention prefill wrapper path."""

    name = "llama_attention_prefill"
    description = "Quantize one tiny LlamaAttention module in prefill mode."
    tags = ("llama", "attention", "prefill")
    min_mean_abs_diff = 0.0
    max_mean_abs_diff = 0.8

    def __init__(self) -> None:
        """Initialize the case with deterministic shape metadata."""
        self.config = _make_llama_config(max_seq=16) if _has_llama().available else None

    def availability(self) -> CaseAvailability:
        """Return whether this case can import Llama modules."""
        return _has_llama()

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by reference-eval Llama attention tests."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"profile": "reference_eval"})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny LlamaAttention module and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaAttention

        torch.manual_seed(0)
        self.config = _make_llama_config(max_seq=16)
        module = LlamaAttention(self.config, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self, batch_size: int = 2, seq_len: int = 6) -> ForwardInput:
        """Create one synthetic prefill attention sample."""
        assert self.config is not None
        hidden = torch.randn(batch_size, seq_len, self.config.hidden_size)
        rope = _rand_rope(batch_size, seq_len, self.config.head_dim)
        mask = torch.zeros(batch_size, seq_len, seq_len)
        return ForwardInput((hidden, rope), {"attention_mask": mask})

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create synthetic prefill attention calibration inputs."""
        return [self._sample() for _ in range(3)]

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
        """Create static prefill export inputs without an attention mask."""
        assert self.config is not None
        hidden = torch.randn(
            1, self.config.max_position_embeddings, self.config.hidden_size
        )
        rope = _rand_rope(1, self.config.max_position_embeddings, self.config.head_dim)
        return ForwardInput((hidden, rope))


class LlamaAttentionDecodeCase(WrapperSmokeCase):
    """Smoke case for the LLaMA attention decode wrapper path."""

    name = "llama_attention_decode"
    description = "Quantize one tiny LlamaAttention module in static decode mode."
    tags = ("llama", "attention", "decode")
    compare_reference_source = "prepared"
    min_mean_abs_diff = 0.0
    max_mean_abs_diff = 2.0

    def __init__(self) -> None:
        """Initialize static decode shape metadata."""
        self.max_seq = 16
        self.config = (
            _make_llama_config(max_seq=self.max_seq) if _has_llama().available else None
        )

    def availability(self) -> CaseAvailability:
        """Return whether this case can import Llama modules."""
        return _has_llama()

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny LlamaAttention module and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaAttention

        torch.manual_seed(123)
        self.config = _make_llama_config(max_seq=self.max_seq)
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
        return [self._decode_sample(prepared) for _ in range(4)]

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


class LlamaDecoderLayerPrefillCase(WrapperSmokeCase):
    """Smoke case for the LLaMA decoder-layer prefill wrapper path."""

    name = "llama_decoder_layer_prefill"
    description = "Quantize one tiny LlamaDecoderLayer module in prefill mode."
    tags = ("llama", "decoder_layer", "prefill")
    max_mean_abs_diff = 1.2

    def __init__(self) -> None:
        """Initialize prefill shape metadata."""
        self.max_seq = 16
        self.config = (
            _make_llama_config(max_seq=self.max_seq) if _has_llama().available else None
        )

    def availability(self) -> CaseAvailability:
        """Return whether this case can import Llama modules."""
        return _has_llama()

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the reference-eval PTQ config used by the decoder prefill test."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"profile": "reference_eval"})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny LlamaDecoderLayer and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        torch.manual_seed(0)
        self.config = _make_llama_config(max_seq=self.max_seq)
        module = LlamaDecoderLayer(self.config, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic decoder-layer prefill sample."""
        assert self.config is not None
        hidden = torch.randn(2, self.max_seq, self.config.hidden_size)
        rope = _rand_rope(2, self.max_seq, self.config.head_dim)
        mask = torch.ones(2, self.max_seq, self.max_seq, dtype=torch.bool)
        return ForwardInput(
            (hidden,), {"attention_mask": mask, "position_embeddings": rope}
        )

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create decoder-layer prefill calibration samples."""
        return [self._sample() for _ in range(3)]

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


class LlamaDecoderLayerDecodeCase(WrapperSmokeCase):
    """Smoke case for the LLaMA decoder-layer decode wrapper path."""

    name = "llama_decoder_layer_decode"
    description = "Quantize one tiny LlamaDecoderLayer module in static decode mode."
    tags = ("llama", "decoder_layer", "decode")
    compare_reference_source = "prepared"
    max_mean_abs_diff = 2.0

    def __init__(self) -> None:
        """Initialize static decode shape metadata."""
        self.max_seq = 16
        self.config = (
            _make_llama_config(max_seq=self.max_seq) if _has_llama().available else None
        )

    def availability(self) -> CaseAvailability:
        """Return whether this case can import Llama modules."""
        return _has_llama()

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny LlamaDecoderLayer and reference copy."""
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        torch.manual_seed(123)
        self.config = _make_llama_config(max_seq=self.max_seq)
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
        return [self._decode_sample() for _ in range(4)]

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
