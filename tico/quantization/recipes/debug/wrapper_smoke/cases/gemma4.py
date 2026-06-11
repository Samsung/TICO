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

"""Smoke cases for Gemma4 wrapper checks."""

from typing import Any, Mapping

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import (
    CaseAvailability,
    ForwardInput,
    WrapperSmokeCase,
)
from tico.quantization.recipes.debug.wrapper_smoke.utils import clone_module


_GEMMA4_FULL_ROPE_PARAMETERS: dict[str, Any] = {
    "rope_type": "proportional",
    "partial_rotary_factor": 0.25,
    "rope_theta": 1_000_000.0,
}
_GEMMA4_SLIDING_ROPE_PARAMETERS: dict[str, Any] = {
    "rope_type": "default",
    "rope_theta": 10_000.0,
}


def _has_gemma4() -> CaseAvailability:
    """Return availability for Hugging Face Gemma4 modules."""
    try:
        from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
            Gemma4TextConfig,
        )

        return CaseAvailability(True)
    except Exception as exc:
        return CaseAvailability(False, f"Gemma4 modules are unavailable: {exc}")


def _set_eager_attention(cfg: Any) -> Any:
    """Set eager attention on configs that expose a configurable implementation."""
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = "eager"
    else:
        setattr(cfg, "_attn_implementation", "eager")
    return cfg


def _rope_parameters_for_layer_types(
    layer_types: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """Return RoPE parameters whose keys exactly match the requested layer types.

    Hugging Face validates Gemma4 RoPE parameters as a nested layer-type mapping
    only when every top-level RoPE key is present in ``config.layer_types``. Tiny
    smoke configs often use a subset of real Gemma4 layer types, so the default
    Gemma4 RoPE dict can trigger warnings when it contains unused keys.
    """
    rope_parameters: dict[str, dict[str, Any]] = {}
    if "sliding_attention" in layer_types:
        rope_parameters["sliding_attention"] = dict(_GEMMA4_SLIDING_ROPE_PARAMETERS)
    if "full_attention" in layer_types:
        rope_parameters["full_attention"] = dict(_GEMMA4_FULL_ROPE_PARAMETERS)
    return rope_parameters


def _make_text_config(
    *,
    layer_types: tuple[str, ...] = ("full_attention",),
    attention_k_eq_v: bool = False,
    num_kv_shared_layers: int = 0,
) -> Any:
    """Create a warning-free tiny Gemma4 text config for synthetic smoke tests.

    The helper intentionally provides ``layer_types`` and ``rope_parameters`` as
    a matched pair. This prevents Hugging Face from treating nested Gemma4 RoPE
    parameters as one global default-RoPE config, which otherwise emits
    ``Unrecognized keys in rope_parameters`` warnings in one-layer smoke cases.
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        max_position_embeddings=128,
        layer_types=list(layer_types),
        rope_parameters=_rope_parameters_for_layer_types(layer_types),
        attention_bias=False,
        attention_dropout=0.0,
        use_cache=False,
        enable_moe_block=False,
        attention_k_eq_v=attention_k_eq_v,
        num_kv_shared_layers=num_kv_shared_layers,
    )
    return _set_eager_attention(cfg)


def _text_rope(
    batch_size: int, seq_len: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic Gemma4 text RoPE embeddings."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


def _attention_mask(seq_len: int, kv_len: int | None = None) -> torch.Tensor:
    """Create an additive attention mask for synthetic Gemma4 attention tests."""
    kv_len = seq_len if kv_len is None else kv_len
    return torch.zeros(1, 1, seq_len, kv_len)


def _clone_value(value: Any) -> Any:
    """Clone tensors nested inside a small smoke-test value."""
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return value


def _clone_forward_input(sample: ForwardInput) -> ForwardInput:
    """Clone a smoke input so reference and quantized runs do not share mutable state."""
    return ForwardInput(
        tuple(_clone_value(arg) for arg in sample.args),
        {key: _clone_value(value) for key, value in sample.kwargs.items()},
    )


class Gemma4BaseCase(WrapperSmokeCase):
    """Base class for Gemma4 E2B wrapper smoke cases."""

    tags: tuple[str, ...] = ("gemma4", "e2b")

    def availability(self) -> CaseAvailability:
        """Return whether Gemma4 modules can be imported."""
        return _has_gemma4()


class Gemma4TextMLPCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 text MLP."""

    name = "gemma4_text_mlp"
    description = "Quantize one tiny dense Gemma4 text MLP module."
    tags = ("gemma4", "e2b", "text", "mlp")

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 text MLP and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

        torch.manual_seed(123)
        self.text_cfg = _make_text_config(layer_types=("full_attention",))
        module = Gemma4TextMLP(self.text_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create calibration samples."""
        return [
            ForwardInput((torch.randn(1, 8, self.text_cfg.hidden_size),))
            for _ in range(3)
        ]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create an evaluation sample."""
        return ForwardInput((torch.randn(1, 8, self.text_cfg.hidden_size),))


class Gemma4TextAttentionBaseCase(Gemma4BaseCase):
    """Base class for tiny Gemma4 text attention smoke cases."""

    tags = ("gemma4", "e2b", "text", "attention")
    max_mean_abs_diff = 2.0
    seq_len = 8
    layer_idx = 0
    layer_types: tuple[str, ...] = ("full_attention",)
    attention_k_eq_v = False
    num_kv_shared_layers = 0

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 text attention module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

        torch.manual_seed(123)
        self.text_cfg = _make_text_config(
            layer_types=self.layer_types,
            attention_k_eq_v=self.attention_k_eq_v,
            num_kv_shared_layers=self.num_kv_shared_layers,
        )
        module = Gemma4TextAttention(self.text_cfg, layer_idx=self.layer_idx).eval()
        return module, clone_module(module)

    def _base_kwargs(self) -> dict[str, Any]:
        """Create keyword arguments shared by non-shared attention samples."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        return {
            "hidden_states": hidden,
            "position_embeddings": _text_rope(1, self.seq_len, self.text_cfg.head_dim),
            "attention_mask": _attention_mask(self.seq_len),
            "shared_kv_states": {},
        }

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 text attention input."""
        return ForwardInput((), self._base_kwargs())

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run a Gemma4 attention wrapper without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        return module(*cloned.args, **dict(cloned.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4 attention module without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        return reference(*cloned.args, **dict(cloned.kwargs))

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create Gemma4 text attention calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the Gemma4 text attention evaluation sample."""
        return self._sample()


class Gemma4TextAttentionCase(Gemma4TextAttentionBaseCase):
    """Smoke case for one tiny full-attention Gemma4 text attention module."""

    name = "gemma4_text_attention"
    description = "Quantize one tiny full-attention Gemma4 text attention module."
    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 1


class Gemma4TextSlidingAttentionCase(Gemma4TextAttentionBaseCase):
    """Smoke case for one tiny sliding Gemma4 text attention module."""

    name = "gemma4_text_attention_sliding"
    description = "Quantize one tiny sliding Gemma4 text attention module."
    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 0


class Gemma4TextAttentionKEqVCase(Gemma4TextAttentionBaseCase):
    """Smoke case for Gemma4 full attention with K-equals-V alternative attention."""

    name = "gemma4_text_attention_k_eq_v"
    description = (
        "Quantize one tiny Gemma4 text attention module with attention_k_eq_v=True."
    )
    layer_types = ("full_attention",)
    layer_idx = 0
    attention_k_eq_v = True


class Gemma4TextAttentionSharedKVCase(Gemma4TextAttentionBaseCase):
    """Smoke case for a Gemma4 shared-KV consumer attention layer."""

    name = "gemma4_text_attention_shared_kv"
    description = (
        "Quantize one tiny Gemma4 text attention module that consumes shared KV states."
    )
    layer_types = ("full_attention", "full_attention")
    layer_idx = 1
    num_kv_shared_layers = 1

    def _sample(self) -> ForwardInput:
        """Create one synthetic shared-KV attention input."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        key_states = torch.randn(
            1,
            self.text_cfg.num_key_value_heads,
            self.seq_len,
            self.text_cfg.head_dim,
        )
        value_states = torch.randn_like(key_states)
        shared_key_value = (key_states, value_states)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _text_rope(
                    1, self.seq_len, self.text_cfg.head_dim
                ),
                "attention_mask": _attention_mask(self.seq_len),
                "shared_kv_states": {"full_attention": shared_key_value},
                # QuantGemma4TextAttention implementations may also accept this
                # explicit tuple form for static-runtime export paths.
                "shared_key_value": shared_key_value,
            },
        )


GEMMA4_CASES = (
    Gemma4TextMLPCase(),
    Gemma4TextAttentionCase(),
    Gemma4TextSlidingAttentionCase(),
    Gemma4TextAttentionKEqVCase(),
    Gemma4TextAttentionSharedKVCase(),
)
