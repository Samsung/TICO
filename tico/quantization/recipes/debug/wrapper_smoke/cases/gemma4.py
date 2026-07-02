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
    hidden_size_per_layer_input: int = 0,
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
        hidden_size_per_layer_input=hidden_size_per_layer_input,
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


def _causal_mask(seq_len: int, fill_value: float = -120.0) -> torch.Tensor:
    """Create an additive causal mask with a large negative upper triangle."""
    mask = torch.zeros(1, 1, seq_len, seq_len)
    blocked = torch.full_like(mask, float(fill_value))
    return torch.triu(blocked, diagonal=1)


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


def _sliding_window_causal_mask(
    seq_len: int,
    sliding_window: int,
    *,
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    fill_value: float = -120.0,
) -> torch.Tensor:
    """Create a fixed-shape additive causal sliding-window mask.

    A query at position ``q`` can attend to keys in the inclusive interval
    ``[max(0, q - sliding_window + 1), q]``. Future keys and keys older than
    the configured window receive ``fill_value``.

    Parameters
    ----------
    seq_len:
        Static query and key/value sequence length.
    sliding_window:
        Number of visible tokens including the current query token.
    batch_size:
        Static batch size represented by the returned mask.
    dtype:
        Floating-point dtype of the additive mask.
    device:
        Device on which to create the mask.
    fill_value:
        Additive value assigned to blocked positions.

    Returns
    -------
    torch.Tensor
        A tensor with shape ``(batch_size, 1, seq_len, seq_len)``.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}.")
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}.")

    query_positions = torch.arange(seq_len, device=device).view(seq_len, 1)
    key_positions = torch.arange(seq_len, device=device).view(1, seq_len)

    future_positions = key_positions > query_positions
    positions_before_window = key_positions < query_positions - sliding_window + 1
    blocked_positions = future_positions | positions_before_window

    mask = torch.zeros((seq_len, seq_len), dtype=dtype, device=device)
    mask.masked_fill_(blocked_positions, float(fill_value))
    return (
        mask.view(1, 1, seq_len, seq_len)
        .expand(batch_size, 1, seq_len, seq_len)
        .contiguous()
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

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Return export input without shared_kv_states.

        shared_kv_states is a mutable dict used to pass KV tensors between
        layers at runtime. torch.export strict mode fails when an empty dict
        is included in kwargs because pytree traversal produces a tensor-count
        mismatch. The export path does not need this side-channel.
        """
        kwargs = {
            k: v for k, v in eval_sample.kwargs.items() if k != "shared_kv_states"
        }
        return ForwardInput(eval_sample.args, kwargs)


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


class Gemma4TextDecoderLayerBaseCase(Gemma4BaseCase):
    """Base class for tiny Gemma4 text decoder-layer smoke cases."""

    tags: tuple[str, ...] = ("gemma4", "e2b", "text", "decoder_layer")
    max_mean_abs_diff = 2.5
    seq_len = 8
    layer_idx = 0
    layer_types: tuple[str, ...] = ("full_attention",)
    attention_k_eq_v = False
    num_kv_shared_layers = 0
    export_mode = "prefill"
    return_kv_on_export = True
    compare_reference_source = "reference"

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4 decoder-layer smoke checks."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"profile": "reference_eval"})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny dense Gemma4 text decoder layer and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        torch.manual_seed(123)
        self.text_cfg = _make_text_config(
            layer_types=self.layer_types,
            attention_k_eq_v=self.attention_k_eq_v,
            num_kv_shared_layers=self.num_kv_shared_layers,
        )
        module = Gemma4TextDecoderLayer(self.text_cfg, layer_idx=self.layer_idx).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic prefill decoder-layer sample."""
        hidden = torch.randn(1, self.seq_len, self.text_cfg.hidden_size)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _text_rope(
                    1, self.seq_len, self.text_cfg.head_dim
                ),
                "attention_mask": _causal_mask(
                    self.seq_len,
                    fill_value=float(self.ptq_config({}).attention_mask_fill_value),
                ),
                "shared_kv_states": {},
            },
        )

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run a Gemma4 decoder layer without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        return module(*cloned.args, **dict(cloned.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4 decoder layer without wrapper-only kwargs."""
        cloned = _clone_forward_input(sample)
        kwargs = dict(cloned.kwargs)
        kwargs.pop("shared_key_value", None)
        output = reference(*cloned.args, **kwargs)
        return output[0] if isinstance(output, tuple) else output

    def calibration_inputs(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> list[ForwardInput]:
        """Create decoder-layer calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self, prepared: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create the decoder-layer evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped decoder layer in the configured static mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        return (
            wrapped.as_export_module(
                self.export_mode, return_kv=self.return_kv_on_export
            ).eval()
            if hasattr(wrapped, "as_export_module")
            else quantized
        )

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the decoder-layer adapter."""
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        hidden = kwargs["hidden_states"]
        mask = kwargs["attention_mask"]
        rope = kwargs["position_embeddings"]
        shared_key_value = kwargs.get("shared_key_value")
        export_kwargs = {}
        if shared_key_value is not None:
            export_kwargs["shared_key_value"] = shared_key_value
        return ForwardInput((hidden, mask, rope), export_kwargs)


class Gemma4TextDecoderLayerPrefillCase(Gemma4TextDecoderLayerBaseCase):
    """Smoke case for one tiny Gemma4 text decoder layer in prefill mode."""

    name = "gemma4_text_decoder_layer_prefill"
    description = "Quantize one tiny dense Gemma4 text decoder layer in prefill mode."
    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 1
    export_mode = "prefill"


class Gemma4TextDecoderLayerSlidingPrefillCase(Gemma4TextDecoderLayerBaseCase):
    """Smoke case for one Gemma4 sliding-attention decoder layer.

    The case creates a two-layer text configuration and selects layer zero.
    This keeps layer zero as sliding attention while satisfying Gemma4's
    requirement that the final decoder layer use full attention.

    The sliding window is intentionally smaller than the sequence length so
    the input covers both future-token masking and left-side window masking.
    """

    name = "gemma4_text_decoder_layer_sliding_prefill"
    description = (
        "Quantize one tiny Gemma4 sliding-attention decoder layer with "
        "a causal sliding-window mask."
    )
    tags = (
        "gemma4",
        "e2b",
        "text",
        "decoder_layer",
        "prefill",
        "sliding",
    )

    layer_types = ("sliding_attention", "full_attention")
    layer_idx = 0
    export_mode = "prefill"

    seq_len = 8
    sliding_window = 4
    mask_fill_value = -120.0

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build a PTQ config matching the sample mask fill value."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(
            model_args={"profile": "reference_eval"},
            attention_mask_fill_value=self.mask_fill_value,
        )

    def build(
        self,
        cfg: Mapping[str, Any],
    ) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a sliding-attention decoder layer and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

        torch.manual_seed(123)
        self.text_cfg = _make_text_config(layer_types=self.layer_types)
        self.text_cfg.sliding_window = self.sliding_window

        module = Gemma4TextDecoderLayer(
            self.text_cfg,
            layer_idx=self.layer_idx,
        ).eval()

        if not module.self_attn.is_sliding:
            raise RuntimeError(
                "The smoke case did not build a sliding-attention layer."
            )
        if module.self_attn.sliding_window != self.sliding_window:
            raise RuntimeError(
                "The decoder layer does not use the requested sliding window: "
                f"expected {self.sliding_window}, "
                f"got {module.self_attn.sliding_window}."
            )

        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one fixed-shape sliding-window prefill sample."""
        batch_size = 1
        hidden = torch.randn(
            batch_size,
            self.seq_len,
            self.text_cfg.hidden_size,
        )
        attention_mask = _sliding_window_causal_mask(
            self.seq_len,
            self.sliding_window,
            batch_size=batch_size,
            dtype=hidden.dtype,
            device=hidden.device,
            fill_value=self.mask_fill_value,
        )

        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _text_rope(
                    batch_size,
                    self.seq_len,
                    self.text_cfg.head_dim,
                ),
                "attention_mask": attention_mask,
                "shared_kv_states": {},
            },
        )


class Gemma4TextDecoderLayerDecodeCase(Gemma4TextDecoderLayerBaseCase):
    """Smoke case for one tiny Gemma4 text decoder layer in decode mode."""

    name = "gemma4_text_decoder_layer_decode"
    description = "Quantize one tiny dense Gemma4 text decoder layer in decode mode."
    tags = ("gemma4", "e2b", "text", "decoder_layer", "decode")
    compare_reference_source = "prepared"
    seq_len = 1
    max_seq = 8
    export_mode = "decode"

    def _sample(self) -> ForwardInput:
        """Create one synthetic single-token decoder-layer decode sample."""
        hidden = torch.randn(1, 1, self.text_cfg.hidden_size)
        past_len = self.max_seq - 1
        past = (
            torch.randn(
                1,
                self.text_cfg.num_key_value_heads,
                past_len,
                self.text_cfg.head_dim,
            ),
            torch.randn(
                1,
                self.text_cfg.num_key_value_heads,
                past_len,
                self.text_cfg.head_dim,
            ),
        )
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _text_rope(1, 1, self.text_cfg.head_dim),
                "attention_mask": _attention_mask(1, self.max_seq),
                "past_key_value": past,
                "use_cache": True,
            },
        )

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static decode inputs expected by the decoder-layer adapter."""
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        return ForwardInput(
            (
                kwargs["hidden_states"],
                kwargs["attention_mask"],
                kwargs["position_embeddings"],
            ),
            {"past_key_value": kwargs["past_key_value"]},
        )


class Gemma4TextDecoderLayerSharedKVCase(Gemma4TextDecoderLayerBaseCase):
    """Smoke case for a Gemma4 shared-KV consumer decoder layer."""

    name = "gemma4_text_decoder_layer_shared_kv"
    description = "Quantize one tiny Gemma4 decoder layer that consumes shared K/V."
    tags = ("gemma4", "e2b", "text", "decoder_layer", "shared_kv")
    layer_types = ("full_attention", "full_attention")
    layer_idx = 1
    num_kv_shared_layers = 1
    export_mode = "prefill"

    def _sample(self) -> ForwardInput:
        """Create one synthetic shared-KV decoder-layer sample."""
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
                "shared_key_value": shared_key_value,
            },
        )


class Gemma4TextModelCase(Gemma4BaseCase):
    """Smoke case for one tiny dense Gemma4 text model."""

    name = "gemma4_text_model"
    description = (
        "Quantize one tiny dense Gemma4 text model with full and sliding attention."
    )
    tags = ("gemma4", "e2b", "text", "model")
    max_mean_abs_diff = 3.0
    seq_len = 8
    supports_circle_export = False
    circle_export_unsupported_reason = (
        "This case validates PTQ numerical parity only. "
        "Full Gemma4TextModel Circle export requires a dedicated static adapter."
    )

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4 text-model smoke checks."""
        from tico.quantization.config.ptq import PTQConfig

        return PTQConfig(model_args={"profile": "reference_eval"})

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 text model and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

        torch.manual_seed(123)
        self.text_cfg = _make_text_config(
            layer_types=("sliding_attention", "full_attention"),
        )
        module = Gemma4TextModel(self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 text-model input."""
        input_ids = torch.randint(0, self.text_cfg.vocab_size, (1, self.seq_len))
        attention_mask = torch.ones_like(input_ids)
        return ForwardInput(
            (),
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_dict": True,
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 text-model calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 text-model evaluation sample."""
        return self._sample()


def _make_vision_config() -> Any:
    """Create a tiny Gemma4 vision config for synthetic smoke tests."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    cfg = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        use_clipped_linears=False,
        rope_parameters={"rope_type": "default", "rope_theta": 100.0},
        standardize=True,
    )
    return _set_eager_attention(cfg)


def _vision_rope(
    batch_size: int,
    seq_len: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic Gemma4 vision RoPE embeddings."""
    emb = torch.randn(batch_size, seq_len, head_dim)
    return emb.cos(), emb.sin()


def _vision_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence."""
    side = 4
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def _pixel_position_ids(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create deterministic 2-D pixel position ids for a tiny patch sequence.

    The pooler requires ``pixel_position_ids`` with shape ``(B, S, 2)`` where
    the last dimension encodes ``(x, y)`` patch coordinates.  We build a
    simple square grid layout that is compatible with the ``output_length``
    used in pooler tests: ``seq_len = output_length * k^2`` where ``k`` is
    the pooling factor.
    """
    side = int(seq_len**0.5)
    coords = torch.arange(seq_len)
    xy = torch.stack((coords % side, coords // side), dim=-1)
    return xy.unsqueeze(0).expand(batch_size, -1, -1).long()


def _padding_positions(batch_size: int, seq_len: int) -> torch.Tensor:
    """Create an all-False padding mask (no padding)."""
    return torch.zeros(batch_size, seq_len, dtype=torch.bool)


class Gemma4VisionAttentionCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision attention module."""

    name = "gemma4_vision_attention"
    description = "Quantize one tiny Gemma4 vision attention module."
    tags = ("gemma4", "e2b", "vision", "attention")
    max_mean_abs_diff = 2.0
    seq_len = 8

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision attention module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionAttention

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config()
        module = Gemma4VisionAttention(self.vision_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision attention input."""
        batch_size = 1
        hidden = torch.randn(batch_size, self.seq_len, self.vision_cfg.hidden_size)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _vision_rope(
                    batch_size,
                    self.seq_len,
                    self.vision_cfg.head_dim,
                ),
                "attention_mask": torch.zeros(
                    batch_size, 1, self.seq_len, self.seq_len
                ),
                "position_ids": _vision_position_ids(batch_size, self.seq_len),
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision attention calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision attention evaluation sample."""
        return self._sample()


class Gemma4VisionEncoderLayerCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision encoder layer."""

    name = "gemma4_vision_encoder_layer"
    description = "Quantize one tiny Gemma4 vision encoder layer."
    tags = ("gemma4", "e2b", "vision", "encoder_layer")
    max_mean_abs_diff = 2.5
    seq_len = 8

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision encoder layer and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoderLayer

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config()
        module = Gemma4VisionEncoderLayer(self.vision_cfg, layer_idx=0).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision encoder-layer input."""
        batch_size = 1
        hidden = torch.randn(batch_size, self.seq_len, self.vision_cfg.hidden_size)
        return ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "position_embeddings": _vision_rope(
                    batch_size,
                    self.seq_len,
                    self.vision_cfg.head_dim,
                ),
                "attention_mask": torch.zeros(
                    batch_size, 1, self.seq_len, self.seq_len
                ),
                "position_ids": _vision_position_ids(batch_size, self.seq_len),
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision encoder-layer calibration samples."""
        return [self._sample() for _ in range(8)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision encoder-layer evaluation sample."""
        return self._sample()


class Gemma4TextScaledWordEmbeddingCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 text scaled word embedding module."""

    name = "gemma4_text_scaled_word_embedding"
    description = "Quantize one tiny Gemma4 text scaled word embedding module."
    tags = ("gemma4", "e2b", "text", "embedding")
    max_mean_abs_diff = 1.0
    vocab_size = 256
    embedding_dim = 64
    seq_len = 16
    embed_scale = 0.125

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 text scaled word embedding module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextScaledWordEmbedding,
        )

        torch.manual_seed(123)
        module = Gemma4TextScaledWordEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=0,
            embed_scale=self.embed_scale,
        ).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 text scaled word embedding input."""
        batch_size = 1
        input_ids = torch.randint(
            0, self.vocab_size, (batch_size, self.seq_len), dtype=torch.long
        )
        return ForwardInput((input_ids,))

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 text scaled word embedding calibration samples."""
        return [self._sample() for _ in range(8)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 text scaled word embedding evaluation sample."""
        return self._sample()


class Gemma4VisionPatchEmbedderCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision patch embedder module."""

    name = "gemma4_vision_patch_embedder"
    description = "Quantize one tiny Gemma4 vision patch embedder module."
    tags = ("gemma4", "e2b", "vision", "patch_embedder")
    max_mean_abs_diff = 2.0
    hidden_size = 32
    patch_size = 4
    position_embedding_size = 8
    batch_size = 1
    num_patches = 16

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision patch embedder module and reference copy."""
        from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

        torch.manual_seed(123)
        self.vision_cfg = Gemma4VisionConfig(
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            position_embedding_size=self.position_embedding_size,
        )
        module = Gemma4VisionPatchEmbedder(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision patch embedder input."""
        patch_dim = 3 * self.patch_size**2
        pixel_values = torch.randn(self.batch_size, self.num_patches, patch_dim)
        pixel_position_ids = torch.randint(
            0, self.position_embedding_size, (self.batch_size, self.num_patches, 2)
        )
        padding_positions = torch.zeros(
            self.batch_size, self.num_patches, dtype=torch.bool
        )
        return ForwardInput((pixel_values, pixel_position_ids, padding_positions))

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision patch embedder calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision patch embedder evaluation sample."""
        return self._sample()


class Gemma4VisionPoolerCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision pooler module."""

    name = "gemma4_vision_pooler"
    description = "Quantize one tiny Gemma4 vision pooler module."
    tags = ("gemma4", "e2b", "vision", "pooler")
    max_mean_abs_diff = 2.0
    # seq_len=16 and output_length=4 so that k=2 (16 / 4 = 4, sqrt(4) = 2).
    seq_len = 16
    output_length = 4

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision pooler module and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config()
        module = Gemma4VisionPooler(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision pooler input."""
        batch_size = 1
        return ForwardInput(
            (),
            {
                "hidden_states": torch.randn(
                    batch_size, self.seq_len, self.vision_cfg.hidden_size
                ),
                "pixel_position_ids": _pixel_position_ids(batch_size, self.seq_len),
                "padding_positions": _padding_positions(batch_size, self.seq_len),
                "output_length": self.output_length,
            },
        )

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run a Gemma4 vision pooler without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = module(*cloned.args, **dict(cloned.kwargs))
        # Return only the pooled features for comparison.
        return output[0] if isinstance(output, tuple) else output

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4 vision pooler without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        return output[0] if isinstance(output, tuple) else output

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision pooler calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision pooler evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped pooler in prefill mode with fixed output_length.

        Passes ``pixel_position_ids`` so the export adapter precomputes the
        pooling weight matrix and output mask at construction time, replacing
        the dynamic ``F.one_hot`` and ``torch.div`` operations with a static
        ``matmul``.
        """
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            pixel_pos_ids = _pixel_position_ids(1, self.seq_len)
            return wrapped.as_export_module(
                mode="prefill",
                output_length=self.output_length,
                pixel_position_ids=pixel_pos_ids,
            ).eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the pooler adapter.

        The export adapter bakes ``output_length`` as a construction-time
        constant, so it is not included in the forward signature.
        """
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        hidden = kwargs["hidden_states"]
        pixel_position_ids = kwargs["pixel_position_ids"]
        padding_positions = kwargs["padding_positions"]
        return ForwardInput((hidden, pixel_position_ids, padding_positions), {})


class Gemma4VisionModelCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 vision model."""

    name = "gemma4_vision_model"
    description = (
        "Quantize one tiny Gemma4 vision model (patch_embedder + encoder + pooler)."
    )
    tags = ("gemma4", "e2b", "vision", "model")
    max_mean_abs_diff = 3.0
    # seq_len=36 and output_length=4 so that k=2 (36 / 3^2 = 4, sqrt(4) = 2).
    seq_len = 36

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision model and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config()
        module = Gemma4VisionModel(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision model input.

        The HF Gemma4VisionModel expects pre-flattened patches:
            pixel_values: (B, num_patches, 3*patch_size^2)
            pixel_position_ids: (B, num_patches, 2)
        """
        batch_size = 1
        patch_size = self.vision_cfg.patch_size
        patch_dim = 3 * patch_size**2
        pixel_values = torch.randn(batch_size, self.seq_len, patch_dim)
        pixel_position_ids = _pixel_position_ids(batch_size, self.seq_len)
        return ForwardInput(
            (),
            {
                "pixel_values": pixel_values,
                "pixel_position_ids": pixel_position_ids,
                "return_dict": True,
            },
        )

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run a Gemma4 vision model without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = module(*cloned.args, **dict(cloned.kwargs))
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4 vision model without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision model calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision model evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped vision model in prefill mode.

        Passes ``pixel_position_ids`` so the pooler's export adapter can
        precompute the pooling weight matrix and output mask at construction
        time.
        """
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            pixel_pos_ids = _pixel_position_ids(1, self.seq_len)
            return wrapped.as_export_module(
                mode="prefill",
                pixel_position_ids=pixel_pos_ids,
            ).eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the vision model adapter.

        The export adapter's forward() takes pixel_values and pixel_position_ids.
        """
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        pixel_values = kwargs["pixel_values"]
        pixel_position_ids = kwargs["pixel_position_ids"]
        return ForwardInput((pixel_values, pixel_position_ids), {})


class Gemma4MultimodalEmbedderCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4 multimodal embedder module."""

    name = "gemma4_multimodal_embedder"
    description = (
        "Quantize one tiny Gemma4 multimodal embedder (RMSNorm + Linear projection)."
    )
    tags = ("gemma4", "e2b", "multimodal", "embedder")
    max_mean_abs_diff = 2.0
    seq_len = 16

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 multimodal embedder and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4MultimodalEmbedder

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config()
        self.text_cfg = _make_text_config()
        module = Gemma4MultimodalEmbedder(self.vision_cfg, self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 multimodal embedder input."""
        batch_size = 1
        inputs_embeds = torch.randn(
            batch_size, self.seq_len, self.vision_cfg.hidden_size
        )
        return ForwardInput((inputs_embeds,))

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 multimodal embedder calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 multimodal embedder evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped multimodal embedder in prefill mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module(mode="prefill").eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the multimodal embedder adapter.

        The export adapter's forward() takes inputs_embeds.
        """
        cloned = _clone_forward_input(eval_sample)
        inputs_embeds = cloned.args[0]
        return ForwardInput((inputs_embeds,), {})


class Gemma4VisionEncoderCase(Gemma4BaseCase):
    """Smoke case for a tiny Gemma4 vision encoder."""

    name = "gemma4_vision_encoder"
    description = "Quantize a tiny Gemma4 vision encoder."
    tags = ("gemma4", "e2b", "vision", "encoder")
    max_mean_abs_diff = 3.0
    seq_len = 8

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4 vision encoder and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionEncoder

        torch.manual_seed(123)
        self.vision_cfg = _make_vision_config()
        module = Gemma4VisionEncoder(self.vision_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4 vision encoder input."""
        batch_size = 1
        hidden = torch.randn(batch_size, self.seq_len, self.vision_cfg.hidden_size)
        return ForwardInput(
            (),
            {
                "inputs_embeds": hidden,
                "attention_mask": torch.ones(batch_size, self.seq_len),
                "pixel_position_ids": _vision_position_ids(batch_size, self.seq_len),
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4 vision encoder calibration samples."""
        return [self._sample() for _ in range(8)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4 vision encoder evaluation sample."""
        return self._sample()

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped vision encoder in prefill mode.

        Passes ``pixel_position_ids`` so the export adapter precomputes
        position embeddings and the bidirectional attention mask as
        registered buffers, replacing dynamic RoPE and mask computation
        with static gather from precomputed tables.
        """
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            pixel_pos_ids = _vision_position_ids(1, self.seq_len)
            return wrapped.as_export_module(
                mode="prefill", pixel_position_ids=pixel_pos_ids
            ).eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the vision encoder adapter.

        The export adapter's ``forward_export`` only takes ``inputs_embeds``.
        Position embeddings and attention mask are precomputed buffers.
        """
        cloned = _clone_forward_input(eval_sample)
        kwargs = dict(cloned.kwargs)
        hidden = kwargs["inputs_embeds"]
        return ForwardInput((hidden,), {})


class Gemma4ModelCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4Model (image-text)."""

    name = "gemma4_model"
    description = (
        "Quantize one tiny Gemma4Model (vision tower + language model + fusion)."
    )
    tags = ("gemma4", "e2b", "model", "image-text")
    max_mean_abs_diff = 5.0
    seq_len = 16
    num_visual_tokens = 4

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4Model smoke checks."""
        from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config

        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_cfg.num_hidden_layers),
            num_vision_layers=int(self.vision_cfg.num_hidden_layers),
            model_args={
                "vision": {
                    "visual_start_idx": 0,
                    "num_visual_tokens": self.num_visual_tokens,
                }
            },
        )

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4Model and reference copy."""
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
        from transformers.models.gemma4.modeling_gemma4 import Gemma4Model

        torch.manual_seed(123)
        self.text_cfg = _make_text_config(layer_types=("full_attention",))
        self.vision_cfg = _make_vision_config()

        config = Gemma4Config(
            text_config=self.text_cfg,
            vision_config=self.vision_cfg,
            audio_config=None,
            image_token_id=10,
            video_token_id=11,
            audio_token_id=12,
        )
        module = Gemma4Model(config).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4Model text-only input.

        Token IDs are kept in [0, 9] to avoid colliding with the image
        placeholder token ID (10).
        """
        input_ids = torch.randint(0, 10, (1, self.seq_len))
        return ForwardInput(
            (),
            {
                "input_ids": input_ids,
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4Model calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4Model evaluation sample."""
        return self._sample()

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run the Gemma4Model without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = module(*cloned.args, **dict(cloned.kwargs))
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4Model without sharing mutable sample state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        return output

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped Gemma4Model in prefill mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module(mode="prefill").eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the Gemma4Model export adapter.

        The export adapter's forward_export() takes precomputed inputs:
        - inputs_embeds: (1, S, H)
        - per_layer_inputs: (1, S, L, P) or None
        - attention_masks: dict[layer_type -> mask]
        - position_embeddings: dict[layer_type -> (cos, sin)]
        """
        hidden_size = int(self.text_cfg.hidden_size)
        head_dim = int(self.text_cfg.head_dim)
        num_layers = int(self.text_cfg.num_hidden_layers)
        ple_dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.text_cfg.layer_types)

        inputs_embeds = torch.randn(1, self.seq_len, hidden_size)

        per_layer_inputs = None
        if ple_dim > 0:
            per_layer_inputs = torch.randn(1, self.seq_len, num_layers, ple_dim)

        attention_masks: dict[str, torch.Tensor] = {}
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in layer_types:
            attention_masks[layer_type] = torch.zeros(1, 1, self.seq_len, self.seq_len)
            cos = torch.ones(1, self.seq_len, head_dim)
            sin = torch.zeros(1, self.seq_len, head_dim)
            position_embeddings[layer_type] = (cos, sin)

        return ForwardInput(
            (inputs_embeds, per_layer_inputs, attention_masks, position_embeddings),
            {},
        )


class Gemma4ForConditionalGenerationCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4ForConditionalGeneration."""

    name = "gemma4_for_conditional_generation"
    description = (
        "Quantize one tiny Gemma4ForConditionalGeneration "
        "(vision + text decoder + lm_head + softcapping)."
    )
    tags = ("gemma4", "e2b", "model", "conditional_generation", "image-text")
    max_mean_abs_diff = 5.0
    seq_len = 16
    num_visual_tokens = 4

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4ForConditionalGeneration smoke checks."""
        from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config

        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_cfg.num_hidden_layers),
            num_vision_layers=int(self.vision_cfg.num_hidden_layers),
            model_args={
                "vision": {
                    "visual_start_idx": 0,
                    "num_visual_tokens": self.num_visual_tokens,
                }
            },
        )

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4ForConditionalGeneration and reference copy."""
        from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
        )

        torch.manual_seed(123)
        self.text_cfg = _make_text_config(layer_types=("full_attention",))
        # Enable logit softcapping to exercise that code path.
        self.text_cfg.final_logit_softcapping = 30.0
        self.vision_cfg = _make_vision_config()

        config = Gemma4Config(
            text_config=self.text_cfg,
            vision_config=self.vision_cfg,
            audio_config=None,
            image_token_id=10,
            video_token_id=11,
            audio_token_id=12,
        )
        module = Gemma4ForConditionalGeneration(config).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4ForConditionalGeneration text-only input.

        Token IDs are kept in [0, 9] to avoid colliding with the image
        placeholder token ID (10).
        """
        input_ids = torch.randint(0, 10, (1, self.seq_len))
        return ForwardInput(
            (),
            {
                "input_ids": input_ids,
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4ForConditionalGeneration calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4ForConditionalGeneration evaluation sample."""
        return self._sample()

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run the Gemma4ForConditionalGeneration without sharing mutable sample state.

        The wrapper returns logits directly (not a Gemma4CausalLMOutputWithPast).
        """
        cloned = _clone_forward_input(sample)
        return module(*cloned.args, **dict(cloned.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4ForConditionalGeneration without sharing mutable state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        return output.logits if hasattr(output, "logits") else output

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped Gemma4ForConditionalGeneration in prefill mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module(mode="prefill").eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the export adapter.

        The export adapter's forward_export() takes precomputed inputs:
        - inputs_embeds: (1, S, H)
        - per_layer_inputs: (1, S, L, P) or None
        - attention_masks: dict[layer_type -> mask]
        - position_embeddings: dict[layer_type -> (cos, sin)]
        """
        hidden_size = int(self.text_cfg.hidden_size)
        head_dim = int(self.text_cfg.head_dim)
        num_layers = int(self.text_cfg.num_hidden_layers)
        ple_dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.text_cfg.layer_types)

        inputs_embeds = torch.randn(1, self.seq_len, hidden_size)

        per_layer_inputs = None
        if ple_dim > 0:
            per_layer_inputs = torch.randn(1, self.seq_len, num_layers, ple_dim)

        attention_masks: dict[str, torch.Tensor] = {}
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in layer_types:
            attention_masks[layer_type] = torch.zeros(1, 1, self.seq_len, self.seq_len)
            cos = torch.ones(1, self.seq_len, head_dim)
            sin = torch.zeros(1, self.seq_len, head_dim)
            position_embeddings[layer_type] = (cos, sin)

        return ForwardInput(
            (inputs_embeds, per_layer_inputs, attention_masks, position_embeddings),
            {},
        )


class Gemma4ForCausalLMCase(Gemma4BaseCase):
    """Smoke case for one tiny Gemma4ForCausalLM (text-only)."""

    name = "gemma4_for_causal_lm"
    description = (
        "Quantize one tiny Gemma4ForCausalLM " "(text decoder + lm_head + softcapping)."
    )
    tags = ("gemma4", "e2b", "model", "causal_lm", "text")
    max_mean_abs_diff = 5.0
    seq_len = 16

    def ptq_config(self, cfg: Mapping[str, Any]) -> Any:
        """Build the PTQ config used by Gemma4ForCausalLM smoke checks."""
        from tico.quantization.config.gemma4_builders import build_gemma4_e2b_ptq_config

        return build_gemma4_e2b_ptq_config(
            num_text_layers=int(self.text_cfg.num_hidden_layers),
            num_vision_layers=0,
        )

    def build(self, cfg: Mapping[str, Any]) -> tuple[torch.nn.Module, torch.nn.Module]:
        """Build a tiny Gemma4ForCausalLM and reference copy."""
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

        torch.manual_seed(123)
        self.text_cfg = _make_text_config(layer_types=("full_attention",))
        # Enable logit softcapping to exercise that code path.
        self.text_cfg.final_logit_softcapping = 30.0

        module = Gemma4ForCausalLM(self.text_cfg).eval()
        return module, clone_module(module)

    def _sample(self) -> ForwardInput:
        """Create one synthetic Gemma4ForCausalLM text-only input."""
        input_ids = torch.randint(0, self.text_cfg.vocab_size, (1, self.seq_len))
        return ForwardInput(
            (),
            {
                "input_ids": input_ids,
            },
        )

    def calibration_inputs(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> list[ForwardInput]:
        """Create Gemma4ForCausalLM calibration samples."""
        return [self._sample() for _ in range(3)]

    def eval_input(
        self,
        prepared: torch.nn.Module,
        cfg: Mapping[str, Any],
    ) -> ForwardInput:
        """Create the Gemma4ForCausalLM evaluation sample."""
        return self._sample()

    def forward(self, module: torch.nn.Module, sample: ForwardInput) -> Any:
        """Run the Gemma4ForCausalLM without sharing mutable sample state.

        The wrapper returns logits directly (not a Gemma4CausalLMOutputWithPast).
        """
        cloned = _clone_forward_input(sample)
        return module(*cloned.args, **dict(cloned.kwargs))

    def reference_forward(
        self, reference: torch.nn.Module, sample: ForwardInput
    ) -> Any:
        """Run the original Gemma4ForCausalLM without sharing mutable state."""
        cloned = _clone_forward_input(sample)
        output = reference(*cloned.args, **dict(cloned.kwargs))
        return output.logits if hasattr(output, "logits") else output

    def export_module(
        self, quantized: torch.nn.Module, cfg: Mapping[str, Any]
    ) -> torch.nn.Module:
        """Export the wrapped Gemma4ForCausalLM in prefill mode."""
        wrapped = getattr(quantized, "wrapped", quantized)
        if hasattr(wrapped, "as_export_module"):
            return wrapped.as_export_module(mode="prefill").eval()
        return quantized

    def export_input(
        self, eval_sample: ForwardInput, cfg: Mapping[str, Any]
    ) -> ForwardInput:
        """Create static export inputs expected by the export adapter.

        The export adapter's forward_export() takes precomputed inputs:
        - inputs_embeds: (1, S, H)
        - per_layer_inputs: (1, S, L, P) or None
        - attention_masks: dict[layer_type -> mask]
        - position_embeddings: dict[layer_type -> (cos, sin)]
        """
        hidden_size = int(self.text_cfg.hidden_size)
        head_dim = int(self.text_cfg.head_dim)
        num_layers = int(self.text_cfg.num_hidden_layers)
        ple_dim = int(getattr(self.text_cfg, "hidden_size_per_layer_input", 0) or 0)
        layer_types = list(self.text_cfg.layer_types)

        inputs_embeds = torch.randn(1, self.seq_len, hidden_size)

        per_layer_inputs = None
        if ple_dim > 0:
            per_layer_inputs = torch.randn(1, self.seq_len, num_layers, ple_dim)

        attention_masks: dict[str, torch.Tensor] = {}
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer_type in layer_types:
            attention_masks[layer_type] = torch.zeros(1, 1, self.seq_len, self.seq_len)
            cos = torch.ones(1, self.seq_len, head_dim)
            sin = torch.zeros(1, self.seq_len, head_dim)
            position_embeddings[layer_type] = (cos, sin)

        return ForwardInput(
            (inputs_embeds, per_layer_inputs, attention_masks, position_embeddings),
            {},
        )


GEMMA4_CASES = (
    Gemma4TextMLPCase(),
    Gemma4TextAttentionCase(),
    Gemma4TextSlidingAttentionCase(),
    Gemma4TextAttentionKEqVCase(),
    Gemma4TextAttentionSharedKVCase(),
    Gemma4TextDecoderLayerPrefillCase(),
    Gemma4TextDecoderLayerSlidingPrefillCase(),
    Gemma4TextDecoderLayerDecodeCase(),
    Gemma4TextDecoderLayerSharedKVCase(),
    Gemma4TextScaledWordEmbeddingCase(),
    Gemma4VisionPatchEmbedderCase(),
    Gemma4TextModelCase(),
    Gemma4VisionAttentionCase(),
    Gemma4VisionEncoderLayerCase(),
    Gemma4VisionPoolerCase(),
    Gemma4VisionModelCase(),
    Gemma4MultimodalEmbedderCase(),
    Gemma4VisionEncoderCase(),
    Gemma4ModelCase(),
    Gemma4ForConditionalGenerationCase(),
    Gemma4ForCausalLMCase(),
)
