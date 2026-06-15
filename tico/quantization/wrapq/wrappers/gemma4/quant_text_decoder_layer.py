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

from typing import Any, Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4TextDecoderLayerDecodeExportAdapter,
    Gemma4TextDecoderLayerPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


LayerKV = Tuple[torch.Tensor, torch.Tensor]


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4TextDecoderLayer")
class QuantGemma4TextDecoderLayer(QuantModuleBase):
    """PTQ wrapper for dense Gemma4 E2B text decoder layers.

    The wrapper mirrors the dense Hugging Face ``Gemma4TextDecoderLayer`` path
    and keeps residual additions, layer-scalar multiplication, and optional
    per-layer embedding (PLE) fusion explicit for activation observation.

    The E2B scope intentionally rejects MoE layers. Static runtime code should
    own dynamic orchestration such as KV-cache writes, shared-KV bookkeeping,
    position/RoPE creation, attention-mask creation, and sampling. This wrapper
    only performs fixed-shape tensor compute and returns optional K/V deltas when
    requested by the export adapters.
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_layer
        self.config = fp_layer.config
        self.layer_idx = int(fp_layer.layer_idx)
        self.hidden_size = int(fp_layer.hidden_size)
        self.hidden_size_per_layer_input = int(
            getattr(fp_layer, "hidden_size_per_layer_input", 0) or 0
        )
        self.enable_moe_block = bool(getattr(fp_layer, "enable_moe_block", False))

        if self.enable_moe_block:
            raise NotImplementedError(
                "Gemma4 E2B static PTQ supports dense decoder layers only. "
                "MoE routing and expert dispatch are intentionally out of scope."
            )

        self.self_attn = PTQWrapper(
            fp_layer.self_attn,
            qcfg=qcfg.child("self_attn") if qcfg else None,
            fp_name=join_name(fp_name, "self_attn"),
        )
        self.mlp = PTQWrapper(
            fp_layer.mlp,
            qcfg=qcfg.child("mlp") if qcfg else None,
            fp_name=join_name(fp_name, "mlp"),
        )
        self.input_layernorm = PTQWrapper(
            fp_layer.input_layernorm,
            qcfg=qcfg.child("input_layernorm") if qcfg else None,
            fp_name=join_name(fp_name, "input_layernorm"),
        )
        self.post_attention_layernorm = PTQWrapper(
            fp_layer.post_attention_layernorm,
            qcfg=qcfg.child("post_attention_layernorm") if qcfg else None,
            fp_name=join_name(fp_name, "post_attention_layernorm"),
        )
        self.pre_feedforward_layernorm = PTQWrapper(
            fp_layer.pre_feedforward_layernorm,
            qcfg=qcfg.child("pre_feedforward_layernorm") if qcfg else None,
            fp_name=join_name(fp_name, "pre_feedforward_layernorm"),
        )
        self.post_feedforward_layernorm = PTQWrapper(
            fp_layer.post_feedforward_layernorm,
            qcfg=qcfg.child("post_feedforward_layernorm") if qcfg else None,
            fp_name=join_name(fp_name, "post_feedforward_layernorm"),
        )

        self.per_layer_input_gate: Optional[nn.Module] = None
        self.per_layer_projection: Optional[nn.Module] = None
        self.post_per_layer_input_norm: Optional[nn.Module] = None
        self.act_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = getattr(
            fp_layer, "act_fn", None
        )
        if self.hidden_size_per_layer_input:
            self.per_layer_input_gate = PTQWrapper(
                fp_layer.per_layer_input_gate,
                qcfg=qcfg.child("per_layer_input_gate") if qcfg else None,
                fp_name=join_name(fp_name, "per_layer_input_gate"),
            )
            self.per_layer_projection = PTQWrapper(
                fp_layer.per_layer_projection,
                qcfg=qcfg.child("per_layer_projection") if qcfg else None,
                fp_name=join_name(fp_name, "per_layer_projection"),
            )
            self.post_per_layer_input_norm = PTQWrapper(
                fp_layer.post_per_layer_input_norm,
                qcfg=qcfg.child("post_per_layer_input_norm") if qcfg else None,
                fp_name=join_name(fp_name, "post_per_layer_input_norm"),
            )

        layer_scalar = getattr(fp_layer, "layer_scalar", torch.ones(1))
        self.register_buffer(
            "layer_scalar",
            layer_scalar.detach().clone().to(dtype=torch.float32),
            persistent=False,
        )

        mk = self._make_obs
        self.obs_act_in = mk("act_in")
        self.obs_attn_residual_out = mk("attn_residual_out")
        self.obs_mlp_residual_out = mk("mlp_residual_out")
        self.obs_layer_scalar = mk("layer_scalar")
        self.obs_layer_scalar_out = mk("layer_scalar_out")

        self.obs_per_layer_gate_act = None
        self.obs_per_layer_mul = None
        self.obs_per_layer_residual_out = None
        if self.hidden_size_per_layer_input:
            self.obs_per_layer_gate_act = mk("per_layer_gate_act")
            self.obs_per_layer_mul = mk("per_layer_mul")
            self.obs_per_layer_residual_out = mk("per_layer_residual_out")

    @staticmethod
    def _resolve_past_key_value(
        past_key_value: Optional[Any],
        past_key_values: Optional[Any],
    ) -> Optional[Any]:
        """Return the cache object while accepting both singular and HF names."""
        if past_key_value is None:
            return past_key_values
        if past_key_values is None:
            return past_key_value
        if past_key_value is past_key_values:
            return past_key_value
        raise ValueError(
            "Provide only one of past_key_value or past_key_values for "
            "Gemma4 text decoder layers."
        )

    @staticmethod
    def _extract_attention_output(
        attn_output: Any,
    ) -> tuple[torch.Tensor, Optional[Any]]:
        """Extract hidden states and optional cache output from attention output."""
        if not isinstance(attn_output, tuple):
            return attn_output, None
        if not attn_output:
            raise RuntimeError("Gemma4 text attention returned an empty tuple.")
        hidden_states = attn_output[0]
        cache_output = attn_output[2] if len(attn_output) > 2 else None
        return hidden_states, cache_output

    def _apply_per_layer_input(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply the optional Gemma4 per-layer embedding fusion block."""
        if not self.hidden_size_per_layer_input:
            return hidden_states
        if per_layer_input is None:
            raise ValueError(
                "per_layer_input must be provided when Gemma4 PLE is enabled."
            )

        per_layer_input_gate = self.per_layer_input_gate
        per_layer_projection = self.per_layer_projection
        post_per_layer_input_norm = self.post_per_layer_input_norm
        act_fn = self.act_fn
        if (
            per_layer_input_gate is None
            or per_layer_projection is None
            or post_per_layer_input_norm is None
            or act_fn is None
            or self.obs_per_layer_gate_act is None
            or self.obs_per_layer_mul is None
            or self.obs_per_layer_residual_out is None
        ):
            raise RuntimeError("Gemma4 PLE modules are not initialized.")

        residual = hidden_states
        hidden_states = per_layer_input_gate(hidden_states)
        hidden_states = self._fq(act_fn(hidden_states), self.obs_per_layer_gate_act)
        hidden_states = self._fq(
            hidden_states * per_layer_input, self.obs_per_layer_mul
        )
        hidden_states = per_layer_projection(hidden_states)
        hidden_states = post_per_layer_input_norm(hidden_states)
        return self._fq(residual + hidden_states, self.obs_per_layer_residual_out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        shared_kv_states: Optional[dict[str, LayerKV]] = None,
        shared_key_value: Optional[LayerKV] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        cache_output_mode: str = "delta",
        **kwargs,
    ):
        """Run one dense Gemma4 text decoder layer.

        Args:
            hidden_states: Input tensor shaped ``(B, S, hidden_size)``.
            per_layer_input: Optional PLE tensor shaped
                ``(B, S, hidden_size_per_layer_input)``.
            shared_kv_states: Optional mutable mapping used by HF-style shared-KV
                layers. Static runtime code may omit it and pass
                ``shared_key_value`` directly instead.
            shared_key_value: Optional explicit K/V tuple for shared-KV consumer
                layers. This is the preferred static-export contract.
            position_embeddings: Tuple ``(cos, sin)`` for the layer type.
            attention_mask: Additive or keep mask consumed by the attention wrapper.
            position_ids: Optional position ids forwarded for API compatibility.
            past_key_value: Optional tuple or cache object for static-runtime decode.
            past_key_values: Hugging Face cache-name alias.
            use_cache: When ``True``, return ``(hidden_states, cache_delta)``.
            cache_output_mode: Cache return policy accepted by
                ``QuantGemma4TextAttention``.
            **kwargs: Extra attention kwargs kept for HF API compatibility.

        Returns:
            ``hidden_states`` when ``use_cache=False``. When ``use_cache=True``,
            returns ``(hidden_states, cache_delta_or_present)``. Shared-KV
            consumer layers return ``cache_delta_or_present=None`` because they do
            not own K/V projection weights.
        """
        if position_embeddings is None:
            raise ValueError("position_embeddings must be provided for Gemma4 text.")

        past = self._resolve_past_key_value(past_key_value, past_key_values)
        hidden_states = self._fq(hidden_states, self.obs_act_in)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            shared_key_value=shared_key_value,
            position_ids=position_ids,
            past_key_value=past,
            use_cache=use_cache,
            cache_output_mode=cache_output_mode,
            **kwargs,
        )
        hidden_states, cache_output = self._extract_attention_output(attn_output)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._fq(residual + hidden_states, self.obs_attn_residual_out)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self._fq(residual + hidden_states, self.obs_mlp_residual_out)

        hidden_states = self._apply_per_layer_input(hidden_states, per_layer_input)

        layer_scalar = self._fq(
            self.layer_scalar.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            ),
            self.obs_layer_scalar,
        )
        hidden_states = self._fq(
            hidden_states * layer_scalar,
            self.obs_layer_scalar_out,
        )

        if use_cache:
            return hidden_states, cache_output
        return hidden_states

    def as_export_module(self, mode: ExportMode = "prefill", *, return_kv: bool = True):
        """Return a static export adapter for the requested execution mode."""
        if mode == "prefill":
            return Gemma4TextDecoderLayerPrefillExportAdapter(self, return_kv=return_kv)
        if mode == "decode":
            return Gemma4TextDecoderLayerDecodeExportAdapter(self, return_kv=return_kv)
        raise ValueError(f"Unsupported Gemma4 export mode: {mode!r}")

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        observers = [
            self.obs_act_in,
            self.obs_attn_residual_out,
            self.obs_mlp_residual_out,
            self.obs_layer_scalar,
            self.obs_layer_scalar_out,
        ]
        if self.hidden_size_per_layer_input:
            observers.extend(
                [
                    self.obs_per_layer_gate_act,
                    self.obs_per_layer_mul,
                    self.obs_per_layer_residual_out,
                ]
            )
        return tuple(obs for obs in observers if obs is not None)
