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

from typing import Callable, Iterable, Optional, Tuple

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


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4TextDecoderLayer")
class QuantGemma4TextDecoderLayer(QuantModuleBase):
    """PTQ wrapper skeleton for dense Gemma4 E2B text decoder layers."""

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
        self.layer_idx = fp_layer.layer_idx
        self.hidden_size = int(fp_layer.hidden_size)
        self.hidden_size_per_layer_input = getattr(
            fp_layer, "hidden_size_per_layer_input", None
        )

        if bool(getattr(fp_layer, "enable_moe_block", False)):
            raise NotImplementedError(
                "Gemma4 E2B skeleton supports dense decoder layers only."
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

        self.layer_scalar = fp_layer.layer_scalar
        self.obs_attn_residual_out = self._make_obs("attn_residual_out")
        self.obs_mlp_residual_out = self._make_obs("mlp_residual_out")
        self.obs_layer_scalar_out = self._make_obs("layer_scalar_out")
        self.obs_per_layer_mul = self._make_obs("per_layer_mul")

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        shared_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cache_output_mode: str = "delta",
        **kwargs,
    ):
        """Run one dense Gemma4 decoder layer.

        TODO: Wire the attention wrapper return contract once
        ``QuantGemma4TextAttention`` is implemented.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_key_value=shared_key_value,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_output_mode=cache_output_mode,
            **kwargs,
        )
        if isinstance(attn_out, tuple):
            hidden_states = attn_out[0]
            present_key_value = attn_out[2] if len(attn_out) > 2 else None
        else:
            hidden_states = attn_out
            present_key_value = None

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._fq(residual + hidden_states, self.obs_attn_residual_out)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = self._fq(residual + hidden_states, self.obs_mlp_residual_out)

        if self.hidden_size_per_layer_input:
            if per_layer_input is None:
                raise ValueError(
                    "per_layer_input must be provided when Gemma4 PLE is enabled."
                )
            per_layer_input_gate = self.per_layer_input_gate
            act_fn = self.act_fn
            per_layer_projection = self.per_layer_projection
            post_per_layer_input_norm = self.post_per_layer_input_norm
            if (
                per_layer_input_gate is None
                or act_fn is None
                or per_layer_projection is None
                or post_per_layer_input_norm is None
            ):
                raise RuntimeError("Gemma4 PLE modules are not initialized.")
            residual = hidden_states
            hidden_states = per_layer_input_gate(hidden_states)
            hidden_states = act_fn(hidden_states)
            hidden_states = self._fq(
                hidden_states * per_layer_input, self.obs_per_layer_mul
            )
            hidden_states = per_layer_projection(hidden_states)
            hidden_states = post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self._fq(
            hidden_states * self.layer_scalar, self.obs_layer_scalar_out
        )
        if use_cache:
            return hidden_states, present_key_value
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
        return (
            self.obs_attn_residual_out,
            self.obs_mlp_residual_out,
            self.obs_layer_scalar_out,
            self.obs_per_layer_mul,
        )
