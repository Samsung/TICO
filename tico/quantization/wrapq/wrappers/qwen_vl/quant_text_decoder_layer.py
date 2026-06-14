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

from typing import Iterable, Literal, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.qwen_vl.export_adapters import (
    Qwen3VLTextDecoderLayerDecodeExportAdapter,
    Qwen3VLTextDecoderLayerPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.registry import try_register


ReturnType = Literal["tensor", "tuple"]


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLTextDecoderLayer",
)
class QuantQwen3VLTextDecoderLayer(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLTextDecoderLayer.

    The wrapper keeps the eager Qwen3-VL text-layer behavior compatible with the
    existing full-model wrapper while also exposing static prefill/decode export
    adapters for a CPU-managed KV-cache runtime.
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
        return_type: Optional[ReturnType] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        assert hasattr(fp_layer, "self_attn")
        assert hasattr(fp_layer, "mlp")
        assert hasattr(fp_layer, "input_layernorm")
        assert hasattr(fp_layer, "post_attention_layernorm")

        self.return_type: ReturnType = return_type or "tensor"

        # --- Wrap submodules via PTQWrapper ----------------------------------
        self_attn_cfg = qcfg.child("self_attn") if qcfg else None
        mlp_cfg = qcfg.child("mlp") if qcfg else None
        input_layernorm_cfg = qcfg.child("input_layernorm") if qcfg else None
        post_attention_layernorm_cfg = (
            qcfg.child("post_attention_layernorm") if qcfg else None
        )

        self.self_attn = PTQWrapper(
            fp_layer.self_attn,
            qcfg=self_attn_cfg,
            fp_name=join_name(fp_name, "self_attn"),
        )

        self.mlp = PTQWrapper(
            fp_layer.mlp,
            qcfg=mlp_cfg,
            fp_name=join_name(fp_name, "mlp"),
        )

        self.input_layernorm = PTQWrapper(
            fp_layer.input_layernorm,
            qcfg=input_layernorm_cfg,
            fp_name=join_name(fp_name, "input_layernorm"),
        )

        self.post_attention_layernorm = PTQWrapper(
            fp_layer.post_attention_layernorm,
            qcfg=post_attention_layernorm_cfg,
            fp_name=join_name(fp_name, "post_attention_layernorm"),
        )

        # --- Observers for residual additions --------------------------------
        mk = self._make_obs
        self.obs_act_in = mk("act_in")
        self.obs_post_attn = mk("post_attn")
        self.obs_act_out = mk("act_out")

    @staticmethod
    def _unpack_attn_outputs(attn_out):
        """
        Normalize attention outputs into hidden states, attention weights, and cache.
        """
        if not isinstance(attn_out, tuple):
            return attn_out, None, None

        hidden_states_attn = attn_out[0]
        attn_weights = attn_out[1] if len(attn_out) > 1 else None
        present_key_value = attn_out[2] if len(attn_out) > 2 else None
        return hidden_states_attn, attn_weights, present_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        cache_output_mode: Literal["present", "delta"] = "present",
        **kwargs,
    ):
        """
        Run the quantized Qwen3-VL text decoder layer.

        Args:
            hidden_states: Input tensor with shape `(batch_size, seq_len, hidden_size)`.
            position_embeddings: RoPE cosine and sine tensors.
            attention_mask: Optional attention mask tensor.
            position_ids: Optional position indices kept for HF compatibility.
            past_key_values: Optional per-layer static KV tuple or HF Cache-like object.
            use_cache: Whether to return cache tensors.
            output_attentions: Whether to include attention weights when returning a tuple.
            cache_position: Optional cache positions for HF Cache-like objects.
            cache_output_mode: Return either full present KV tensors or only the KV delta.
            **kwargs: Additional keyword arguments passed to attention.

        Returns:
            By default, the transformed hidden states. When `return_type` is set
            to `"tuple"`, returns `(hidden_states, ...)` with optional attention
            weights and cache tensors.
        """
        # Quantize input activation
        hidden_states = self._fq(hidden_states, self.obs_act_in)

        # Save input for residual connection
        residual = hidden_states

        # Pre-attention normalization
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention
        attn_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            cache_output_mode=cache_output_mode,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states_attn, attn_weights, present_key_value = self._unpack_attn_outputs(
            attn_out
        )

        # Post-attention residual connection
        hidden_states = hidden_states_attn + residual
        hidden_states = self._fq(hidden_states, self.obs_post_attn)

        # Save for MLP residual connection
        residual = hidden_states

        # Pre-MLP normalization
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Feed-forward network
        hidden_states = self.mlp(hidden_states)

        # Post-MLP residual connection
        hidden_states = hidden_states + residual

        # Quantize output activation
        hidden_states = self._fq(hidden_states, self.obs_act_out)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        if self.return_type == "tuple":
            return outputs
        if self.return_type == "tensor":
            return hidden_states
        raise RuntimeError(f"Invalid return_type: {self.return_type!r}")

    def _all_observers(self) -> Iterable:
        """Yield all observers from this module."""
        yield from (
            self.obs_act_in,
            self.obs_post_attn,
            self.obs_act_out,
        )

    def as_export_module(self, mode: ExportMode = "prefill", *, return_kv: bool = True):
        """
        Return a static-runtime export adapter for the requested execution mode.
        """
        if mode == "prefill":
            return Qwen3VLTextDecoderLayerPrefillExportAdapter(
                self,
                return_kv=return_kv,
            )
        if mode == "decode":
            return Qwen3VLTextDecoderLayerDecodeExportAdapter(
                self,
                return_kv=return_kv,
            )
        raise ValueError(f"Unsupported export mode: {mode!r}")
