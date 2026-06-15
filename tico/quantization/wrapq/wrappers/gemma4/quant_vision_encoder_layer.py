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

from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4VisionEncoderLayerPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionEncoderLayer")
class QuantGemma4VisionEncoderLayer(QuantModuleBase):
    """PTQ wrapper for one Gemma4 E2B vision encoder layer.

    The wrapper mirrors Hugging Face ``Gemma4VisionEncoderLayer`` for the dense
    vision path. It keeps the two residual additions explicit so activation
    observers can collect and fake-quantize the outputs of the attention residual
    block and the MLP residual block.

    Static runtime code should construct ``attention_mask``, ``position_ids``,
    and ``position_embeddings`` outside this wrapper. This wrapper only performs
    fixed-shape tensor compute that can later be exported as an NPU-friendly
    prefill graph.
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
        self.hidden_size = int(fp_layer.hidden_size)
        self.layer_idx = int(fp_layer.layer_idx)

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

        mk = self._make_obs
        self.obs_act_in = mk("act_in")
        self.obs_attn_residual_out = mk("attn_residual_out")
        self.obs_mlp_residual_out = mk("mlp_residual_out")

    @staticmethod
    def _extract_attention_hidden(attn_output: Any) -> torch.Tensor:
        """Return hidden states from the Gemma4 vision attention output."""
        if not isinstance(attn_output, tuple):
            return attn_output
        if not attn_output:
            raise RuntimeError("Gemma4 vision attention returned an empty tuple.")
        return attn_output[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run one Gemma4 vision encoder layer.

        Args:
            hidden_states: Input patch states shaped ``(B, S, hidden_size)``.
            position_embeddings: Tuple ``(cos, sin)`` produced by
                ``Gemma4VisionRotaryEmbedding`` for the static patch layout.
            attention_mask: Optional additive or keep mask consumed by the
                vision attention wrapper.
            position_ids: Optional 2-D pixel position ids shaped ``(B, S, 2)``.
            **kwargs: Additional attention keyword arguments kept for Hugging
                Face API compatibility.

        Returns:
            Output patch states shaped ``(B, S, hidden_size)``.
        """
        if position_embeddings is None:
            raise ValueError(
                "position_embeddings must be provided for Gemma4 vision encoder layers."
            )

        hidden_states = self._fq(hidden_states, self.obs_act_in)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        hidden_states = self._extract_attention_hidden(attn_output)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self._fq(residual + hidden_states, self.obs_attn_residual_out)

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return self._fq(residual + hidden_states, self.obs_mlp_residual_out)

    def as_export_module(self, mode: ExportMode = "prefill") -> nn.Module:
        """Return a static export adapter for the requested execution mode."""
        if mode == "prefill":
            return Gemma4VisionEncoderLayerPrefillExportAdapter(self)
        raise ValueError(
            f"Unsupported Gemma4 vision encoder-layer export mode: {mode!r}"
        )

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (
            self.obs_act_in,
            self.obs_attn_residual_out,
            self.obs_mlp_residual_out,
        )
