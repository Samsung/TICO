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

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4TextAttention")
class QuantGemma4TextAttention(QuantModuleBase):
    """PTQ wrapper skeleton for dense Gemma4 text attention.

    The wrapper must support two layer modes:

    - Non-shared KV layers with q/k/v/o projections.
    - Shared-KV layers that reuse key/value tensors produced by an earlier layer.

    The first production implementation should avoid HF ``Cache`` objects in
    export mode and accept explicit static key/value tensors instead.
    """

    def __init__(
        self,
        fp_attn: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_attn
        self.config = fp_attn.config
        self.layer_idx = fp_attn.layer_idx
        self.layer_type = fp_attn.layer_type
        self.is_sliding = bool(getattr(fp_attn, "is_sliding", False))
        self.is_kv_shared_layer = bool(getattr(fp_attn, "is_kv_shared_layer", False))
        self.store_full_length_kv = bool(
            getattr(fp_attn, "store_full_length_kv", False)
        )
        self.use_alternative_attention = bool(
            getattr(fp_attn, "use_alternative_attention", False)
        )
        self.head_dim = int(fp_attn.head_dim)
        self.num_key_value_groups = int(fp_attn.num_key_value_groups)
        self.scaling = float(getattr(fp_attn, "scaling", 1.0))

        self.q_proj = PTQWrapper(
            fp_attn.q_proj,
            qcfg=qcfg.child("q_proj") if qcfg else None,
            fp_name=join_name(fp_name, "q_proj"),
        )
        self.q_norm = PTQWrapper(
            fp_attn.q_norm,
            qcfg=qcfg.child("q_norm") if qcfg else None,
            fp_name=join_name(fp_name, "q_norm"),
        )
        if not self.is_kv_shared_layer:
            self.k_proj = PTQWrapper(
                fp_attn.k_proj,
                qcfg=qcfg.child("k_proj") if qcfg else None,
                fp_name=join_name(fp_name, "k_proj"),
            )
            self.k_norm = PTQWrapper(
                fp_attn.k_norm,
                qcfg=qcfg.child("k_norm") if qcfg else None,
                fp_name=join_name(fp_name, "k_norm"),
            )
            self.v_norm = PTQWrapper(
                fp_attn.v_norm,
                qcfg=qcfg.child("v_norm") if qcfg else None,
                fp_name=join_name(fp_name, "v_norm"),
            )
            self.v_proj = None
            if fp_attn.v_proj is not None:
                self.v_proj = PTQWrapper(
                    fp_attn.v_proj,
                    qcfg=qcfg.child("v_proj") if qcfg else None,
                    fp_name=join_name(fp_name, "v_proj"),
                )

        self.o_proj = PTQWrapper(
            fp_attn.o_proj,
            qcfg=qcfg.child("o_proj") if qcfg else None,
            fp_name=join_name(fp_name, "o_proj"),
        )

        self.obs_hidden = self._make_obs("hidden")
        self.obs_cos = self._make_obs("cos")
        self.obs_sin = self._make_obs("sin")
        self.obs_logits = self._make_obs("logits")
        self.obs_softmax = self._make_obs("softmax")
        self.obs_attn_out = self._make_obs("attn_out")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        shared_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        cache_output_mode: str = "delta",
        **kwargs,
    ):
        """Run Gemma4 text attention.

        TODO: Implement the decomposed static attention math. The implementation
        should return ``(attn_output, attn_weights, present_key_value)`` when
        ``use_cache=True`` and return only hidden states otherwise.
        """
        raise NotImplementedError(
            "Gemma4 text attention static implementation is not wired yet."
        )

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (
            self.obs_hidden,
            self.obs_cos,
            self.obs_sin,
            self.obs_logits,
            self.obs_softmax,
            self.obs_attn_out,
        )
