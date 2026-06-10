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

from typing import Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionModel")
class QuantGemma4VisionModel(QuantModuleBase):
    """PTQ wrapper skeleton for the Gemma4 vision model."""

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config
        self.patch_embedder = PTQWrapper(
            fp_model.patch_embedder,
            qcfg=qcfg.child("patch_embedder") if qcfg else None,
            fp_name=join_name(fp_name, "patch_embedder"),
        )
        self.encoder = PTQWrapper(
            fp_model.encoder,
            qcfg=qcfg.child("encoder") if qcfg else None,
            fp_name=join_name(fp_name, "encoder"),
        )
        self.pooler = PTQWrapper(
            fp_model.pooler,
            qcfg=qcfg.child("pooler") if qcfg else None,
            fp_name=join_name(fp_name, "pooler"),
        )
        self.obs_last_hidden_state = self._make_obs("last_hidden_state")

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run Gemma4 vision model.

        TODO: Replace fallback pieces with a fully static implementation once the
        patch embedder and pooler wrappers are complete.
        """
        outputs = self.module(
            pixel_values=pixel_values,
            pixel_position_ids=pixel_position_ids,
            return_dict=True,
            **kwargs,
        )
        outputs.last_hidden_state = self._fq(
            outputs.last_hidden_state, self.obs_last_hidden_state
        )
        return outputs

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_last_hidden_state,)
