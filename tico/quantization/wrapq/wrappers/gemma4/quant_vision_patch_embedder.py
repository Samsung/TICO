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


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4VisionPatchEmbedder")
class QuantGemma4VisionPatchEmbedder(QuantModuleBase):
    """PTQ wrapper skeleton for Gemma4 vision patch embedding."""

    def __init__(
        self,
        fp_patch: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_patch
        self.input_proj = PTQWrapper(
            fp_patch.input_proj,
            qcfg=qcfg.child("input_proj") if qcfg else None,
            fp_name=join_name(fp_name, "input_proj"),
        )
        self.obs_position_add = self._make_obs("position_add")

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: Optional[torch.Tensor] = None,
    ):
        """Run patch projection and positional embedding addition.

        TODO: Replace the fallback call with a decomposed static implementation
        after the exact Gemma4 patch embedder fields are finalized.
        """
        hidden_states = self.module(pixel_values, pixel_position_ids)
        return self._fq(hidden_states, self.obs_position_add)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (self.obs_position_add,)
