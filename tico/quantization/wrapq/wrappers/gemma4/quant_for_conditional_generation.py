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
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.gemma4.modeling_gemma4.Gemma4ForConditionalGeneration"
)
class QuantGemma4ForConditionalGeneration(QuantModuleBase):
    """Top-level PTQ wrapper for Gemma4 E2B conditional generation."""

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        assert_gemma4_e2b_no_moe(fp_model)
        super().__init__(qcfg, fp_name=fp_name)
        self.module = fp_model
        self.config = fp_model.config
        self.model = PTQWrapper(
            fp_model.model,
            qcfg=qcfg.child("model") if qcfg else None,
            fp_name=join_name(fp_name, "model"),
        )
        self.lm_head = PTQWrapper(
            fp_model.lm_head,
            qcfg=qcfg.child("lm_head") if qcfg else None,
            fp_name=join_name(fp_name, "lm_head"),
        )

        # Observers for the logit softcapping path.
        self.obs_logit_softcapping_div = self._make_obs("logit_softcapping_div")
        self.obs_logit_softcapping_tanh = self._make_obs("logit_softcapping_tanh")
        self.obs_logits = self._make_obs("logits")

    def forward(self, *args, logits_to_keep: int | torch.Tensor = 0, **kwargs):
        """Run the wrapped conditional generation model (calibration path).

        Mirrors ``Gemma4ForConditionalGeneration.forward`` including logit
        softcapping.  Fake-quantization observers are inserted after the
        ``tanh`` and on the final logits so that the export path carries
        correct qparam metadata.

        TODO: Return ``Gemma4CausalLMOutputWithPast`` for full HF compatibility.
        """
        outputs = self.model(*args, **kwargs)
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs
        )
        # Match the original's logits_to_keep handling: int → slice, tensor → index.
        if isinstance(logits_to_keep, int) and logits_to_keep:
            slice_indices = slice(-logits_to_keep, None)
        elif isinstance(logits_to_keep, torch.Tensor):
            slice_indices = logits_to_keep
        else:
            slice_indices = slice(None)
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        return self._apply_logit_softcapping(logits)

    def _apply_logit_softcapping(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply logit softcapping with fake-quantization observers.

        Mirrors the original ``Gemma4ForConditionalGeneration`` softcapping:
        ``logits = tanh(logits / softcap) * softcap``.

        Three observers are inserted so that every graph node in the
        softcapping chain carries quantization parameter metadata:
        - ``obs_logit_softcapping_div``  — after the division
        - ``obs_logit_softcapping_tanh`` — after the tanh
        - ``obs_logits``                 — on the final logits
        """
        final_logit_softcapping = self.config.get_text_config().final_logit_softcapping
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = self._fq(logits, self.obs_logit_softcapping_div)
            logits = torch.tanh(logits)
            logits = self._fq(logits, self.obs_logit_softcapping_tanh)
            logits = logits * final_logit_softcapping

        logits = self._fq(logits, self.obs_logits)
        return logits

    def generate(self, *args, **kwargs):
        """Delegate generation to the original module until static runtime is wired."""
        return self.module.generate(*args, **kwargs)

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return (
            self.obs_logit_softcapping_div,
            self.obs_logit_softcapping_tanh,
            self.obs_logits,
        )
