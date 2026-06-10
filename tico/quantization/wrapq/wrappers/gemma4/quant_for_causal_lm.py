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
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4LMHeadExportAdapter,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4ForCausalLM")
class QuantGemma4ForCausalLM(QuantModuleBase):
    """PTQ wrapper skeleton for Gemma4 text-only causal LM."""

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

    def forward(self, *args, **kwargs):
        """Run text-only causal LM inference.

        TODO: Complete return object compatibility after the text model wrapper
        is implemented.
        """
        outputs = self.model(*args, **kwargs)
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs
        )
        logits = self.lm_head(hidden_states)
        return logits

    def as_export_module(self, mode: str, **kwargs):
        """Return a static export adapter for the requested execution mode."""
        if mode == "lm_head":
            return Gemma4LMHeadExportAdapter(self)
        raise ValueError(f"Unsupported Gemma4ForCausalLM export mode: {mode!r}")

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        return ()
