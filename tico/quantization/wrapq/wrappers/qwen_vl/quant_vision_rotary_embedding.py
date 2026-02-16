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

from typing import Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.qwen3_vl.modeling_qwen3_vl.Qwen3VLVisionRotaryEmbedding",
)
class QuantQwen3VLVisionRotaryEmbedding(QuantModuleBase):
    """
    Quantization wrapper for Qwen3VLVisionRotaryEmbedding module.

    This module generates rotary positional embedding frequencies for vision tokens.
    Since it has no learnable parameters (only a constant buffer), only the
    output activation is quantized.
    """

    def __init__(
        self,
        fp_rope: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg, fp_name=fp_name)

        assert hasattr(fp_rope, "dim")
        assert hasattr(fp_rope, "theta")
        assert hasattr(fp_rope, "inv_freq")

        self.dim = fp_rope.dim
        self.theta = fp_rope.theta

        # Copy the inv_freq buffer to the wrapper
        self.register_buffer("inv_freq", fp_rope.inv_freq.clone())

        # Observer for output activation (rotary frequencies)
        self.obs_output = self._make_obs("output")

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Forward pass with fake quantization.

        Args:
            seqlen: Sequence length (number of positions/tokens)

        Returns:
            Rotary frequencies of shape (seqlen, dim/2)
        """
        # Compute sequence tensor
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )

        # Compute outer product (frequencies)
        freqs = torch.outer(seq, self.inv_freq)

        # Quantize output activation
        freqs = self._fq(freqs, self.obs_output)

        return freqs

    def _all_observers(self):
        """Yield the output observer."""
        yield self.obs_output
