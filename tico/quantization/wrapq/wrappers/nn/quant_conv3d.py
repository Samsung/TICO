# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import register


@register(nn.Conv3d)
class QuantConv3d(QuantModuleBase):
    """
    Quantization wrapper for nn.Conv3d with:
    - Per-channel weight quantization (asymmetric)
    - Per-tensor input activation quantization
    - Per-tensor output activation quantization
    """

    def __init__(
        self,
        fp: nn.Conv3d,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None
    ):
        """
        Initialize a QuantConv3d wrapper around a floating-point Conv3d module.

        This wrapper adds per-channel weight quantization (asymmetric) and per-tensor
        activation quantization (input and output) to support post-training quantization (PTQ)
        of 3D convolution layers.

        Parameters
        ----------
        fp : nn.Conv3d
            The floating-point Conv3d module to wrap and quantize. This is the original
            unquantized module whose weights will be quantized on a per-output-channel basis
            and whose input/output activations will be quantized during calibration and
            inference.

        qcfg : PTQConfig, optional
            Quantization configuration object that specifies:
            - Default data type for quantization (e.g., uint8, uint4)
            - Default observer type (e.g., MinMax, EMA)
            - Default quantization scheme (e.g., per-tensor, per-channel)
            - Per-observer overrides for weight, input activation, and output activation

            If None, defaults to PTQConfig with uint8 quantization, MinMax observer,
            and per-channel quantization for weights (along output channel axis 0).

        fp_name : str, optional
            Human-readable name for the floating-point module, used for:
            - Debugging and logging
            - Error messages when quantization fails
            - Configuration path resolution when using nested configurations

            If None, the module's class name will be used.
        """

        super().__init__(qcfg, fp_name=fp_name)

        # Weight observer: per-channel asymmetric quantization
        # channel_axis=0 means quantize per output channel (Conv3d weights' format is (C_out, C_in, D, H, W))
        self.obs_weight = self._make_obs(
            "weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )

        # Activation observers
        self.obs_act_in = self._make_obs("act_in")
        self.obs_act_out = self._make_obs("act_out")

        # Store the original module
        self.module = fp

    def enable_calibration(self) -> None:
        """
        Enable calibration mode.

        Immediately capture the fixed weight range since weights don't change
        during calibration.
        """
        super().enable_calibration()

        # Collect weight statistics immediately (weights are static)
        self.obs_weight.collect(self.module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fake quantization.

        During CALIB mode: collects statistics, no quantization applied
        During QUANT mode: applies fake quantization to input, weight, and output
        """
        # Quantize input activation
        x_q = self._fq(x, self.obs_act_in)

        # Get weight (quantized if in QUANT mode)
        w = self.module.weight
        if self._mode is Mode.QUANT:
            w = self.obs_weight.fake_quant(w)

        # Get bias (not quantized)
        b = self.module.bias

        # Perform convolution with quantized input and weights
        out = F.conv3d(
            x_q,
            w,
            bias=b,
            stride=self.module.stride,
            padding=self.module.padding,
            dilation=self.module.dilation,
            groups=self.module.groups,
        )

        # Quantize output activation
        return self._fq(out, self.obs_act_out)

    def _all_observers(self):
        """
        Return all observers for this module.

        Used by the parent QuantModuleBase for iteration and calibration.
        """
        return (self.obs_weight, self.obs_act_in, self.obs_act_out)
