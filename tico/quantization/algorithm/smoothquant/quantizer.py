# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import Any, Dict, Optional

import torch

from tico.quantization.algorithm.smoothquant.observer import ChannelwiseMaxActsObserver

from tico.quantization.algorithm.smoothquant.smooth_quant import apply_smoothing
from tico.quantization.config.smoothquant import SmoothQuantConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer


@register_quantizer(SmoothQuantConfig)
class SmoothQuantQuantizer(BaseQuantizer):
    """
    Quantizer for applying the SmoothQuant algorithm

    Q) Why allow choosing between input and output activations?

    SmoothQuant relies on channel-wise activation statistics to balance
    weights and activations. In practice, there are two natural sources:

    - "input": captures the tensor right before a Linear layer
      (forward-pre-hook). This matches the original SmoothQuant paper
      and focuses on scaling the raw hidden state.

    - "output": captures the tensor right after a Linear layer
      (forward-hook). This can better reflect post-weight dynamics,
      especially when subsequent operations (bias, activation functions)
      dominate the dynamic range.

    Allowing both options provides flexibility: depending on model
    architecture and calibration data, one may yield lower error than
    the other. The default remains "input" for compatibility, but "output"
    can be selected to empirically reduce error or runtime overhead.
    """

    def __init__(self, config: SmoothQuantConfig):
        super().__init__(config)

        self.alpha = config.alpha
        self.custom_alpha_map = config.custom_alpha_map
        self.acts_from = config.acts_from  # "input" (default) or "output"
        self.observer: Optional[ChannelwiseMaxActsObserver] = None

    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Parameters:
            model: The target PyTorch model.
            args: Positional example inputs required for capturing graph.
            kwargs: Keyword example inputs required for capturing graph.

        Returns:
            The model prepared for SmoothQuant quantization.
        """
        # Attach hooks according to `config.acts_from`
        self.observer = ChannelwiseMaxActsObserver(model, acts_from=self.acts_from)
        self.observer.attach()

        return model

    @torch.no_grad()
    def convert(self, model):
        """
        Convert the prepared model to its SmoothQuant quantized version.
        Applies the SmoothQuant quantization on weights based on the collected statistics.

        Parameters:
            model: The prepared PyTorch model.

        Returns:
            The quantized model.
        """
        if self.observer is not None:
            self.observer.remove()
            apply_smoothing(
                model, self.observer.get_max_acts(), self.alpha, self.custom_alpha_map
            )

        return model
