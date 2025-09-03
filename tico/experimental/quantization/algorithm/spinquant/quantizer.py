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

from tico.experimental.quantization.algorithm.spinquant.spin_quant import apply_spin
from tico.experimental.quantization.config import SpinQuantConfig
from tico.experimental.quantization.quantizer import BaseQuantizer


class SpinQuantQuantizer(BaseQuantizer):
    """
    Quantizer for applying the SpinQuant algorithm
    """

    def __init__(self, config: SpinQuantConfig):
        super().__init__(config)

        self.mode = config.mode

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
            The model prepared for SpinQuant quantization.
        """
        # Do nothing
        # TODO: Add observer for quantization

        return model

    def convert(self, model):
        """
        Apply SpinQuant algorithm to the prepared model by rotating the weight matrices.
        The rotation matrices for applying SpinQuant are generated according to the mode. (default: random hadamard matrix)

        Parameters:
            model: The prepared PyTorch model.

        Returns:
            The rotated model.
        """

        return apply_spin(model, self.mode)
