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

from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e

from tico.quantization.algorithm.pt2e.annotation.annotator import (
    get_asymmetric_quantization_config,
    PT2EAnnotator,
)
from tico.quantization.config.pt2e import PT2EConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer


@register_quantizer(PT2EConfig)
class PT2EQuantizer(BaseQuantizer):
    """
    Quantizer for applying pytorch 2.0 export quantization (typically for activation quantization).
    """

    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Prepare the model for pt2e quantization.

        Registers activation observers using the provided example inputs.

        Parameters:
            model: The target PyTorch model.
            args: Positional example inputs required for capturing graph.
            kwargs: Keyword example inputs required for capturing graph.

        Returns:
            The model prepared for pt2e quantization.
        """
        # Program capture
        assert isinstance(args, tuple)
        model = torch.export.export_for_training(
            model, args=args, kwargs=kwargs
        ).module()
        quantizer = PT2EAnnotator()
        quantizer = quantizer.set_global(get_asymmetric_quantization_config())

        # Register observers in each nodes
        assert isinstance(model, torch.fx.GraphModule)
        model = prepare_pt2e(model, quantizer)

        return model

    def convert(self, model: torch.fx.GraphModule):
        """
        Convert the prepared model to its pt2e quantized version.

        Applies the pt2e quantization on activations based on the collected statistics.

        Parameters:
            model: The prepared PyTorch model.

        Returns:
            The quantized model.
        """
        return convert_pt2e(model)
