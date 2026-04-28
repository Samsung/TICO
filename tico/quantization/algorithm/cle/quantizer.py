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

from typing import Any, Dict, Optional

import torch

from tico.quantization.algorithm.cle.cle import apply_cross_layer_equalization
from tico.quantization.config.cle import CLEConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer


@register_quantizer(CLEConfig)
class CLEQuantizer(BaseQuantizer):
    """
    Quantizer for applying Cross-Layer Equalization.

    Cross-Layer Equalization is a data-free PTQ preprocessing algorithm.
    Therefore, ``prepare`` is a no-op and the actual transformation is
    performed in ``convert``.
    """

    def __init__(self, config: CLEConfig):
        super().__init__(config)
        self.applied_scales = {}

    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Prepare the model for Cross-Layer Equalization.

        CLE does not require calibration data or observers, so this method
        returns the model unchanged.

        Args:
            model: Target PyTorch model.
            args: Optional positional example inputs. Unused.
            kwargs: Optional keyword example inputs. Unused.

        Returns:
            The unchanged model.
        """
        return model

    @torch.no_grad()
    def convert(self, model: torch.nn.Module):
        """
        Apply Cross-Layer Equalization to the prepared model.

        Args:
            model: Prepared PyTorch model.

        Returns:
            The equalized model.
        """
        self.applied_scales = apply_cross_layer_equalization(model, self.config)
        return model
