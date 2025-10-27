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

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from tico.quantization.config.base import BaseConfig


class BaseQuantizer(ABC):
    """
    Abstract base class for quantizers that apply a quantization algorithm to a target model.
    """

    def __init__(self, config: BaseConfig):
        """
        Initialize the quantizer with the given configuration.

        Parameters:
            config (BaseConfig): Quantization configuration parameters.
        """
        self.config = config

    @abstractmethod
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Prepare the given model for quantization based on the provided algorithm-specific
         configuration. This involves setting up necessary observers or hooks, and may
        optionally use example inputs—particularly useful for activation quantization.

        Parameters:
            model: The target PyTorch model.
            args (Any, optional): Positional example inputs required for activation quantization.
            kwargs (Dict[str, Any], optional): Keyword example inputs required for activation quantization.

        Returns:
            The prepared model.
        """
        pass

    @abstractmethod
    def convert(self, model):
        """
        Convert the prepared (or calibrated) model into its quantized form. This function leverages
         the statistics collected during calibration to perform the quantization transformation.

        Parameters:
            model: The prepared PyTorch model.

        Returns:
            The quantized model.
        """
        pass
