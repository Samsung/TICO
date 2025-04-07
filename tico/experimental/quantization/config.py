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
from typing import Dict, Optional


class BaseConfig(ABC):
    """
    Base configuration class for quantization.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class PT2EConfig(BaseConfig):
    """
    Configuration for pytorch 2.0 export quantization.
    """

    @property
    def name(self) -> str:
        return "pt2e"


class GPTQConfig(BaseConfig):
    """
    Configuration for GPTQ.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "gptq"


class SmoothQuantConfig(BaseConfig):
    """
    Configuration for smooth quant.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        custom_alpha_map: Optional[Dict[str, float]] = None,
    ):
        self.alpha = alpha
        self.custom_alpha_map = custom_alpha_map

    @property
    def name(self) -> str:
        return "smooth_quant"
