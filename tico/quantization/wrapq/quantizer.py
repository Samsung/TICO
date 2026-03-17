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
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer
from tico.quantization.wrapq.wrap_helper import PTQWrapHelper
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


@register_quantizer(PTQConfig)
class PTQQuantizer(BaseQuantizer):
    """
    Post-Training Quantization (PTQ) quantizer integrated with the public interface.

    Features
    --------
    • Automatically wraps quantizable modules using PTQWrapper.
    • Supports leaf-level (single-module) quantization (e.g., prepare(model.fc, PTQConfig())).
    • Enforces strict wrapping if `strict_wrap=True`: raises NotImplementedError if
      no quantizable module was found at any boundary.
    • If `strict_wrap=False`, unquantizable modules are silently skipped.
    """

    def __init__(self, config: PTQConfig):
        super().__init__(config)
        self.qcfg: PTQConfig = config
        self.strict_wrap: bool = bool(getattr(config, "strict_wrap", True))
        self.wrapper = PTQWrapHelper(strict_wrap=self.strict_wrap)

    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Wrap the tree (or single module) according to strictness policy
        model = self.wrapper.wrap_supported(model, self.qcfg)

        # Switch all quant modules into calibration mode
        if isinstance(model, QuantModuleBase):
            model.enable_calibration()
        for m in model.modules():
            if isinstance(m, QuantModuleBase):
                m.enable_calibration()
        return model

    @torch.no_grad()
    def convert(self, model):
        # Freeze qparams across the tree (QUANT mode)
        if isinstance(model, QuantModuleBase):
            model.freeze_qparams()
        for m in model.modules():
            if isinstance(m, QuantModuleBase):
                m.freeze_qparams()
        return model
