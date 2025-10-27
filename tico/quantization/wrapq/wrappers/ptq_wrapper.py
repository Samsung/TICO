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

from typing import Optional

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import lookup


class PTQWrapper(QuantModuleBase):
    """
    Adapter that turns a fp module into its quantized counterpart.

    It is itself a QuantModuleBase so composite wrappers can treat
     it exactly like any other quant module.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        qcfg: Optional[PTQConfig] = None,
        *,
        fp_name: Optional[str] = None,
    ):
        super().__init__(qcfg)
        wrapped_cls = lookup(type(module))
        if wrapped_cls is None:
            raise NotImplementedError(f"No quant wrapper for {type(module).__name__}")
        self.wrapped: QuantModuleBase = wrapped_cls(module, qcfg=qcfg, fp_name=fp_name)  # type: ignore[arg-type, misc]

    def forward(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    def _all_observers(self):
        """
        PTQWrapper itself owns NO observers (transparent node).
        Returning an empty iterator prevents double-processing when parents
        traverse the tree and then recurse into `self.wrapped`.
        """
        return ()  # no local observers

    def named_observers(self):
        """
        Proxy to the wrapped module so debugging tools can still enumerate observers.
        """
        yield from self.wrapped.named_observers()

    def get_observer(self, name: str):
        """
        Proxy to the wrapped module for direct lookup by name.
        """
        return self.wrapped.get_observer(name)

    def extra_repr(self) -> str:
        return self.wrapped.extra_repr()
