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

    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Wrap the tree (or single module) according to strictness policy
        model = self._wrap_supported(model, self.qcfg)

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

    def _wrap_supported(
        self,
        root: nn.Module,
        qcfg: PTQConfig,
    ) -> nn.Module:
        """
        Recursively attempt to wrap boundaries. Strictness is applied at every boundary.
        """
        assert not isinstance(root, QuantModuleBase), "The module is already wrapped."

        # Case A: HuggingFace-style transformers: model.model.layers
        lm = getattr(root, "model", None)
        layers = getattr(lm, "layers", None) if isinstance(lm, nn.Module) else None
        if isinstance(layers, nn.ModuleList):
            new_list = nn.ModuleList()
            for idx, layer in enumerate(layers):
                child_scope = f"layer{idx}"
                child_cfg = qcfg.child(child_scope)

                # Enforce strictness at the child boundary
                wrapped = self._try_wrap(
                    layer,
                    child_cfg,
                    fp_name=child_scope,
                    raise_on_fail=self.strict_wrap,
                )
                new_list.append(wrapped)
            lm.layers = new_list  # type: ignore[union-attr]
            return root

        # Case B: Containers
        if isinstance(root, (nn.Sequential, nn.ModuleList)):
            for i, child in enumerate(list(root)):
                name = str(i)
                child_cfg = qcfg.child(name)

                wrapped = self._try_wrap(
                    child, child_cfg, fp_name=name, raise_on_fail=self.strict_wrap
                )
                if wrapped is child:
                    assert not self.strict_wrap
                    wrapped = self._wrap_supported(wrapped, child_cfg)
                root[i] = wrapped  # type: ignore[index]

        if isinstance(root, nn.ModuleDict):
            for k, child in list(root.items()):
                name = k
                child_cfg = qcfg.child(name)

                wrapped = self._try_wrap(
                    child, child_cfg, fp_name=name, raise_on_fail=self.strict_wrap
                )
                if wrapped is child:
                    assert not self.strict_wrap
                    wrapped = self._wrap_supported(wrapped, child_cfg)
                root[k] = wrapped  # type: ignore[index]

        # Case C: Leaf node
        root_name = getattr(root, "_get_name", lambda: None)()
        wrapped = self._try_wrap(
            root, qcfg, fp_name=root_name, raise_on_fail=self.strict_wrap
        )
        if wrapped is not root:
            return wrapped

        assert not self.strict_wrap
        # Case D: Named children
        for name, child in list(root.named_children()):
            child_cfg = qcfg.child(name)

            wrapped = self._try_wrap(
                child, child_cfg, fp_name=name, raise_on_fail=self.strict_wrap
            )
            if wrapped is child:
                assert not self.strict_wrap
                wrapped = self._wrap_supported(wrapped, child_cfg)
            setattr(root, name, wrapped)

        return root

    def _try_wrap(
        self,
        module: nn.Module,
        qcfg_for_child: PTQConfig,
        *,
        fp_name: Optional[str],
        raise_on_fail: bool,
    ) -> nn.Module:
        """
        Attempt to wrap a boundary with PTQWrapper.

        Behavior:
          • If PTQWrapper succeeds: return wrapped module.
          • If PTQWrapper raises NotImplementedError:
                - raise_on_fail=True  -> re-raise (strict)
                - raise_on_fail=False -> return original module (permissive)
        """
        try:
            return PTQWrapper(module, qcfg=qcfg_for_child, fp_name=fp_name)
        except NotImplementedError as e:
            if raise_on_fail:
                raise NotImplementedError(
                    f"PTQQuantizer: no quantization wrapper for {type(module).__name__}"
                ) from e
            return module
