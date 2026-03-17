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

from typing import Optional

import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class PTQWrapHelper:
    """
    Reusable helper that applies PTQWrapper recursively according to PTQConfig.

    This class contains only the structural wrapping logic and can be reused by
    other algorithms (e.g. SpinQuant) that want to leverage the same wrapper
    hierarchy without enabling calibration immediately.
    """

    def __init__(self, *, strict_wrap: bool = True):
        self.strict_wrap = strict_wrap

    def wrap_supported(
        self,
        root: nn.Module,
        qcfg: PTQConfig,
    ) -> nn.Module:
        """
        Recursively attempt to wrap boundaries. Strictness is applied at every boundary.
        """
        assert not isinstance(root, QuantModuleBase), "The module is already wrapped."

        try:
            return PTQWrapper(root, qcfg=qcfg, fp_name="model")
        except NotImplementedError:
            print(
                f"No specialized wrapper found for {type(root).__name__}; applying recursive wrapping."
            )

        # Case A: HuggingFace-style transformers: model.model.layers
        lm = getattr(root, "model", None)

        embeddings = (
            getattr(lm, "embed_tokens", None) if isinstance(lm, nn.Module) else None
        )
        if isinstance(embeddings, nn.Module):
            child_scope = "model.embeddings"
            child_cfg = qcfg.child(child_scope)
            wrapped = self.try_wrap(
                embeddings,
                child_cfg,
                fp_name=child_scope,
                raise_on_fail=self.strict_wrap,
            )
            lm.embed_tokens = wrapped  # type: ignore[union-attr]

        model_norm = getattr(lm, "norm", None) if isinstance(lm, nn.Module) else None
        if isinstance(model_norm, nn.Module):
            child_scope = "model.norm"
            child_cfg = qcfg.child(child_scope)
            wrapped = self.try_wrap(
                model_norm,
                child_cfg,
                fp_name=child_scope,
                raise_on_fail=self.strict_wrap,
            )
            lm.norm = wrapped  # type: ignore[union-attr]

        lm_head = getattr(root, "lm_head", None)
        if isinstance(lm_head, nn.Module):
            child_scope = "lm_head"
            child_cfg = qcfg.child(child_scope)
            wrapped = self.try_wrap(
                lm_head,
                child_cfg,
                fp_name=child_scope,
                raise_on_fail=self.strict_wrap,
            )
            root.lm_head = wrapped  # type: ignore[attr-defined]

        layers = getattr(lm, "layers", None) if isinstance(lm, nn.Module) else None
        if isinstance(layers, nn.ModuleList):
            new_list = nn.ModuleList()
            for idx, layer in enumerate(layers):
                child_scope = f"layer{idx}"
                child_cfg = qcfg.child(child_scope)

                wrapped = self.try_wrap(
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

                wrapped = self.try_wrap(
                    child,
                    child_cfg,
                    fp_name=name,
                    raise_on_fail=self.strict_wrap,
                )
                if wrapped is child:
                    assert not self.strict_wrap
                    wrapped = self.wrap_supported(wrapped, child_cfg)
                root[i] = wrapped  # type: ignore[index]
            return root

        if isinstance(root, nn.ModuleDict):
            for k, child in list(root.items()):
                name = k
                child_cfg = qcfg.child(name)

                wrapped = self.try_wrap(
                    child,
                    child_cfg,
                    fp_name=name,
                    raise_on_fail=self.strict_wrap,
                )
                if wrapped is child:
                    assert not self.strict_wrap
                    wrapped = self.wrap_supported(wrapped, child_cfg)
                root[k] = wrapped  # type: ignore[index]
            return root

        # Case C: Leaf node
        root_name = getattr(root, "_get_name", lambda: None)()
        wrapped = self.try_wrap(
            root,
            qcfg,
            fp_name=root_name,
            raise_on_fail=self.strict_wrap,
        )
        if wrapped is not root:
            return wrapped

        assert not self.strict_wrap

        # Case D: Named children
        for name, child in list(root.named_children()):
            child_cfg = qcfg.child(name)

            wrapped = self.try_wrap(
                child,
                child_cfg,
                fp_name=name,
                raise_on_fail=self.strict_wrap,
            )
            if wrapped is child:
                wrapped = self.wrap_supported(wrapped, child_cfg)
            setattr(root, name, wrapped)

        return root

    def try_wrap(
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
          - If PTQWrapper succeeds: return wrapped module.
          - If PTQWrapper raises NotImplementedError:
              * raise_on_fail=True  -> re-raise
              * raise_on_fail=False -> return original module
        """
        try:
            return PTQWrapper(module, qcfg=qcfg_for_child, fp_name=fp_name)
        except NotImplementedError as e:
            if raise_on_fail:
                raise NotImplementedError(
                    f"PTQWrapHelper: no quantization wrapper for {type(module).__name__}"
                ) from e
            return module
