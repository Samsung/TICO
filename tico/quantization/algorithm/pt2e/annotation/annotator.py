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

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch.fx
    from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor
import torch
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.quantizer import QuantizationSpec, Quantizer
from torch.ao.quantization.quantizer.utils import _get_module_name_filter

from tico.quantization.algorithm.pt2e.annotation.op import *
import tico.quantization.algorithm.pt2e.annotation.spec as annot_spec
import tico.quantization.algorithm.pt2e.annotation.utils as annot_utils
import tico.quantization.algorithm.pt2e.utils as quant_utils
from tico.quantization.algorithm.pt2e.annotation.config import QuantizationConfig
from tico.quantization.algorithm.pt2e.transformation.convert_scalars_to_attrs import (
    convert_scalars_to_attrs,
)


class PT2EAnnotator(Quantizer):
    """
    The class annotates quantization configurations on each nodes.

    Observers would be attached according to those configurations in
     'torch.prepare_pt2e'.
    """

    def __init__(self):
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
        self.operator_type_config: Dict[
            torch._ops.OpOverloadPacket, QuantizationConfig
        ] = {}
        self.module_type_config: Dict[Callable, QuantizationConfig] = {}
        self.module_name_config: Dict[str, QuantizationConfig] = {}

    def set_global(self, quantization_config: QuantizationConfig) -> PT2EAnnotator:
        """
        Set quantization config globally.
        """
        assert quantization_config is not None
        self.global_config = quantization_config
        return self

    def set_operator_type(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: QuantizationConfig,
    ) -> PT2EAnnotator:
        """
        Set quantization config for given operator type.
        """
        assert quantization_config is not None
        self.operator_type_config[operator_type] = quantization_config
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: QuantizationConfig
    ):
        """
        Set quantization config for given module type.

        For example, let's say quantizer.set_module_type(nn.Linear).
        It will quantize all 'nn.Linear' modules with the `quantization_config`.
        """
        assert quantization_config is not None
        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(
        self, module_name: str, quantization_config: QuantizationConfig
    ):
        """
        Set quantization config for given module name.

        For example, let's say quantizer.set_module_name("blocks.sub").
        It will quantize all nodes that come from a module whose name is "blocks.sub"
         with the `quantization_config`.
        """
        assert quantization_config is not None
        self.module_name_config[module_name] = quantization_config
        return self

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Allows for user defined transforms to run before annotating the graph."""
        model = convert_scalars_to_attrs(model)
        return model

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        model = self._annotate_for_quantization(model)
        annot_utils.propagate_annotation_forward(model)
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        # TODO Consider this method.
        pass

    def _annotate_by_config_and_filter(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
    ) -> torch.fx.GraphModule:
        assert quantization_config is not None

        for node in model.graph.nodes:
            if node.target not in annot_spec.OP_TO_ANNOTATOR:
                continue
            annot_spec.OP_TO_ANNOTATOR[node.target](
                model, node, quantization_config, filter_fn
            )
        return model

    def _annotate_for_quantization(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        # Annotate according to the given module names.
        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_by_config_and_filter(
                model, config, _get_module_name_filter(module_name)
            )

        # Annotate according to the given module types.
        tp_list = list(self.module_type_config.keys())
        for module_type, config in self.module_type_config.items():
            self._annotate_by_config_and_filter(
                model, config, quant_utils.get_module_type_filter(module_type)
            )

        # TODO Annotate according to the given operator types.

        self._annotate_by_config_and_filter(
            model,
            self.global_config,
            quant_utils.get_not_module_type_or_name_filter(tp_list, module_name_list),
        )
        return model


@functools.lru_cache
def get_asymmetric_quantization_config(
    weight_is_per_channel: bool = True,
    act_qmin: int = 0,
    act_qmax: int = 255,
    weight_qmin: int = 0,
    weight_qmax: int = 255,
) -> QuantizationConfig:
    # activation
    act_extra_args: Dict[str, Any] = {"eps": 2**-12}
    act_observer = MinMaxObserver
    act_qspec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=act_qmin,
        quant_max=act_qmax,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
        observer_or_fake_quant_ctr=act_observer.with_args(
            **act_extra_args,
        ),
    )
    # weight
    weight_extra_args: Dict[str, Any] = {"eps": 2**-12}
    weight_qscheme = (
        torch.per_channel_affine if weight_is_per_channel else torch.per_tensor_affine
    )
    weight_observer: _ObserverOrFakeQuantizeConstructor = (
        PerChannelMinMaxObserver if weight_is_per_channel else MinMaxObserver
    )
    weight_qspec = QuantizationSpec(
        dtype=torch.uint8,
        quant_min=weight_qmin,
        quant_max=weight_qmax,
        qscheme=weight_qscheme,
        ch_axis=0,
        is_dynamic=False,
        observer_or_fake_quant_ctr=weight_observer.with_args(**weight_extra_args),
    )

    # Set bias qspec in each annotation functions.
    bias_qspec = None
    quantization_config = QuantizationConfig(
        act_qspec,
        act_qspec,
        weight_qspec,
        bias_qspec,
    )
    return quantization_config
