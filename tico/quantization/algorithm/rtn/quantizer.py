# Copyright (c) 2024 Intel Corporation
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

import types
from typing import Any, Callable, Dict, List, Optional

import torch
from tico.quantization.algorithm.gptq.quant import quantize, Quantizer

from tico.quantization.algorithm.gptq.utils import (
    find_layers,
    gather_single_batch_from_dict,
    gather_single_batch_from_list,
)
from tico.quantization.config.rtn import RTNConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import register_quantizer
from tqdm.auto import tqdm


@register_quantizer(RTNConfig)
class RTNQuantizer(BaseQuantizer):
    """
    Quantizer for applying the GPTQ algorithm (typically for weight quantization).
    This implementation expects:
        1) prepare(model, ...) to only attach hooks/Catchers and NOT run the model internally.
        2) The user runs the model with arbitrary number of batches to collect calibration data.
        3) convert(model) to consume the collected data and apply GPTQ.
    """

    def __init__(self, config: RTNConfig):
        super().__init__(config)

    @torch.no_grad()
    def prepare(
        self,
        model: torch.nn.Module,
        args: Optional[Any] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        return model

    @torch.no_grad()
    def convert(self, model):
        rtn_conf = self.config
        assert isinstance(rtn_conf, RTNConfig)

        # Disable use_cache during calibration
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            orig_use_cache = model.config.use_cache
            model.config.use_cache = False
        else:
            orig_use_cache = None

        # Identify layers
        if hasattr(model, "model"):
            target_layers = model.model.layers
        else:
            target_layers = [model]

        quantizers: Dict[str, Any] = {}
        for l_idx, layer in enumerate(
            tqdm(
                target_layers,
                desc="Quantizing layers",
                unit="layer",
                disable=not rtn_conf.show_progress,
            )
        ):
            # 1) Identify quantizable submodules within the layer
            full = find_layers(
                layer, layers=[torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d]
            )
            sequential = [list(full.keys())]

            # 2) Set up
            for names in sequential:
                subset = {n: full[n] for n in names}

                # 3) Quantize each submodule
                for name in subset:
                    layer = subset[name]
                    quantizer = Quantizer()
                    quantizer.configure(bits=4, perchannel=True, sym=False, mse=False)

                    W = layer.weight.data.clone()
                    if isinstance(layer, torch.nn.Conv2d) or isinstance(
                        layer, torch.nn.Conv1d
                    ):
                        W = W.flatten(1)
                    W = W.float()
                    quantizer.find_params(W, weight=True)
                    Q = quantize(W, quantizer.scale, quantizer.zero, quantizer.maxq)

                    quantizers[f"model.layers.{l_idx}.{name}"] = quantizer
                    layer.weight.data = Q.reshape(layer.weight.shape).to(
                        layer.weight.data.dtype
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Restore the original cache configuration.
        if orig_use_cache is not None:
            model.config.use_cache = orig_use_cache

        model.quantizers = quantizers

        return model
