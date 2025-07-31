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

import copy
from typing import Any, Dict, Optional, Type

import torch

from tico.experimental.quantization.algorithm.gptq.quantizer import GPTQQuantizer
from tico.experimental.quantization.algorithm.pt2e.quantizer import PT2EQuantizer
from tico.experimental.quantization.algorithm.smoothquant.quantizer import (
    SmoothQuantQuantizer,
)
from tico.experimental.quantization.config import BaseConfig
from tico.experimental.quantization.quantizer import BaseQuantizer


config_to_quantizer: Dict[str, Type[BaseQuantizer]] = {
    "pt2e": PT2EQuantizer,
    "gptq": GPTQQuantizer,
    "smooth_quant": SmoothQuantQuantizer,
}

QUANTIZER_ATTRIBUTE_NAME = "tico_quantizer"


def prepare(
    model: torch.nn.Module,
    quant_config: BaseConfig,
    args: Optional[Any] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    inplace: Optional[bool] = False,
):
    """
    Prepare the model for quantization using the provided configuration.

    Determines the appropriate quantizer based on the type of `quant_config` and
    prepares the model accordingly.

    Parameters:
        model: The PyTorch model to be quantized.
        quant_config (BaseConfig): The quantization configuration.
        args (Any, optional): Positional example inputs required for activation quantization.
        kwargs (Dict[str, Any], optional): Keyword example inputs required for activation quantization.
        inplace (bool, optional): If true, the model will be modified in place;
                                   otherwise, a new prepared model is returned.

    Returns:
        The model prepared for quantization.
    """
    if hasattr(model, QUANTIZER_ATTRIBUTE_NAME):
        raise RuntimeError("prepare() already has been called.")
    if quant_config.name == "pt2e" and inplace:
        raise RuntimeError(
            "In-place is not supported for PT2E quantization due to limitation in the underlying Torch APIs. Please set 'inplace=False' to proceed."
        )

    model = model if inplace else copy.deepcopy(model)

    quantizer = config_to_quantizer[quant_config.name](quant_config)
    model = quantizer.prepare(model, args, kwargs)
    setattr(model, QUANTIZER_ATTRIBUTE_NAME, quantizer)

    return model


def convert(model, inplace: Optional[bool] = False):
    """
    Convert the prepared model to a quantized model using the provided configuration.

    Determines the appropriate quantizer based on the type of quant_config and
    converts the model accordingly.

    Parameters:
        model: The prepared PyTorch model.
        inplace (bool, optional): If true, the model will be modified in place;
                                   otherwise, a new prepared model is returned.

    Returns:
        The quantized model.
    """
    # Get quantizer first before calling deepcopy that does not copy attributes properly.
    if hasattr(model, QUANTIZER_ATTRIBUTE_NAME):
        quantizer = getattr(model, QUANTIZER_ATTRIBUTE_NAME)
        delattr(model, QUANTIZER_ATTRIBUTE_NAME)
    else:
        raise RuntimeError("Call prepare() function first.")

    if isinstance(quantizer, PT2EQuantizer) and inplace:
        raise RuntimeError(
            "In-place is not supported for PT2E quantization due to limitation in the underlying Torch APIs. Please set 'inplace=False' to proceed."
        )
    # deepcopy prevents the quantizer from restoring the catcher used for calibration.
    # TODO Revisit `inplace` policy.
    if isinstance(quantizer, GPTQQuantizer) and not inplace:
        raise RuntimeError(
            "GPTQ quantization only supports `in-place=True`. Please set 'inplace=True' to proceed."
        )

    model = model if inplace else copy.deepcopy(model)

    assert isinstance(quantizer, BaseQuantizer)
    model = quantizer.convert(model)

    return model
