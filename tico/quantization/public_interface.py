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
from typing import Any, Dict, Optional

import torch

from tico.quantization.algorithm.gptq.quantizer import GPTQQuantizer
from tico.quantization.config.base import BaseConfig
from tico.quantization.quantizer import BaseQuantizer
from tico.quantization.quantizer_registry import get_quantizer

QUANTIZER_ATTRIBUTE_NAME = "tico_quantizer"


def prepare(
    model: torch.nn.Module,
    quant_config: BaseConfig,
    args: Optional[Any] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    inplace: Optional[bool] = True,
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
    quantizer = get_quantizer(quant_config)

    model = model if inplace else copy.deepcopy(model)

    model = quantizer.prepare(model, args, kwargs)
    setattr(model, QUANTIZER_ATTRIBUTE_NAME, quantizer)

    return model


def convert(model, inplace: Optional[bool] = True):
    """
    Convert a model returned by prepare() using its attached quantizer.

    Retrieves the ``tico_quantizer`` installed by prepare(), reusing its existing
    configuration and collected calibration state. PTQ wrappers retain their
    resolved quantization policies. This function does not accept ``quant_config``
    or select or instantiate a new quantizer.

    The converted result loses its quantizer handle only after conversion succeeds.
    On failure, the original prepared model retains its handle for retry; arbitrary
    in-place mutations made by the algorithm are not rolled back by this API.

    Parameters:
        model: The model returned by prepare(), calibrated as required by its algorithm.
        inplace (bool, optional): Defaults to True, converting the model in place.
            For algorithms supporting False, converts a copy and retains the original
            prepared model's quantizer handle. GPTQ supports only inplace=True.

    Returns:
        The converted model without a quantizer handle. With supported inplace=False
        conversion, this is a copy; the original prepared model retains its handle.
    """
    # Get quantizer first before calling deepcopy that does not copy attributes properly.
    if hasattr(model, QUANTIZER_ATTRIBUTE_NAME):
        quantizer = getattr(model, QUANTIZER_ATTRIBUTE_NAME)
    else:
        raise RuntimeError("Call prepare() function first.")

    # deepcopy prevents the quantizer from restoring the catcher used for calibration.
    # TODO Revisit `inplace` policy.
    if isinstance(quantizer, GPTQQuantizer) and not inplace:
        raise RuntimeError(
            "GPTQ quantization only supports `in-place=True`. Please set 'inplace=True' to proceed."
        )

    # The result copy must not duplicate quantizer-owned calibration caches.
    # Sharing this metadata during the copy also preserves the original handle.
    model = model if inplace else copy.deepcopy(model, memo={id(quantizer): quantizer})

    assert isinstance(quantizer, BaseQuantizer)
    model = quantizer.convert(model)
    if hasattr(model, QUANTIZER_ATTRIBUTE_NAME):
        delattr(model, QUANTIZER_ATTRIBUTE_NAME)

    return model
