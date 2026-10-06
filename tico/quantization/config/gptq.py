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

from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING, Union

import torch
import torch.nn as nn

from tico.quantization.config.base import BaseConfig

if TYPE_CHECKING:
    from tico.quantization.algorithm.universal_gptq.quantizer import (
        GPTQFactory,
        GPTQProtocol,
    )


GPTQFactoryArg = Union["GPTQFactory", str, None]
"""
Type alias for GPTQ factory argument.

Accepts either:
- A factory callable: `lambda layer: GPTQ(layer)`
- A fully-qualified class path string: `"tico.quantization.algorithm.qwen3_vl_gptq.gptq.GPTQ"`
- None (uses default classic GPTQ v1)
"""


@dataclass
class GPTQConfig(BaseConfig):
    """
    Configuration for GPTQ weight quantization.

    Attributes
    ----------
    weight_bits : int
        Default bit-width applied to quantized weights.
    weight_bits_overrides : dict[str, int]
        Optional per-module bit-width overrides.

        Supported keys are matched in the following order:
          1) Full module name, for example `model.layers.0.self_attn.o_proj`
          2) Layer-local module name, for example `self_attn.o_proj`
          3) Full-name suffix, for example `self_attn.o_proj` or `down_proj`

        This makes it possible to keep a default bit-width for most modules
        while selectively increasing precision for specific projections.
    quantize_lm_head : bool
        Whether to apply GPTQ to the language-model output head. This option
        is disabled by default because many language models tie
        `lm_head.weight` with the input embedding table, and quantizing the
        head can modify the shared embedding weights.
    """

    # general
    verbose: bool = False
    show_progress: bool = True

    # model-specific quantization switches
    quantize_lm_head: bool = False

    # quantizer.configure params (weight quantization spec)
    weight_bits: int = 8
    weight_bits_overrides: dict[str, int] = field(default_factory=dict)
    perchannel: bool = True
    symmetric: bool = False
    mse: str | None = None
    sensitivity: dict[str, torch.Tensor] | None = None

    # GPTQ.fasterquant params (algorithm hyperparams)
    percdamp: float = 0.01
    groupsize: int = -1
    actorder: bool = True
    static_groups: bool = False

    # use this option to stabilize GPTQ for deep models
    use_orig_model_inference: bool = False

    @property
    def name(self) -> str:
        return "gptq"

    def validate(self) -> None:
        if not isinstance(self.quantize_lm_head, bool):
            raise TypeError(
                f"quantize_lm_head must be bool. got {type(self.quantize_lm_head)}"
            )
        if self.weight_bits <= 0:
            raise ValueError(f"weight_bits must be positive. got {self.weight_bits}")
        for module_name, bits in self.weight_bits_overrides.items():
            if bits <= 0:
                raise ValueError(
                    f"weight_bits_overrides[{module_name!r}] must be positive. got {bits}"
                )
        if self.groupsize != -1 and self.groupsize <= 0:
            raise ValueError(f"groupsize must be -1 or positive. got {self.groupsize}")
        if not (0.0 < self.percdamp <= 1.0):
            raise ValueError(f"percdamp must be in (0, 1]. got {self.percdamp}")


@dataclass
class UniversalGPTQConfig(GPTQConfig):
    """
    Configuration for universal GPTQ quantizer.

    This config class is identical to GPTQConfig but maps to the universal
    GPTQ quantizer implementation, which works with any PyTorch model without
    requiring model-specific wrapper classes.

    Inherits all GPTQConfig options:
        - weight_bits, weight_bits_overrides
        - perchannel, symmetric, mse
        - percdamp, groupsize, actorder, static_groups
        - verbose, show_progress

    Usage example:
        from tico.quantization.config.gptq import UniversalGPTQConfig
        from tico.quantization import prepare, convert

        config = UniversalGPTQConfig(
            weight_bits=8,
        )
        model = prepare(model, config)
        # ... calibrate ...
        model = convert(model)
    """

    # Retain gptq_data attributes after quantization finishes (to explore them in debugger)
    debug_mode: bool = False

    # Disable GPTQ quantization of modules that are called multiple times in a single batch
    ignore_multi_call_modules: bool = True

    # Regex patterns specifying full hierarchical module names of modules
    # that should cache their outputs for performance optimization during model replay.
    # Recommendation: enable caching for relatively large repeating blocks of the model,
    # e.g. decoder layers (something like "model\.language_model\.decoder_layers\.[0-9]+").
    # If you also want to allow caching of the large blocks' submodules add ".*" at the end
    # e.g. "model\.language_model\.decoder_layers\.[0-9]+.*". In this case you also need to
    # enable calls between cacheable modules by setting `allow_calls_between_cacheable_modules` to `True`
    cacheable_modules: list[str] = field(default_factory=list)

    # Allow parent cacheable module calls to its cacheable submodules.
    # Switching this option may affect performance depending on specific model.
    allow_calls_between_cacheable_modules: bool = True

    # Factory function for creating GPTQ instances.
    # Allows using different GPTQ implementations (GPTQ v1, GPTQv2, etc.)
    # with the UniversalGPTQQuantizer.
    #
    # Accepts either:
    # - None: Uses default classic GPTQ v1 (lambda layer: GPTQ(layer))
    # - str: Fully-qualified class path, e.g., "tico.quantization.algorithm.qwen3_vl_gptq.gptq.GPTQ"
    # - Callable: Factory function, e.g., lambda layer: GPTQ(layer, normalize_H=True)
    #
    # Default: None (uses classic GPTQ v1).
    gptq_factory: GPTQFactoryArg = None

    # GPTQ (v1) requires collecting only inputs from upstream quantized layers: collect_native_inputs=False.
    # GPTAQ (GPTQ v2) requires collecting the "native" inputs from the original unquantized model: collect_native_inputs=True.
    # Set this flag to True if you are going to use GPTAQ (GPTQ v2).
    # Note that in this case you should also use the appropriate gptq_factory supporting GPTAQ.
    collect_native_inputs: bool = False

    @property
    def name(self) -> str:
        return "universal_gptq"

    def validate(self) -> None:
        # First validate parent class
        super().validate()

        if not isinstance(self.debug_mode, bool):
            raise TypeError(f"debug_mode must be bool. got {type(self.debug_mode)}")

        if not isinstance(self.ignore_multi_call_modules, bool):
            raise TypeError(
                f"ignore_multi_call_modules must be bool. got {type(self.ignore_multi_call_modules)}"
            )

        if not isinstance(self.cacheable_modules, list):
            raise TypeError(
                f"cacheable_modules must be list. got {type(self.cacheable_modules)}"
            )

        if not isinstance(self.allow_calls_between_cacheable_modules, bool):
            raise TypeError(
                f"allow_calls_between_cacheable_modules must be bool. got {type(self.allow_calls_between_cacheable_modules)}"
            )

        if not isinstance(self.collect_native_inputs, bool):
            raise TypeError(
                f"collect_native_inputs must be bool. got {type(self.collect_native_inputs)}"
            )

        # gptq_factory is optional - if provided, it must be str or callable
        if self.gptq_factory is not None:
            if not isinstance(
                self.gptq_factory, (str, Callable)  # type: ignore[arg-type]
            ):
                raise TypeError(
                    f"gptq_factory must be str, callable, or None. got {type(self.gptq_factory)}"
                )

            if isinstance(self.gptq_factory, str) and not self.gptq_factory.strip():
                raise ValueError("gptq_factory string cannot be empty")

        # collect_native_inputs=True requires allow_calls_between_cacheable_modules=True
        # collect_native_inputs=True requires using GPTAQ (GPTQ v2) compatible gptq_factory,
        # specifically, GPTQ instances created with such factory must have 'native_inp' attribute.
        if self.collect_native_inputs:
            if not self.allow_calls_between_cacheable_modules:
                raise ValueError(
                    "collect_native_inputs=True requires allow_calls_between_cacheable_modules to also be True."
                )

            gptq_factory: "GPTQFactory | None" = self.resolve_gptq_factory()
            if gptq_factory is None:
                raise ValueError(
                    "collect_native_inputs=True requires a GPTQ factory that supports native inputs. "
                    "Please provide gptq_factory parameter."
                )

            gptq: "GPTQProtocol" = gptq_factory(
                nn.Linear(1, 1, bias=False, device="cpu")
            )
            if gptq is None:
                raise ValueError(
                    "could not create a GPTQ instance using specified gptq_factory."
                )

            if not hasattr(gptq, "native_inp"):
                raise ValueError(f"{type(gptq)} does not have 'native_inp' attribute.")

        # use_orig_model_inference is incompatible with frontier-based execution
        if self.use_orig_model_inference:
            raise ValueError(
                "use_orig_model_inference=True is incompatible with UniversalGPTQConfig. "
                "The universal quantizer uses a frontier-based execution strategy where "
                "downstream modules necessarily receive quantized outputs from upstream "
                "modules during replay. If you need this feature, use GPTQConfig with "
                "the layer-by-layer quantizer instead."
            )

    def resolve_gptq_factory(self) -> "GPTQFactory | None":
        """
        Resolve gptq_factory from string path to callable.

        Returns:
            GPTQFactory if gptq_factory is set, None otherwise.

        Raises:
            ImportError: If the specified class cannot be imported.
            AttributeError: If the specified class does not exist in the module.
        """
        if self.gptq_factory is None:
            return None

        if callable(self.gptq_factory):
            return self.gptq_factory

        if isinstance(self.gptq_factory, str):
            return _create_factory_from_class_path(self.gptq_factory)

        # Should never reach here due to validation
        raise TypeError(
            f"gptq_factory must be str, callable, or None. "
            f"got {type(self.gptq_factory)}"
        )


def _create_factory_from_class_path(class_path: str) -> "GPTQFactory":
    """
    Create a GPTQ factory from a fully-qualified class path.

    Example:
        >>> factory = _create_factory_from_class_path(
        ...     "tico.quantization.algorithm.qwen3_vl_gptq.gptq.GPTQ"
        ... )
        >>> gptq = factory(nn.Linear(10, 10))

    Args:
        class_path: Fully-qualified Python class path, e.g.,
            "tico.quantization.algorithm.qwen3_vl_gptq.gptq.GPTQ"

    Returns:
        A factory function that takes a layer module and returns a GPTQ instance.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    import importlib

    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    gptq_class = getattr(module, class_name)

    return lambda layer: gptq_class(layer)
