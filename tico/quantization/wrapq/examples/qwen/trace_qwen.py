#!/usr/bin/env python3
# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
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

"""
Trace, debug, and validate quantized Qwen3VL models.

This script traces tensor flow through Qwen3VLForConditionalGeneration submodules,
comparing outputs between the original (unquantized) and quantized models to help
identify quantization issues.

Usage Examples
-----
Basic usage - print all modules' inputs and outputs and compare them:

    python trace_qwen.py --model ~/models/qwen3-vl-2b

Don't print outputs, only compare them:

    python trace_qwen.py --model ~/models/qwen3-vl-2b \\
        --no-trace-unquantized --no-trace-quantized

Detailed examination of specific submodules:

    python trace_qwen.py --model ~/models/qwen3-vl-2b \\
        --interesting-modules model.language_model model.visual

Enable debugging on specific submodules:

    python trace_qwen.py --model ~/models/qwen3-vl-2b \\
        --interesting-modules model.language_model \\
        --breakpoint-on-interesting-modules

Command-line Arguments
----------------------
--model : str (required)
    HuggingFace repo name (e.g., "Qwen/Qwen3-VL-2B-Instruct") or local path
    to cached model directory (e.g., ~/models/qwen3-vl-2b).

--cache-dir : str (optional)
    Optional cache directory for downloaded models.

--hf-token : str (optional)
    Optional HuggingFace token for gated/private repositories.

--interesting-modules : list[str] (optional)
    Space-separated list of module names to inspect in detail. For these modules,
    actual tensor elements are printed (not just statistics).

--breakpoint-on-interesting-modules : flag
    Switch to PDB debug mode when encountering interesting modules. Allows
    examination of stack trace and program state.

--no-trace-unquantized : flag
    Don't print input/output traces for the unquantized model.

--no-trace-quantized : flag
    Don't print input/output traces for the quantized model.

--no-side-by-side : flag
    Don't perform side-by-side comparison between quantized and unquantized models.

--enable-quantization : flag
    By default fake quantization operations are disabled in the 'quantized' model.
    So, such 'quantized' model must show a close-to-zero output divergence from the original (unquantized) model
    - this can be used as the criterion of validity of 'quantized' model's internal logic.
    Opposed to that, the flag --enable-quantization enables fake quatization operations in the 'quantized' model
    and thus allows for examining the error introduced by fake quantization operations.

--dtype : str (optional)
    Quantization data type (uint4, int4, int8, uint8, int16). uint8 is the default.

Output Format
-------------
The script always prints the generated model inputs (input_ids, attention_mask,
pixel_values, image_grid_thw) with their shapes and dtypes.

Then the script prints submodule trace for the original (unquantized) model and the quantized one.
For each submodule, the trace includes:
    - module_name: Fully qualified name (e.g., "model.language_model.embed_tokens")
    - module_type: Class name (e.g., "Embedding")
    - inputs: Tensor shapes, dtypes, and statistics (mean, min, max, stddev)
    - kwargs: Named arguments to the submodule's forward method
    - output: Tensor shapes, dtypes, and statistics

Side-by-side comparison shows the difference between unquantized and quantized
outputs for each submodule, helping identify where quantization errors occur.

Implementation Notes
--------------------
The script uses PyTorch's forward hook mechanism to intercept module inputs/outputs
during inference. Two models are probed: the original model and the quantized model.
Outputs are stored in dictionaries keyed by module name for comparison.

Large differences in the side-by-side comparison can indicate quantization issues
that may need investigation.
"""

import json
import os
import sys
from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Number
from typing import Any, Callable, NamedTuple

import torch
import torch.nn as nn

from transformers import AutoProcessor
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.wrapq.dtypes import DType, INT16, INT4, INT8, UINT4, UINT8
from tico.quantization.wrapq.utils.introspection import build_fqn_map
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

# Names exposed to wildcard imports from this module
__all__ = [
    # Model preparation
    "prepare_inputs",
    "prepare_config",
    "prepare_quantized_model",
    # Data structures
    "TensorStatistics",
    "ModuleInputOutput",
    # Core tracing
    "trace_model_input_output",
    "module_hook",
    # Comparison utilities
    "compare_outputs",
    "compare_side_by_side",
    # Tensor utilities
    "get_tensor_statistics",
    "detach_tensors",
    "model_output_to_serializable",
    "full_tensor_printing",
]

# Type aliases (for more descriptive type hints)
ArgName = str
ArgValue = Any
ModuleOutput = Any
ModelNameOrPath = str
ModuleName = str
DirPath = str


DTYPE_MAP = {
    "uint4": UINT4,
    "int4": INT4,
    "int8": INT8,
    "uint8": UINT8,
    "int16": INT16,
}


def build_vlm_inputs(
    processor,
    image,
    question: str,
    return_tensors: str = "pt",
    max_seq_len: int | None = None,
) -> dict[ArgName, torch.Tensor]:
    """
    Build processor inputs for a single image-question example.

    Args:
        processor: Hugging Face multimodal processor.
        image: Input image object accepted by the processor.
        question: User question associated with the image.
        return_tensors: Tensor format requested from the processor (default='pt' which means PyTorch tensor format).
        max_seq_len: Optional maximum text sequence length. If provided,
                     text inputs are truncated to this length.

    Returns:
        A processor output object containing model-ready multimodal inputs.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    processor_kwargs: dict[str, Any] = {
        "text": prompt,
        "images": image,
        "return_tensors": return_tensors,
    }
    if max_seq_len is not None and max_seq_len > 0:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = max_seq_len

    return processor(**processor_kwargs)


def prepare_inputs(
    image_token_id: int,
    vocab_size: int,
    model_name: ModelNameOrPath,
    cache_dir: DirPath | None = None,
    image_width: int = 128,
    image_height: int = 96,
    text_prompt: str = "Describe the image.",
    hf_token: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Prepare model inputs from a zero-filled image and text prompt.

    Creates a zero-filled image tensor and processes it with the text prompt
    using the HuggingFace processor to generate model-ready inputs.

    Args:
        image_token_id: Token ID used for image placeholder tokens.
        vocab_size: Vocabulary size for normalizing input token IDs.
        model_name: HuggingFace model name or local path to the model.
        cache_dir: Optional cache directory for the model/processor.
        image_width: Width of the generated image in pixels (default: 128).
        image_height: Height of the generated image in pixels (default: 96).
        text_prompt: Text prompt to accompany the image (default: "Describe the image.").
        hf_token: Optional HuggingFace token for gated repositories.

    Returns:
        Dictionary containing model inputs (input_ids, attention_mask, pixel_values,
        image_grid_thw) ready for model forward pass.
    """
    # Create a zero-filled image
    image = torch.zeros((3, image_width, image_height), dtype=torch.uint8)

    # Load processor
    processor = AutoProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=cache_dir is not None,
        trust_remote_code=True,
        token=hf_token,
    )

    # Build model inputs
    model_inputs = build_vlm_inputs(
        processor=processor,
        image=image,
        question=text_prompt,
        return_tensors="pt",
        max_seq_len=1024,
    )

    # Normalize input_ids to be consistent with our image_token_id
    image_pad_token_id = 151655
    input_ids: torch.Tensor = model_inputs["input_ids"]
    input_ids[input_ids == image_pad_token_id] = image_token_id

    # Make sure that our input IDs don't go beyond vocabulary size
    input_ids = input_ids % vocab_size
    model_inputs["input_ids"] = input_ids

    return model_inputs


def print_model_inputs(dictionary: dict[ArgName, torch.Tensor]) -> None:
    """
    Print model inputs with their tensor values, shapes, and dtypes.

    Args:
        dictionary: Dictionary mapping argument names to tensor values.
    """
    for arg_name, arg_val in dictionary.items():
        print(f"{arg_name}:")
        lines = str(arg_val).split("\n")
        for line in lines:
            print("    " + line)
        if isinstance(arg_val, torch.Tensor):
            print(f"    shape: {arg_val.shape}")
            print(f"    dtype: {arg_val.dtype}")
        print()


def prepare_config() -> Qwen3VLConfig:
    """
    Create a reduced Qwen3VL model configuration for faster testing.

    Returns a configuration with reduced dimensions:
    - Vision: hidden_size=64, depth=2, num_heads=4
    - Text: hidden_size=64, num_hidden_layers=2, vocab_size=1000

    Returns:
        Qwen3VLConfig with reduced sizes suitable for quick testing.
    """

    cfg = Qwen3VLConfig(
        vision_config={
            "hidden_size": 64,
            "num_heads": 4,
            "depth": 2,  # Number of vision blocks
            "temporal_patch_size": 2,
            "patch_size": 16,
            "out_hidden_size": 64,
            "deepstack_visual_indexes": [0, 1],
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 256,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "num_hidden_layers": 2,  # Number of decoder layers
            "attention_bias": False,
            "attention_dropout": 0.0,
            "max_position_embeddings": 1024,
            "vocab_size": 1000,
            "use_cache": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        image_token_id=998,
        video_token_id=999,
    )
    assert cfg.image_token_id < cfg.text_config.vocab_size
    assert cfg.video_token_id < cfg.text_config.vocab_size

    # Ensure eager attention implementation so outputs are deterministic
    # and do not require GPU flash attention kernels.
    cfg.text_config._attn_implementation = "eager"
    cfg.vision_config._attn_implementation = "eager"

    return cfg


def prepare_quantized_model(
    model: nn.Module,
    model_inputs: dict[str, torch.Tensor],
    enable_quantization: bool,
    dtype: DType = DType.uint(8),
):
    """
    Prepare and calibrate a quantized model.

    Configures PTQ (Post-Training Quantization), prepares the model for
    quantization, runs calibration, and optionally converts to quantized model.

    Args:
        model: The model to quantize.
        model_inputs: Input data for calibration.
        enable_quantization: If True, convert to quantized model after calibration.
        dtype: Quantization data type (uint8, int16, etc.).

    Returns:
        The prepared (and optionally quantized) model.
    """
    # Configure PTQ
    thw = tuple(model_inputs["image_grid_thw"].squeeze().tolist())
    ptq_config = tico.quantization.config.ptq.PTQConfig(
        default_dtype=dtype,
        model_args={
            "vision": {
                "grid_thw": thw,
            }
        },
    )

    # Prepare the model for quantization
    prepared_model = tico.quantization.prepare(
        model, ptq_config, inplace=True  # Transform the model in place
    )

    # Calibrate the model (collect statistics)
    with torch.no_grad():
        prepared_model(**model_inputs)

    if enable_quantization:
        prepared_model = tico.quantization.convert(prepared_model, inplace=True)

    return prepared_model


@contextmanager
def module_hook(
    hook: Callable[
        [nn.Module, tuple[torch.Tensor, ...], dict[str, Any], ModuleOutput], Any
    ]
):
    """
    Context manager for registering a global forward hook on all modules.

    Args:
        hook: Callback function to be called for each module's forward pass.

    Yields:
        None. The hook is automatically removed when exiting the context.
    """
    handle = nn.modules.module.register_module_forward_hook(
        hook, with_kwargs=True, always_call=True
    )
    yield
    handle.remove()


@contextmanager
def full_tensor_printing():
    """
    Context manager for enabling full tensor printing.

    Sets torch print options to 'full' profile to show all tensor elements,
    then restores default settings on exit.
    """
    torch.set_printoptions(profile="full")
    yield
    torch.set_printoptions(profile="default")


class DataMismatchError(Exception):
    ...


def compare_outputs(
    lhs: ModuleOutput,
    rhs: ModuleOutput,
    full_tensor_diff: bool = False,
) -> Number | torch.Tensor | dict[str, Any] | Exception | None:
    """
    Compare two module outputs and compute their difference.

    Recursively compares outputs of various types (tensors, numbers, dicts,
    lists, named tuples, ModelOutput objects) and returns the difference.

    Args:
        lhs: Left-hand side output (from unquantized model).
        rhs: Right-hand side output (from quantized model).
        full_tensor_diff: If True, return full tensor difference instead of statistics.

    Returns:
        The difference between outputs. For tensors, returns statistics or full tensor.
        For containers, returns a dict of differences. Returns None if both are None.

    Raises:
        DataMismatchError: If the types or structure of lhs and rhs don't match.
    """
    if type(lhs) != type(rhs):
        raise DataMismatchError(f"Type mismatch: {type(lhs)} != {type(rhs)}")

    # None
    if lhs is None:
        return None

    # Tensor
    if isinstance(lhs, torch.Tensor):
        if full_tensor_diff:
            return lhs.to(torch.float) - rhs.to(torch.float)
        else:
            abs_delta: torch.Tensor = (lhs - rhs).abs().to(torch.float)
            delta_stats: TensorStatistics = get_tensor_statistics(abs_delta)
            interval = (lhs.max() - lhs.min()).item()
            if interval != 0.0:
                peir = delta_stats.max / interval
                return DifferenceStatistics(**delta_stats._asdict(), peir=peir)
            return delta_stats

    # Number
    if isinstance(lhs, Number):
        return abs(lhs - rhs)

    diff: dict[str, Any] = {}

    # List, Tuple: compare element-wise
    if isinstance(lhs, Sequence):
        if len(lhs) != len(rhs):
            raise DataMismatchError(f"Length mismatch: {len(lhs)} != {len(rhs)}")
        for i, (lhs_val, rhs_val) in enumerate(zip(lhs, rhs)):
            try:
                diff[str(i)] = compare_outputs(lhs_val, rhs_val)
            except Exception as ex:
                diff[str(i)] = ex
        return diff

    # Dict
    if isinstance(lhs, dict):
        dict_keys = lhs.keys()
        for key in dict_keys:
            lhs_val = lhs[key]
            rhs_val = rhs[key]
            try:
                diff[key] = compare_outputs(lhs_val, rhs_val)
            except Exception as ex:
                diff[key] = ex
        return diff

    # Arbitrary type: compare by fields
    attr_names = lhs.__dict__.keys()
    for attr_name in attr_names:
        lhs_val = lhs.__dict__[attr_name]
        rhs_val = rhs.__dict__[attr_name]
        try:
            diff[attr_name] = compare_outputs(lhs_val, rhs_val)
        except Exception as ex:
            diff[attr_name] = ex
    return diff


@dataclass(frozen=True)
class TensorStatistics:
    """Statistical summary of a tensor's elements."""

    mean: float
    """Mean value of all tensor elements."""

    min: float
    """Minimum value among all tensor elements."""

    max: float
    """Maximum value among all tensor elements."""

    stddev: float
    """Standard deviation of all tensor elements."""

    def _asdict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class DifferenceStatistics(TensorStatistics):
    peir: float
    """PEIR (Peak Error To Interval Ratio)."""


def get_tensor_statistics(x: torch.Tensor) -> TensorStatistics:
    """
    Compute statistical summary of a tensor's elements.

    Args:
        x: Input tensor.

    Returns:
        TensorStatistics containing mean, min, max, and stddev.
    """
    x = x.to(torch.float)
    return TensorStatistics(
        mean=torch.mean(x).item(),
        min=torch.min(x).item(),
        max=torch.max(x).item(),
        stddev=torch.std(x).item(),
    )


class ModuleInputOutput(NamedTuple):
    """
    Captured input/output data for a single module during forward pass.

    This named tuple stores all information about a module's execution,
    including its inputs, keyword arguments, and output.
    """

    module: nn.Module
    """The module instance that was executed."""

    module_name: ModuleName
    """Fully qualified name of the module in the model hierarchy."""

    inputs: tuple[torch.Tensor, ...]
    """Positional arguments passed to the module's forward method."""

    kwargs: dict[str, Any]
    """Keyword arguments passed to the module's forward method."""

    output: ModuleOutput
    """Output returned by the module's forward method."""

    def as_serializable(
        self,
        include_tensor_content: bool = False,
        include_type: bool = True,
    ) -> dict[str, Any]:
        """
        Convert the module input/output data to a JSON-serializable dictionary.

        Args:
            include_tensor_content: If True, include actual tensor elements
                (for small tensors or "interesting" modules).
            include_type: If True, include 'type' field in the output dictionary.

        Returns:
            Dictionary containing module_name, module_type, inputs, kwargs, and output,
            all in JSON-serializable format.
        """
        data: dict[str, Any] = {
            "module_name": self.module_name,
            "module_type": type(self.module).__name__,
            "inputs": model_output_to_serializable(
                self.inputs,
                include_tensor_content=include_tensor_content,
                include_type=include_type,
            ),
            "kwargs": {
                k: model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for k, v in self.kwargs.items()
            },
            "output": model_output_to_serializable(
                self.output,
                include_tensor_content=include_tensor_content,
                include_type=include_type,
            ),
        }
        return data


def model_output_to_serializable(
    x: ModuleOutput,
    include_tensor_content: bool = False,
    include_type: bool = True,
) -> str | dict[str, Any]:
    """
    Convert a module output to a JSON-serializable format.

    Recursively converts tensors, ModelOutput objects, named tuples,
    dicts, and iterables to dictionaries with type information.

    Args:
        x: The output value to convert.
        include_tensor_content: If True, include actual tensor elements
            (for small tensors or "interesting" modules).
        include_type: If True, include 'type' field in the output dictionary.

    Returns:
        A JSON-serializable representation of the output.
    """
    data: dict[str, Any] = {}
    if include_type:
        data["type"] = x.__class__.__name__

    if isinstance(x, torch.Tensor):
        data["dtype"] = str(x.dtype)
        data["shape"] = str(x.shape)
        if x.numel() > 1:
            data["statistics"] = get_tensor_statistics(x)._asdict()
        if x.numel() <= 1 or include_tensor_content:
            data["value"] = x.tolist()
        return data

    if isinstance(x, ModelOutput):
        data.update(
            {
                k: model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for k, v in x.__dict__.items()
            }
        )
        return data

    # NamedTuple
    if hasattr(x, "_asdict"):
        data.update(x._asdict())
        return data

    if isinstance(x, dict):
        data.update(
            {
                str(k): model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for k, v in x.items()
            }
        )
        return data

    if isinstance(x, Iterable):
        data.update(
            {
                str(i): model_output_to_serializable(
                    v,
                    include_tensor_content=include_tensor_content,
                    include_type=include_type,
                )
                for i, v in enumerate(x)
            }
        )
        return data

    return str(x)


def trim_prefix_up_to(s: str, char: str) -> str:
    """
    Remove prefix from a string up to and including the first occurrence of a character.

    Args:
        s: Input string.
        char: Character to search for.

    Returns:
        String with prefix removed, or original string if character not found.
    """
    char_index: int = s.find(char)
    if char_index >= 0:
        return s[char_index + 1 :]
    else:
        return s


def detach_tensors(x: ModuleOutput) -> ModuleOutput:
    """
    Recursively detach and clone tensors in a module output.

    Creates detached copies of tensors to prevent gradient tracking and
    preserve tensor values for later comparison.

    Args:
        x: Module output (tensor, ModelOutput, dict, or iterable).

    Returns:
        A copy of the input with all tensors detached and cloned.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().clone()

    data: dict[str, Any] | Iterable

    if isinstance(x, ModelOutput):
        data = {str(k): detach_tensors(v) for k, v in x.__dict__.items()}
        return x.__class__(**data)

    if isinstance(x, dict):
        data = {k: detach_tensors(v) for k, v in x.items()}
        return data

    if isinstance(x, Iterable):
        data = (detach_tensors(i) for i in x)
        # For iterables like lists/tuples, we need to convert the generator to the appropriate type
        if hasattr(x, "__class__"):
            try:
                return x.__class__(data)  # type: ignore[call-arg]
            except TypeError:
                # If the constructor doesn't accept a generator, convert to list first
                return x.__class__(list(data))  # type: ignore[call-arg]
        return list(data)

    return x


def trace_model_input_output(
    model: torch.nn.Module,
    model_inputs: dict[ArgName, torch.Tensor],
    hook: Callable[[ModuleInputOutput], Any],
    skip_ptqwrappers: bool = True,
):
    """
    Run model forward pass and trace all module inputs/outputs.

    Registers a forward hook on all modules, runs the model, and calls
    the provided hook function for each module's execution.

    Args:
        model: The model to trace.
        model_inputs: Input data for the model forward pass.
        hook: Callback function called for each module with ModuleInputOutput data.
        skip_ptqwrappers: If True, skip PTQWrapper modules (default: True).
    """
    module_to_name: dict[nn.Module, ModuleName] | None
    if isinstance(model, QuantModuleBase):
        module_to_name = None
    else:
        module_to_name = build_fqn_map(model)

    def _hook(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, Any],
        output: ModuleOutput,
    ):
        if isinstance(module, PTQWrapper) and skip_ptqwrappers:
            return

        if not module_to_name and not hasattr(module, "fp_name"):
            return

        module_name: ModuleName
        if module_to_name:
            module_name = module_to_name[module]
        else:
            module_name = module.fp_name
            # PTQWrapper adds an fp_name "model" to the top-level model,
            # which is why every submodule obtains an additional "model."
            # prefix in its fp_name. We trim that prefix in order to be
            # consistent with usual (unquantized, unwrapped) models.
            module_name = trim_prefix_up_to(module_name, char=".")

        data = ModuleInputOutput(
            module=module,
            module_name=module_name,
            inputs=inputs,
            kwargs=kwargs,
            output=detach_tensors(output),
        )
        hook(data)

    with module_hook(_hook):
        with torch.no_grad():
            _ = model(**model_inputs)


def print_header(header: str, char: str = "*"):
    """
    Print a formatted header with centered text.

    Args:
        header: Text to display in the header.
        char: Character to use for the border (default: '*').
    """
    print()
    print(char * 80)
    print(f"{char} {header :^76} {char}")
    print(char * 80)
    print()


def compare_side_by_side(
    model_outputs_a: Mapping[str, ModuleOutput],
    model_outputs_b: Mapping[str, ModuleOutput],
    interesting_modules: Iterable[str] = [],
    breakpoint_on_interesting_modules: bool = False,
) -> None:
    """
    Compare outputs from two models side-by-side and print differences.

    For similarly named submodules that are present in both models, computes and prints the difference
    between outputs. Useful for comparing quantized vs unquantized model outputs.

    Args:
        model_outputs_a: First model's outputs (keyed by module name).
        model_outputs_b: Second model's outputs (keyed by module name).
        interesting_modules: Modules to inspect in detail (full tensor diff).
        breakpoint_on_interesting_modules: If True, break into debugger for interesting modules.
    """
    common_module_names: set[ModuleName] = (
        model_outputs_a.keys() & model_outputs_b.keys()
    )
    max_module_name_len = max(len(name) for name in common_module_names)
    format_str = f"{{: <{max_module_name_len}}} {{:}}"

    print("-" * 80)
    print(format_str.format("MODULE NAME", "DIFFERENCE"))
    print("-" * 80)

    for module_name in model_outputs_a.keys():
        if module_name not in model_outputs_b:
            continue
        output_a = model_outputs_a[module_name]
        output_b = model_outputs_b[module_name]
        diff: Number | torch.Tensor | dict[str, Any] | Exception | None
        this_module_is_interesting = module_name in interesting_modules
        try:
            diff = compare_outputs(
                output_a, output_b, full_tensor_diff=this_module_is_interesting
            )
        except Exception as ex:
            diff = ex
        print(
            format_str.format(
                module_name,
                model_output_to_serializable(
                    diff,
                    include_tensor_content=this_module_is_interesting,
                    include_type=False,
                ),
            )
        )

        if this_module_is_interesting and breakpoint_on_interesting_modules:
            breakpoint()


def create_tracing_hook(
    print_input_output: bool,
    module_outputs: MutableMapping[ModuleName, ModuleOutput] | None,
    interesting_modules: Iterable[str] = [],
    breakpoint_on_interesting_modules: bool = False,
):
    """
    Create a hook function for tracing module inputs/outputs.

    Args:
        print_input_output: If True, print module input/output data.
        module_outputs: Dictionary to store module outputs (keyed by module name).
        interesting_modules: List of module names to inspect in detail.
        breakpoint_on_interesting_modules: If True, break into debugger for interesting modules.

    Returns:
        A hook function suitable for use with trace_model_input_output.
    """

    def hook(data: ModuleInputOutput):
        this_module_is_interesting = data.module_name in interesting_modules
        if print_input_output:
            print(f"\n{'='*80}")
            print(
                json.dumps(
                    data.as_serializable(
                        include_tensor_content=this_module_is_interesting
                    ),
                    indent=4,
                )
            )
            print(f"{'='*80}")

        if module_outputs is not None:
            module_outputs[data.module_name] = data.output

        if this_module_is_interesting and breakpoint_on_interesting_modules:
            breakpoint()

    return hook


def parse_arguments():
    """
    Parse and validate command-line arguments.

    Returns:
        Namespace object containing all parsed arguments.

    Raises:
        AssertionError: If --breakpoint-on-interesting-modules is set without
            --interesting-modules, or if model name doesn't contain 'Qwen'.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Trace data flow within model during inference."
    )

    # E.g. "Qwen/Qwen3-VL-2B-Instruct" (for downloading) or ~/models/qwen3-vl-2b (for cached)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF repo name or local path.",
    )

    # E.g. ~/models/qwen3-vl-2b
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for downloaded models.",
    )

    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token for gated/private repos.",
    )

    parser.add_argument(
        "--interesting-modules",
        nargs="+",
        default=[],
        help="Optional list of module names to inspect in detail.",
    )

    parser.add_argument(
        "--breakpoint-on-interesting-modules",
        action="store_true",
        help="Switch to debug mode on interesting modules.",
    )

    parser.add_argument(
        "--no-trace-unquantized",
        action="store_true",
        help="Don't trace unquantized model.",
    )

    parser.add_argument(
        "--no-trace-quantized",
        action="store_true",
        help="Don't trace quantized model.",
    )

    parser.add_argument(
        "--no-side-by-side",
        action="store_true",
        help="Don't do side-by-side validation between quantized and unquantized models.",
    )

    parser.add_argument(
        "--enable-quantization",
        action="store_true",
        help="Enable fake quantization operations to check quantization errors.",
    )

    parser.add_argument(
        "--dtype",
        choices=DTYPE_MAP.keys(),
        type=str.lower,
        help="Quantization data type",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    if os.path.isdir(args.model):
        if args.cache_dir is not None and args.cache_dir != args.model:
            print(
                f"[WARNING] Your cache directory {args.cache_dir} is different from model directory {args.model}."
            )
        if not "qwen" in args.model and not "Qwen" in args.model:
            print(
                f"[WARNING] Your model directory {args.model} doesn't include word 'Qwen'. Note that this script was designed specifically for Qwen3-VL model."
            )
    else:
        print(
            f"[WARNING] Model name {args.model} does not refer to an existing directory. So, we'll try to download the model from huggingface."
        )
        if "Qwen" not in args.model:
            print("[ERROR] This script was designed specifically for Qwen3-VL model.")
            sys.exit(1)

    if args.breakpoint_on_interesting_modules:
        if not args.interesting_modules:
            print(
                "[ERROR] --breakpoint-on-interesting-modules flag requires --interesting-modules to be specified."
            )
            sys.exit(1)

    if args.dtype is not None and not args.enable_quantization:
        print(
            f"[ERROR] --dtype {args.dtype} requires --enable-quantization flag to be specified."
        )
        sys.exit(1)

    if args.dtype is None:
        args.dtype = "uint8"

    return args


def main():
    """
    Main entry point for the trace_qwen script.

    Parses command-line arguments, creates a reduced model configuration,
    generates model inputs, instantiates the model, traces both the original
    and quantized models, and performs side-by-side comparison of outputs.
    """
    args = parse_arguments()
    torch.manual_seed(args.seed)

    cfg: Qwen3VLConfig = prepare_config()

    # Generate model inputs
    model_inputs: dict[str, torch.Tensor] = prepare_inputs(
        model_name=args.model,
        cache_dir=args.cache_dir,
        image_token_id=cfg.image_token_id,
        vocab_size=cfg.text_config.vocab_size,
        image_width=128,
        image_height=96,
        text_prompt="Describe the image.",
        hf_token=args.hf_token,
    )

    print_header("MODEL INPUTS")
    print_model_inputs(model_inputs)

    model = Qwen3VLForConditionalGeneration(cfg).eval()

    # Trace original model's dataflow
    model_outputs: OrderedDict[ModuleName, ModuleOutput] | None
    if not (args.no_trace_unquantized and args.no_side_by_side):
        if not args.no_trace_unquantized:
            print_header("ORIGINAL MODEL")
        model_outputs = None if args.no_side_by_side else OrderedDict()
        trace_model_input_output(
            model=model,
            model_inputs=model_inputs,
            hook=create_tracing_hook(
                print_input_output=not args.no_trace_unquantized,
                module_outputs=model_outputs,
                interesting_modules=args.interesting_modules,
                breakpoint_on_interesting_modules=args.breakpoint_on_interesting_modules,
            ),
        )

    quant_model = prepare_quantized_model(
        model=model,
        model_inputs=model_inputs,
        enable_quantization=args.enable_quantization,
        dtype=DTYPE_MAP[args.dtype],
    )

    # Trace quantized model's dataflow
    quant_model_outputs: OrderedDict[ModuleName, ModuleOutput] | None
    if not (args.no_trace_quantized and args.no_side_by_side):
        if not args.no_trace_quantized:
            print_header("QUANTIZED MODEL")
        quant_model_outputs = None if args.no_side_by_side else OrderedDict()
        trace_model_input_output(
            model=quant_model,
            model_inputs=model_inputs,
            hook=create_tracing_hook(
                print_input_output=not args.no_trace_quantized,
                module_outputs=quant_model_outputs,
                interesting_modules=args.interesting_modules,
                breakpoint_on_interesting_modules=args.breakpoint_on_interesting_modules,
            ),
        )

    if not args.no_side_by_side:
        assert model_outputs is not None and quant_model_outputs is not None
        print_header("SIDE-BY-SIDE COMPARISON")
        compare_side_by_side(
            model_outputs,
            quant_model_outputs,
            interesting_modules=args.interesting_modules,
            breakpoint_on_interesting_modules=args.breakpoint_on_interesting_modules,
        )


if __name__ == "__main__":
    main()
