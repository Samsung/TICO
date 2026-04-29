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

import os
import sys
from collections import OrderedDict
from typing import Any

import torch
import torch.nn as nn

from transformers import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration,
)

import tico
import tico.quantization
import tico.quantization.config.ptq
from tico.quantization.wrapq.dtypes import DType, INT16, INT4, INT8, UINT4, UINT8
from tico.quantization.wrapq.utils.introspection import (
    ArgName,
    compare_side_by_side,
    create_tracing_hook,
    ModuleName,
    ModuleOutput,
    trace_model_input_output,
)
from tico.utils.version import package_version_is_at_least

# Names exposed to wildcard imports from this module
__all__ = [
    # Model preparation
    "prepare_inputs",
    "prepare_config",
    "prepare_quantized_model",
]

# Type aliases (for more descriptive type hints)
ModelNameOrPath = str
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
    vision_start_token_id: int,
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
        local_files_only=os.path.isdir(model_name),
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
    old_image_pad_token_id = 151655
    old_vision_start_token_id = 151652
    input_ids: torch.Tensor = model_inputs["input_ids"]
    input_ids[input_ids == old_image_pad_token_id] = image_token_id
    input_ids[input_ids == old_vision_start_token_id] = vision_start_token_id

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
    vocab_size = 1000

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
            "vocab_size": vocab_size,
            "use_cache": False,
            "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        image_token_id=vocab_size - 2,
        video_token_id=vocab_size - 1,
        vision_start_token_id=vocab_size - 3,
    )
    assert cfg.image_token_id < cfg.text_config.vocab_size
    assert cfg.video_token_id < cfg.text_config.vocab_size
    assert cfg.vision_start_token_id < cfg.text_config.vocab_size

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
                "visual_start_idx": 4,
                "spatial_merge_size": 2,
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
        "--no-skip-ptqwrappers",
        action="store_true",
        help="Don't skip PTQ wrappers when tracing.",
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

    if args.no_trace_quantized and args.no_skip_ptqwrappers:
        print(
            f"[WARNING] --no-skip-ptqwrappers has no effect when --no-trace-quantized flag is specified."
        )

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
        vision_start_token_id=cfg.vision_start_token_id,
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
            skip_ptqwrappers=not args.no_skip_ptqwrappers,
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
