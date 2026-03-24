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

import argparse
import contextlib
import io
from typing import Any, Optional

import torch
from transformers import AutoProcessor

from tico.quantization import convert, prepare
from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig
from tico.quantization.evaluation.vlm_eval_utils import (
    get_accuracy_on_dataset,
    get_calib_inputs,
    get_dataset,
)

DTYPE_MAP = {
    "float32": torch.float32,
    # TODO: Support more dtypes if needed.
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}


def build_processor_inputs(
    processor: Any,
    image: Any,
    question: str,
    seq_len: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """
    Build one multimodal processor input with optional text truncation.

    Args:
        processor: Hugging Face processor for the target model.
        image: Input image.
        question: User question text.
        seq_len: Optional maximum text sequence length. If None, processor
            default behavior is used.

    Returns:
        Processor output mapping.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{question}\n"
                        "Return ONLY the final answer with no extra words."
                    ),
                },
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    processor_kwargs = {"text": prompt, "images": image, "return_tensors": "pt"}
    if seq_len is not None and seq_len > 0:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = seq_len
    return dict(processor(**processor_kwargs))


def evaluate_model(
    model,
    processor,
    tasks: str,
    device: str,
    nsamples: int = 50,
    max_seq_len: Optional[int] = None,
) -> dict[str, tuple[int, int]]:
    """
    Evaluate a VLM on one or more mini VQA tasks.

    Args:
        model: Model to evaluate.
        processor: Hugging Face processor.
        tasks: Comma-separated task names.
        device: Target device string.
        nsamples: Number of evaluation samples per task. -1 means full dataset.
        max_seq_len: Optional maximum text sequence length for evaluation inputs.

    Returns:
        Mapping from task name to (exact_match_count, total_count).
    """
    tasks_list = tasks.split(",")
    results: dict[str, tuple[int, int]] = {}

    for task in tasks_list:
        with (
            io.StringIO() as buffer,
            contextlib.redirect_stdout(buffer),
            contextlib.redirect_stderr(buffer),
        ):
            ds, adapter = get_dataset(task, n=nsamples)
            em_cnt, total = get_accuracy_on_dataset(
                model,
                processor,
                ds,
                adapter,
                device,
                max_seq_len=max_seq_len,
            )
            results[task] = (em_cnt, total)

    return results


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """
    Move one processor batch to the target device.

    Args:
        batch: Processor output mapping.
        device: Target device.

    Returns:
        Device-moved batch.
    """
    return {k: v.to(device) for k, v in batch.items()}


def print_eval_results(
    title: str,
    results: dict[str, tuple[int, int]],
) -> None:
    """
    Print evaluation results in a simple readable format.

    Args:
        title: Section title.
        results: Task result mapping.
    """
    print(title)
    for key, (correct, total) in results.items():
        print(f"{key}: EM={correct / total:.4f}  (n={total})")


def print_markdown_comparison(
    original_results: dict[str, tuple[int, int]],
    quantized_results: dict[str, tuple[int, int]],
) -> None:
    """
    Print a markdown table comparing original and quantized metrics.

    Args:
        original_results: Baseline results.
        quantized_results: Quantized results.
    """
    tasks = list(quantized_results.keys())

    header = "|model|" + "|".join(tasks) + "|"
    sep = "|--|" + "|".join(["--"] * len(tasks)) + "|"

    original_row = "|original|"
    for task in tasks:
        correct, total = original_results[task]
        original_row += f"{correct / total:.4f}|"

    quantized_row = "|quantized|"
    for task in tasks:
        correct, total = quantized_results[task]
        quantized_row += f"{correct / total:.4f}|"

    print(header)
    print(sep)
    print(original_row)
    print(quantized_row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL GPTQ pipeline (architecture-aware, stagewise)"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF repo name or local path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda|cpu|mps).",
    )
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default="float32",
        help="Model dtype for load.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only if you trust the model repo code.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token for gated/private repos.",
    )
    parser.add_argument(
        "--no_GPTQ",
        action="store_true",
        default=False,
        help="Skip GPTQ and keep the model in floating-point.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache dir for model/dataset loading.",
    )
    parser.add_argument(
        "--nsamples_for_qcalibration",
        type=int,
        default=128,
        help="Number of samples to be used in GPTQ calibration.",
    )
    parser.add_argument(
        "--nsamples_for_evaluation",
        type=int,
        default=50,
        help="Number of samples for evaluation. -1 means full dataset.",
    )
    parser.add_argument(
        "--calib_seq_len",
        type=int,
        default=2048,
        help=(
            "Maximum text sequence length for calibration inputs. "
            "If not set, processor default behavior is used."
        ),
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help=(
            "Maximum text sequence length for evaluation and export. "
            "If not set, processor default behavior is used."
        ),
    )
    parser.add_argument(
        "--linear_weight_bits",
        type=int,
        default=4,
        help="Weight bit-width for GPTQ quantization.",
    )
    parser.add_argument(
        "--gptq_mse",
        type=str,
        default=None,
        choices=["mse", "smse"],
        help="Whether and how to use mse in GPTQ.",
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default=None,
        help="Tasks to evaluate, e.g. `vqav2,textvqa`.",
    )
    parser.add_argument(
        "--sensitivity_path",
        type=str,
        default=None,
        help="Optional path to precomputed sensitivity tensors.",
    )

    # Qwen3-VL GPTQ-specific switches
    parser.add_argument(
        "--quantize_vision",
        action="store_true",
        default=True,
        help="Quantize the vision tower.",
    )
    parser.add_argument(
        "--quantize_text",
        action="store_true",
        default=True,
        help="Quantize the text tower.",
    )
    parser.add_argument(
        "--quantize_lm_head",
        action="store_true",
        default=True,
        help="Quantize lm_head.",
    )
    parser.add_argument(
        "--move_cache_to_cpu",
        action="store_true",
        default=False,
        help="Move cached stage inputs to CPU between stages to save device memory.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="GPTQ group size. -1 disables grouping.",
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="GPTQ percdamp value.",
    )
    parser.add_argument(
        "--actorder",
        action="store_true",
        default=True,
        help="Enable activation-order column permutation.",
    )
    parser.add_argument(
        "--static_groups",
        action="store_true",
        default=False,
        help="Enable static group quantizers.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose GPTQ logging.",
    )
    parser.add_argument(
        "--hide_progress",
        action="store_true",
        default=False,
        help="Disable tqdm progress bars.",
    )

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    quantize_vision = args.quantize_vision
    quantize_text = args.quantize_text
    quantize_lm_head = args.quantize_lm_head

    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print(f"Calib seq len    : {args.calib_seq_len}")
    print(f"Max seq len      : {args.max_seq_len}")
    print(f"Quantize vision  : {quantize_vision}")
    print(f"Quantize text    : {quantize_text}")
    print(f"Quantize lm_head : {quantize_lm_head}")
    print()

    print("Loading FP model …")

    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )

    dev_map = "auto" if args.device != "cpu" else "cpu"

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            device_map=dev_map,
        )
    except Exception:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            token=args.hf_token,
            cache_dir=args.cache_dir,
            device_map=dev_map,
        )

    model.eval()

    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        if hasattr(model.config.text_config, "use_cache"):
            model.config.text_config.use_cache = False

    if args.eval_tasks is not None:
        original_results = evaluate_model(
            model,
            processor,
            args.eval_tasks,
            args.device,
            args.nsamples_for_evaluation,
            max_seq_len=args.max_seq_len,
        )
        print_eval_results("Evaluating original model", original_results)

    calib_inputs = get_calib_inputs(
        "vqav2",
        processor,
        n_samples=args.nsamples_for_qcalibration,
        max_seq_len=args.calib_seq_len,
    )

    if not args.no_GPTQ:
        print("Applying Qwen3-VL GPTQ …")

        sens = None
        if args.gptq_mse == "smse":
            if args.sensitivity_path is not None:
                sens = torch.load(args.sensitivity_path, map_location="cpu")
            else:
                calibrator = SensitivityCalibrator(model, calib_inputs)
                sens = calibrator.compute_sensitivity_info()

        gptq_config = Qwen3VLGPTQConfig(
            weight_bits=args.linear_weight_bits,
            perchannel=True,
            symmetric=False,
            mse=args.gptq_mse,
            sensitivity=sens,
            percdamp=args.percdamp,
            groupsize=args.groupsize,
            actorder=args.actorder,
            static_groups=args.static_groups,
            verbose=args.verbose,
            show_progress=not args.hide_progress,
            quantize_vision=quantize_vision,
            quantize_text=quantize_text,
            quantize_lm_head=quantize_lm_head,
            quantize_vision_patch_embed=quantize_vision,
            quantize_vision_blocks=quantize_vision,
            quantize_vision_merger=quantize_vision,
            quantize_vision_deepstack_mergers=quantize_vision,
            quantize_text_layers=quantize_text,
            move_cache_to_cpu=args.move_cache_to_cpu,
        )

        q_m = prepare(model, gptq_config, inplace=True)

        with torch.no_grad():
            for inp in calib_inputs:
                dev_inp = move_batch_to_device(inp, args.device)
                q_m(**dev_inp)

        q_m = convert(q_m, inplace=True)
    else:
        q_m = model

    if args.eval_tasks is not None:
        quantized_results = evaluate_model(
            q_m,
            processor,
            args.eval_tasks,
            args.device,
            args.nsamples_for_evaluation,
            max_seq_len=args.max_seq_len,
        )
        print_eval_results("Evaluating quantized model", quantized_results)
        print_markdown_comparison(original_results, quantized_results)


if __name__ == "__main__":
    main()
