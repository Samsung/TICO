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

import torch
from transformers import AutoProcessor

from tico.quantization import convert, prepare

from tico.quantization.algorithm.gptq.utils import SensitivityCalibrator
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.evaluation.vlm_eval_utils import get_calib_inputs
from tico.quantization.wrapq.examples.quantize_qwen3_vl_with_gptq import (
    evaluate_model,
    print_eval_results,
    print_markdown_comparison,
)

DTYPE_MAP = {
    "float32": torch.float32,
    # TODO Support more dtypes
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}


def main():
    parser = argparse.ArgumentParser(
        description="GPTQ+PTQ pipeline (weight-only + activation)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HF repo name or local path."
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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
        "--cache_dir",
        type=str,
        default=None,
        help="cache_dir for using model/datasets loading",
    )
    parser.add_argument(
        "--nsamples_for_qcalibration",
        type=int,
        default="128",  # almost standard
        help="number of samples to be used in GPTQ/PTQ calibration",
    )
    parser.add_argument(
        "--nsamples_for_evaluation",
        type=int,
        default="50",
        help="number of samples to be used in equantized model valuation. -1 stands for the whole dataset",
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
    )

    args = parser.parse_args()
    print(args)

    # Basic setup
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print(f"Calib seq len    : {args.calib_seq_len}")
    print(f"Max seq len      : {args.max_seq_len}")
    print()

    # -------------------------------------------------------------------------
    # Load model and processor
    # -------------------------------------------------------------------------
    print("Loading FP model …")

    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, cache_dir=args.cache_dir
    )
    dev_map = "balanced" if args.device != "cpu" else "cpu"
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=dtype,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            device_map=dev_map,
        )
    except:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            args.model,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
            device_map=dev_map,
        )

    model.eval()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        if hasattr(model.config.text_config, "use_cache"):
            model.config.text_config.use_cache = False

    if args.calib_seq_len is not None:
        model.config.text_config.max_position_embeddings = min(
            model.config.text_config.max_position_embeddings, args.calib_seq_len
        )

    if args.eval_tasks is not None:
        print("Evaluating original model")
        original_results = evaluate_model(
            model,
            processor,
            args.eval_tasks,
            args.device,
            args.nsamples_for_evaluation,
            max_seq_len=args.max_seq_len,
        )
        print_eval_results("Evaluating original model", original_results)
        for key in original_results:
            result = original_results[key]
            print(
                f"Original EM: {result[0]/result[1]:.4f}  (dataset={key}, n={result[1]})"
            )

    calib_inputs = get_calib_inputs(
        "vqav2", processor, n_samples=args.nsamples_for_qcalibration
    )

    # -------------------------------------------------------------------------
    # Run GPTQ (weight-only) pass
    # -------------------------------------------------------------------------
    print("Applying GPTQ …")

    sens = None
    if args.gptq_mse is not None and args.gptq_mse == "smse":
        if args.sensitivity_path is not None:
            sens = torch.load(args.sensitivity_path)
        else:
            calibrator = SensitivityCalibrator(model, calib_inputs)
            sens = calibrator.compute_sensitivity_info()

    gptq_config = GPTQConfig(
        weight_bits=args.linear_weight_bits,
        perchannel=True,
        mse=args.gptq_mse,
        sensitivity=sens,
    )
    q_m = prepare(model, gptq_config, inplace=True)

    with torch.no_grad():
        for inp in calib_inputs:
            for item in inp:
                inp[item] = inp[item].to(args.device)
            q_m(**inp)

    q_m = convert(q_m, inplace=True)

    # -------------------------------------------------------------------------
    # evaluate quantized model
    # -------------------------------------------------------------------------
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
