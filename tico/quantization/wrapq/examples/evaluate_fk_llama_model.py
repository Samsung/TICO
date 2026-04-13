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

from lm_eval.utils import make_table

from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks

DTYPE_MAP = {
    "float32": torch.float32,
    # TODO Support more dtypes
    # "bfloat16": torch.bfloat16,
    # "float16": torch.float16,
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fake-quantized Llama model"
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
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token for gated/private repos.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only if you trust the model repo code.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="cache_dir for using model/datasets loading",
    )
    parser.add_argument(
        "--fk_model_path", type=str, required=True, help="Path to fake_quantized model"
    )
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default=None,
        required=True,  # TODO revisit this option as this script can also be used for sample generation
        help="tasks to be evaluated using lm_eval, e.g. `winogrande,arc_easy,arc_challenge,openbookqa,mmlu_pro,ifeval,bbh`",
    )
    parser.add_argument(
        "--skip_fp_eval",
        action="store_true",
        help="Enable only if you trust the model repo code.",
    )

    args = parser.parse_args()
    print(args)

    # -------------------------------------------------------------------------
    # Basic setup
    # -------------------------------------------------------------------------
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    print("=== Config ===")
    print(f"Model            : {args.model}")
    print(f"Device           : {device.type}")
    print(f"DType            : {args.dtype}")
    print(f"fk_model_path    : {args.fk_model_path}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )

    if not args.skip_fp_eval:

        # -------------------------------------------------------------------------
        # FP model evaluation
        # -------------------------------------------------------------------------
        print("Loading FP model …")
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                dtype=dtype,
                trust_remote_code=args.trust_remote_code,
                token=args.hf_token,
                cache_dir=args.cache_dir,
            )
            .cpu()
            .eval()
        )

        if args.eval_tasks is not None:
            config = model.config
            max_seq_len = config.max_position_embeddings
            results = evaluate_llm_on_tasks(
                model, tokenizer, args.eval_tasks, max_length=max_seq_len
            )
            print("Original RESULTS ARE:")
            print(make_table(results))

        model = model.cpu()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # FK model evaluation
    # -------------------------------------------------------------------------
    print("Loading fake quantized model …")
    fk_model = torch.load(args.fk_model_path, weights_only=False).eval().to(args.device)

    if args.eval_tasks is not None:
        config = fk_model.wrapped.config
        max_seq_len = config.max_position_embeddings

        results = evaluate_llm_on_tasks(
            fk_model, tokenizer, args.eval_tasks, max_length=max_seq_len
        )
        print("Quantized RESULTS ARE:")
        print(make_table(results))


if __name__ == "__main__":
    main()
