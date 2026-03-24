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


import argparse
import random

import torch

from tico.quantization.evaluation.vlm_eval_utils import (
    DATASETS,
    get_accuracy_on_dataset,
    get_dataset,
)


def load_model_and_processor(model_id: str, torch_dtype: torch.dtype):
    """
    Load a vision-language model and its processor.

    The loader first tries the newer Hugging Face auto class for image-text
    generation models and falls back to the older vision-to-sequence class for
    compatibility with older `transformers` versions.

    Args:
        model_id: Hugging Face model ID or local model path.
        torch_dtype: Torch dtype to use when loading the model.

    Returns:
        A tuple of:
        - loaded processor
        - loaded model
    """
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=torch_dtype,
            trust_remote_code=True,
        )
    except ImportError:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            dtype=torch_dtype,
            trust_remote_code=True,
        )

    return processor, model


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for mini VQA evaluation.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model name or local path (e.g. Qwen/Qwen2.5-VL-3B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASETS.keys()),
        default="vqav2",
        help="Evaluation dataset.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of samples for mini evaluation. Use -1 for all samples.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to evaluate. If omitted, the dataset default split is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate per example.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Zero means greedy decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device such as cpu or cuda.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading dtype.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-sample logging during evaluation.",
    )
    return parser.parse_args()


def main():
    """Run lightweight VLM evaluation from the command line."""
    args = parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    ds, adapter = get_dataset(
        dataset=args.dataset,
        n=args.n,
        split=args.split,
    )

    processor, model = load_model_and_processor(args.model_id, torch_dtype)
    model = model.to(args.device)
    model.eval()

    em_cnt, total = get_accuracy_on_dataset(
        model=model,
        processor=processor,
        ds=ds,
        adapter=adapter,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verbose=not args.quiet,
    )

    if total == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    print(f"\nFinal EM: {em_cnt / total:.4f}  (dataset={args.dataset}, n={total})")


if __name__ == "__main__":
    main()
