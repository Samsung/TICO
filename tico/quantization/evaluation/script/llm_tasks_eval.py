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
from typing import Any

from tico.quantization.evaluation.optional_deps import require_attr, require_module

_LM_EVAL_INSTALL_HINT = "pip install lm-eval"
_TRANSFORMERS_INSTALL_HINT = "pip install transformers"


def evaluate_llm_on_tasks(
    model: Any,
    tokenizer: Any,
    tasks: str,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Evaluate a Hugging Face causal LM on one or more ``lm_eval`` tasks.

    Optional third-party dependencies are loaded only when this function runs,
    keeping recipe imports lightweight for users who only run quantization.
    """
    evaluator = require_module(
        "lm_eval.evaluator",
        feature="lm_eval task evaluation",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )
    HFLM = require_attr(
        "lm_eval.models.huggingface",
        "HFLM",
        feature="lm_eval task evaluation",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )

    if hasattr(model, "wrapped"):
        model = model.wrapped

    model_to_evaluate = HFLM(
        pretrained=model,
        backend="causal",
        tokenizer=tokenizer,
        max_length=max_length,
        truncation=True,
    )
    tasks_list: list[str] = [task.strip() for task in tasks.split(",") if task.strip()]
    return evaluator.simple_evaluate(model_to_evaluate, tasks=tasks_list)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path (e.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')",
    )
    ap.add_argument(
        "--eval_tasks",
        type=str,
        default="arc_easy",
        help=(
            "Tasks to run evaluation, e.g. "
            "`winogrande,arc_easy,arc_challenge,openbookqa,mmlu_pro,ifeval,bbh`"
        ),
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    ap.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable only if you trust the model repo code.",
    )
    ap.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token for gated/private repos.",
    )
    ap.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="cache_dir for to use for model loading",
    )

    args = ap.parse_args()

    AutoTokenizer = require_attr(
        "transformers",
        "AutoTokenizer",
        feature="standalone LLM evaluation script",
        install_hint=_TRANSFORMERS_INSTALL_HINT,
    )
    AutoModelForCausalLM = require_attr(
        "transformers",
        "AutoModelForCausalLM",
        feature="standalone LLM evaluation script",
        install_hint=_TRANSFORMERS_INSTALL_HINT,
    )
    make_table = require_attr(
        "lm_eval.utils",
        "make_table",
        feature="standalone LLM evaluation script",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )

    print("Loading FP model …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            token=args.hf_token,
            cache_dir=args.cache_dir,
        )
        .to(args.device)
        .eval()
    )

    results = evaluate_llm_on_tasks(model, tokenizer, args.eval_tasks)

    print(f"results of {args.model} evaluation:")
    print(make_table(results))


if __name__ == "__main__":
    main()
