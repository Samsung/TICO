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

"""HellaSwag benchmark evaluation utilities.

HellaSwag is a dataset for commonsense natural language inference (NLI).
It evaluates an LLM's commonsense reasoning by requiring it to choose the
most logical ending to a sentence describing an everyday situation.

Dataset structure:
- ctx_a, ctx_b: Context (short story or event description)
- ending_a, ending_b, ending_c, ending_d: Four possible continuations
- label: Index (0-3) of the correct ending

Evaluation metric: Multiple-choice accuracy
"""

from typing import Any

import torch

from tico.quantization.evaluation.optional_deps import require_attr, require_module

_LM_EVAL_INSTALL_HINT = "pip install lm-eval"


def _load_lm_eval_runtime() -> tuple[Any, Any]:
    """Load ``lm_eval`` runtime pieces lazily."""
    evaluator = require_module(
        "lm_eval.evaluator",
        feature="HellaSwag evaluation",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )
    HFLM = require_attr(
        "lm_eval.models.huggingface",
        "HFLM",
        feature="HellaSwag evaluation",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )
    return evaluator, HFLM


def evaluate_hellaswag(
    model,
    tokenizer,
    device: str | torch.device = "cuda",
    n_shots: int = 10,
    n_samples: int = -1,
    batch_size: int = 1,
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a model on the HellaSwag benchmark using lm_eval.

    This function uses the lm_eval framework for standardized HellaSwag evaluation.
    The model can be a standard HuggingFace model or a wrapped quantized model
    (e.g., QuantQwen3VLForConditionalGeneration).

    Args:
        model: Language model with generation capability. Can be a wrapped model.
        tokenizer: Matching tokenizer for the model.
        device: Device for inference.
        n_shots: Number of few-shot examples (default: 10 for HellaSwag).
        n_samples: Number of test samples. Use -1 for full test set.
        batch_size: Batch size for evaluation.
        max_seq_len: Maximal sequence length to be generated.

    Returns:
        Results dictionary with accuracy metrics including:
        - acc: Accuracy
        - acc_norm: Normalized accuracy (length-normalized)
    """
    evaluator, HFLM = _load_lm_eval_runtime()

    # Unwrap if needed (handles PTQWrapper)
    if hasattr(model, "wrapped"):
        model = model.wrapped

    lm = HFLM(
        pretrained=model,
        backend="causal",
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        max_length=max_seq_len,
        truncation=True,
    )

    # Run lm_eval evaluation on hellaswag task
    results: dict[str, Any] = evaluator.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],
        num_fewshot=n_shots,
        batch_size=batch_size,
        limit=n_samples if n_samples > 0 else None,
        device=str(device) if device else None,
    )
    return results


def print_hellaswag_results(results: dict[str, Any]) -> None:
    """
    Print HellaSwag evaluation results in a formatted table.

    Args:
        results: Results dictionary from evaluate_hellaswag().
    """
    make_table = require_attr(
        "lm_eval.utils",
        "make_table",
        feature="HellaSwag result printing",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )
    print(make_table(results))


def get_hellaswag_accuracy(results: dict[str, Any]) -> dict[str, float]:
    """
    Extract accuracy metrics from HellaSwag evaluation results.

    Args:
        results: Results dictionary from evaluate_hellaswag().

    Returns:
        Dictionary with accuracy metrics:
        - acc: Accuracy
        - acc_norm: Normalized accuracy (length-normalized)
    """
    hellaswag_results = results.get("results", {}).get("hellaswag", {})
    return {
        "acc": hellaswag_results.get("acc", 0.0),
        "acc_norm": hellaswag_results.get("acc_norm", 0.0),
    }
