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

from typing import Any

import torch

from tico.quantization.evaluation.optional_deps import require_attr, require_module

_LM_EVAL_INSTALL_HINT = "pip install lm-eval"


def _normalize_subject(subject: str | None) -> str | None:
    if subject is None:
        return None

    if subject.startswith("mmlu"):
        return subject

    return f"mmlu_{subject}"


def _load_lm_eval_runtime() -> tuple[Any, Any]:
    """Load ``lm_eval`` runtime pieces lazily."""
    evaluator = require_module(
        "lm_eval.evaluator",
        feature="MMLU evaluation",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )
    HFLM = require_attr(
        "lm_eval.models.huggingface",
        "HFLM",
        feature="MMLU evaluation",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )
    return evaluator, HFLM


def evaluate_mmlu(
    model,
    tokenizer,
    subjects: list[str] | None = None,
    device: str | torch.device = "cuda",
    n_shots: int = 5,
    n_samples: int = -1,
    batch_size: int = 1,
    max_seq_len: int | None = None,
) -> dict[str, Any]:
    """
    Evaluate a model on the MMLU benchmark using lm_eval.

    This function uses the lm_eval framework for standardized MMLU evaluation.
    The model can be a standard HuggingFace model or a wrapped quantized model
    (e.g., QuantQwen3VLForConditionalGeneration).

    Args:
        model: Language model with generation capability. Can be a wrapped model.
        tokenizer: Matching tokenizer for the model.
        subjects: list of subjects to evaluate (e.g. 'stem', 'humanities', 'social_sciences', 'astronomy', etc.). Use None for all subjects.
        device: Device for inference.
        n_shots: Number of few-shot examples per subject.
        n_samples: Number of test samples per subject. Use -1 for full test sets.
        batch_size: Batch size for evaluation.
        max_seq_len: Maximal sequence length to be generated.

    Returns:
        Aggregated results dictionary with per-subject, per-domain, and overall accuracy.
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

    # Convert subjects to lm_eval task names
    tasks: list[str] = (
        [task for task in (_normalize_subject(subject) for subject in subjects) if task]
        if subjects
        else ["mmlu"]
    )

    # Run lm_eval evaluation
    results: dict[str, Any] = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=n_shots,
        batch_size=batch_size,
        limit=n_samples if n_samples > 0 else None,
        device=str(device) if device else None,
    )
    return results


def print_mmlu_results(results: dict[str, Any]) -> None:
    make_table = require_attr(
        "lm_eval.utils",
        "make_table",
        feature="MMLU result printing",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )
    print(make_table(results))
