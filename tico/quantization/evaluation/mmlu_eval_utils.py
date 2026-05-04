from typing import Any

import torch

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table


def _normalize_subject(subject: str) -> str:
    if subject is None:
        return None

    if subject.startswith("mmlu"):
        return subject

    return f"mmlu_{subject}"


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
        [_normalize_subject(subject) for subject in subjects] if subjects else ["mmlu"]
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
    print(make_table(results))
