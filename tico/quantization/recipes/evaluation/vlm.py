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

import contextlib
import io
from typing import Any, Mapping, Sequence

import tqdm

from tico.quantization.evaluation.vlm_eval_utils import (
    evaluate_ppl,
    evaluate_ppl_chat_prefix,
    get_accuracy_on_dataset,
    get_coco_scores_on_dataset,
    get_dataset,
)
from tico.quantization.recipes.data.dataset_usage import EVALUATION_ROLE
from tico.quantization.recipes.evaluation.selection import should_run_evaluation

VQA_TARGET = "vqa"


def run_vqa_evaluation(
    *,
    eval_cfg: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: str,
    n_samples: int,
    max_seq_len: int | None,
    verbose: bool,
    show_progress: bool,
) -> dict[str, tuple[int, int]] | None:
    """Select, normalize, run, and print the ``vqa`` evaluation target.

    VLM adapters call this at their VQA position so that the selection and
    ``evaluation.vlm_tasks`` interpretation policy is edited once instead of
    once per model family. The evaluator itself stays in ``evaluate_vqa_tasks``.

    Selection follows ``should_run_evaluation``: ``evaluation.selected_tasks``
    is an exclusive allow-list when present; otherwise the truthiness of the
    raw ``evaluation.vlm_tasks`` value decides. Normalization and validation
    only happen when the target is selected:

    - a string is split on commas and each entry is stripped;
    - a sequence must contain only strings, each of which is stripped but
      never re-split;
    - empty entries are dropped while order and duplicates are preserved;
    - a selected target whose normalized list is empty is an error.

    The caller supplies the already-resolved evaluation arguments; this helper
    does not read other config sections or substitute defaults for them.

    Returns the evaluator result, or ``None`` when the target did not run.
    """
    raw_vqa_tasks = eval_cfg.get("vlm_tasks") or []
    if not should_run_evaluation(
        eval_cfg,
        VQA_TARGET,
        default_enabled=bool(raw_vqa_tasks),
    ):
        return None

    if isinstance(raw_vqa_tasks, str):
        vqa_tasks = [task.strip() for task in raw_vqa_tasks.split(",") if task.strip()]
    elif isinstance(raw_vqa_tasks, Sequence):
        vqa_tasks = []
        for task in raw_vqa_tasks:
            if not isinstance(task, str):
                raise TypeError("evaluation.vlm_tasks must contain only strings.")
            normalized_task = task.strip()
            if normalized_task:
                vqa_tasks.append(normalized_task)
    else:
        raise TypeError(
            "evaluation.vlm_tasks must be a sequence or comma-separated string."
        )

    if not vqa_tasks:
        raise ValueError("evaluation.vlm_tasks must be non-empty when vqa runs.")

    vqa_results = evaluate_vqa_tasks(
        model=model,
        processor=processor,
        tasks=vqa_tasks,
        device=device,
        n_samples=n_samples,
        max_seq_len=max_seq_len,
        verbose=verbose,
        show_progress=show_progress,
    )
    print_vqa_results("VQA evaluation", vqa_results)
    return vqa_results


def evaluate_vqa_tasks(
    *,
    model: Any,
    processor: Any,
    tasks: list[str],
    device: str,
    n_samples: int,
    max_seq_len: int | None,
    verbose: bool = False,
    show_progress: bool = True,
) -> dict[str, tuple[int, int]]:
    results: dict[str, tuple[int, int]] = {}
    for task in tasks:
        if "vqa" not in task:
            continue

        with contextlib.ExitStack() as stack:
            # Keep the default VQA path quiet, but leave stderr visible when
            # progress bars are enabled because tqdm writes there by default.
            if not verbose:
                stdout_buffer = stack.enter_context(io.StringIO())
                stack.enter_context(contextlib.redirect_stdout(stdout_buffer))
                if not show_progress:
                    stderr_buffer = stack.enter_context(io.StringIO())
                    stack.enter_context(contextlib.redirect_stderr(stderr_buffer))

            dataset, adapter = get_dataset(task, role=EVALUATION_ROLE, n=n_samples)
            total_for_progress = n_samples if n_samples >= 0 else None
            dataset_iter = tqdm.tqdm(
                dataset,
                desc=f"{task} eval",
                total=total_for_progress,
                disable=not show_progress,
            )
            correct, total = get_accuracy_on_dataset(
                model,
                processor,
                dataset_iter,
                adapter,
                device,
                max_seq_len=max_seq_len,
                verbose=verbose,
            )
        results[task] = (correct, total)
    return results


def print_vqa_results(title: str, results: dict[str, tuple[int, int]]) -> None:
    print(title)
    for task, (correct, total) in results.items():
        print(f"{task}: EM={correct / total:.4f}  (n={total})")


def evaluate_coco_score_dataset(
    *,
    model: Any,
    processor: Any,
    dataset_name: str,
    device: str,
    n_samples: int,
    max_seq_len: int | None,
) -> dict[str, float]:
    """
    Evaluate a COCO-score-compatible VLM dataset.

    Args:
        model: Model to evaluate.
        processor: Hugging Face processor paired with the model.
        dataset_name: Dataset key accepted by the shared VLM evaluation helper.
        device: Device string used for inference.
        n_samples: Number of samples to evaluate. -1 means the full dataset.
        max_seq_len: Optional maximum text sequence length.

    Returns:
        COCO-style metric values returned by the shared evaluation helper.
    """
    dataset, _ = get_dataset(dataset_name, role=EVALUATION_ROLE, n=n_samples)
    return get_coco_scores_on_dataset(
        model=model,
        processor=processor,
        dataset_name=dataset_name,
        ds=dataset,
        device=device,
        max_seq_len=max_seq_len,
    )


def evaluate_coco(
    *,
    model: Any,
    processor: Any,
    device: str,
    n_samples: int,
    max_seq_len: int | None,
    dataset_name: str = "coco",
) -> dict[str, float]:
    """
    Evaluate a COCO-score-compatible captioning dataset.

    Args:
        model: Model to evaluate.
        processor: Hugging Face processor paired with the model.
        device: Device string used for inference.
        n_samples: Number of samples to evaluate. -1 means the full dataset.
        max_seq_len: Optional maximum text sequence length.
        dataset_name: Dataset key accepted by the shared COCO-score helper.

    Returns:
        COCO-style metric values.
    """
    return evaluate_coco_score_dataset(
        model=model,
        processor=processor,
        dataset_name=dataset_name,
        device=device,
        n_samples=n_samples,
        max_seq_len=max_seq_len,
    )


def evaluate_llava_bench(
    *,
    model: Any,
    processor: Any,
    device: str,
    n_samples: int,
    max_seq_len: int | None,
) -> dict[str, float]:
    """
    Evaluate LLaVA-Bench-in-the-Wild with legacy COCO-style metrics.

    LLaVA-Bench is open-ended natural QA, so CIDEr/BLEU should only be used as
    a legacy diagnostic signal. Use the judge-based LLaVA-Bench path for
    benchmark-style scoring.

    Args:
        model: Model to evaluate.
        processor: Hugging Face processor paired with the model.
        device: Device used for inference.
        n_samples: Number of samples to evaluate. -1 means the full dataset.
        max_seq_len: Optional maximum text sequence length.

    Returns:
        COCO-style metric values.
    """
    return evaluate_coco_score_dataset(
        model=model,
        processor=processor,
        dataset_name="llava_bench",
        device=device,
        n_samples=n_samples,
        max_seq_len=max_seq_len,
    )


def print_coco_score_results(title: str, results: dict[str, float]) -> None:
    """
    Print COCO-style score results while keeping count fields readable.

    Args:
        title: Section title to print before metrics.
        results: Metric mapping returned by a COCO-style evaluation helper.
    """
    print(title)
    for metric, value in results.items():
        if isinstance(value, int):
            print(f"{metric:<14} {value:d}")
        else:
            print(f"{metric:<14} {float(value):.3f}")


def evaluate_vlm_text_ppl(
    *,
    model: Any,
    processor: Any,
    dataset_name: str,
    split: str,
    device: str,
    stride: int,
    max_seq_len: int,
    show_progress: bool,
) -> float:
    dataset, _ = get_dataset(dataset_name, role=EVALUATION_ROLE, split=split, n=-1)
    return float(
        evaluate_ppl(
            model=model,
            tokenizer=processor.tokenizer,
            ds=dataset,
            device=device,
            stride=stride,
            max_seq_len=max_seq_len,
            show_progress=show_progress,
        )
    )


def evaluate_vlm_text_ppl_chat_prefix(
    *,
    model: Any,
    processor: Any,
    dataset_name: str,
    split: str,
    device: str,
    stride: int,
    max_seq_len: int,
    show_progress: bool = True,
) -> float:
    """Evaluate conditional perplexity using chat-prefix protocol.

    A preceding text span is placed in a user prompt via
    ``apply_chat_template``.  Only the reference assistant continuation is
    scored; chat/control/prompt tokens are context and are not scored.

    This is the natural evaluation for instruction-tuned models (e.g. Gemma 4
    IT) where the raw token-stream PPL is not directly comparable because the
    model expects chat-formatted input.

    Args:
        model: Model to evaluate.
        processor: Hugging Face processor paired with the model.
        dataset_name: Dataset key accepted by the shared VLM evaluation helper.
        split: Dataset split to load.
        device: Device string used for inference.
        stride: stride for evaluation.
        max_seq_len: max seq_len for evaluation.
        show_progress: Whether to show a progress bar.

    Returns:
        Conditional perplexity score.
    """
    dataset, _ = get_dataset(dataset_name, role=EVALUATION_ROLE, split=split, n=-1)
    return float(
        evaluate_ppl_chat_prefix(
            model=model,
            processor=processor,
            ds=dataset,
            device=device,
            stride=stride,
            max_seq_len=max_seq_len,
            show_progress=show_progress,
        )
    )
