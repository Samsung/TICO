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

import ast
import re
from typing import Any, Iterable

import torch
from datasets import load_dataset

from tico.quantization.evaluation.vlm_eval_utils import (
    generate_answer,
    generate_image_only_answer,
)


MMMU_DATASETS = ["MMMU/MMMU", "MMMU/MMMU_Pro"]

MMMU_SUBJECTS: dict[str, list[str]] = {
    "MMMU/MMMU": [
        "Accounting",
        "Agriculture",
        "Architecture_and_Engineering",
        "Art",
        "Art_Theory",
        "Basic_Medical_Science",
        "Biology",
        "Chemistry",
        "Clinical_Medicine",
        "Computer_Science",
        "Design",
        "Diagnostics_and_Laboratory_Medicine",
        "Economics",
        "Electronics",
        "Energy_and_Power",
        "Finance",
        "Geography",
        "History",
        "Literature",
        "Manage",
        "Marketing",
        "Materials",
        "Math",
        "Mechanical_Engineering",
        "Music",
        "Pharmacy",
        "Physics",
        "Psychology",
        "Public_Health",
        "Sociology",
    ],
    "MMMU/MMMU_Pro": [
        "standard (10 options)",
        "standard (4 options)",
        "vision",
    ],
}

MMMU_SPLITS: dict[str, list[str]] = {
    "MMMU/MMMU": [
        "dev",
        "validation",
        "test",
    ],
    "MMMU/MMMU_Pro": [
        "test",
    ],
}

# Keep the official prompt strings aligned with
# MMMU-Benchmark/MMMU: mmmu-pro/prompts.yaml.
MMMU_PRO_VISION_PROMPTS: dict[str, str] = {
    "official_direct": (
        "Answer with the option letter from the given choices directly. "
        "The last line of your response should be of the following format: "
        "'Answer: $LETTER' (without quotes) where LETTER is one of options."
    ),
    "official_cot": (
        "Write out the multiple-choice question in the image and then solve it. "
        "The last line of your response should be of the following format: "
        "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
        "Think step by step before answering."
    ),
}
DEFAULT_MMMU_PRO_VISION_PROMPT_MODE = "official_direct"
DEFAULT_MMMU_MAX_NEW_TOKENS = 16
MMMU_PRO_VISION_DEFAULT_MAX_NEW_TOKENS: dict[str, int] = {
    "official_direct": 50,
    "official_cot": 50,
}

_ANSWER_DECLARATION_PATTERNS = (
    re.compile(
        r"\b(?:answer|option|choice)\s*(?:(?:is|:|=)\s*)?"
        r"(?:\*\*)?\(?([A-J])\)?(?:\*\*)?"
        r"(?=$|[\s.,;:!?])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:choose|select)\s+(?:option\s+)?"
        r"(?:\*\*)?\(?([A-J])\)?(?:\*\*)?"
        r"(?=$|[\s.,;:!?])",
        re.IGNORECASE,
    ),
)
_LEADING_ANSWER_PATTERN = re.compile(
    r"^\s*(?:\*\*)?(?:\(([A-J])\)|([A-J])[.):])(?:\*\*)?(?=\s|$)",
    re.IGNORECASE,
)
_BARE_ANSWER_PATTERN = re.compile(
    r"\s*(?:\*\*)?\(?([A-J])\)?(?:\*\*)?[.):]?\s*",
    re.IGNORECASE,
)


def take_from_dataset(ds, start: int, n: int) -> Iterable[dict[str, Any]]:
    assert start >= 0
    i = 0
    for ex in ds:
        if n >= 0 and i >= start + n:
            break
        if i >= start:
            yield ex
        i += 1


def load_data(
    dataset: str,
    subject: str,
    split: str,
    start: int = 0,
    n_samples: int = -1,
    streaming: bool = True,
) -> Iterable[dict[str, Any]]:

    if dataset not in MMMU_DATASETS:
        raise ValueError(f"Invalid dataset '{dataset}'")

    if subject not in MMMU_SUBJECTS[dataset]:
        raise ValueError(f"Invalid subject '{subject}'")

    if split not in MMMU_SPLITS[dataset]:
        raise ValueError(f"Invalid split '{split}'")

    ds: Iterable[dict[str, Any]] = load_dataset(
        path=dataset,
        name=subject,
        split=split,
        streaming=streaming,
    )

    if n_samples > 0:
        ds = take_from_dataset(ds, start=start, n=n_samples)

    return ds


def get_item_mmmu(ex: dict[str, Any]) -> dict[str, Any]:
    choices = ex["options"]
    if isinstance(choices, str):
        # Convert string "['choice1', 'choice2']" to a list ['choice1', 'choice2']
        choices = ast.literal_eval(choices)

    return {
        "id": ex["id"],
        "image": ex["image_1"] if "image_1" in ex else ex["image"],
        "question": ex["question"] if "question" in ex else "",
        "choices": choices,
        "answer": ex["answer"],
    }


def format_multichoice_question(
    question: str,
    choices: list[str],
    answer: str | None = None,
) -> str:
    """
    Format a single multichoice question.

    Args:
        question: The question text.
        choices: List of 4 answer choices.
        answer: The correct answer letter (A/B/C/D), or None for target questions.

    Returns:
        Formatted question string.
    """
    lines = [question]
    for i, choice in enumerate(choices):
        letter = chr(ord("A") + i)
        lines.append(f"{letter}. {choice}")
    if answer is not None:
        lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def build_few_shot_prompt(
    question: str,
    choices: list[str],
    subject: str,
    few_shot_examples: list[dict[str, Any]],
) -> str:
    """
    Build a few-shot prompt.

    The prompt includes:
    - A header indicating the subject
    - Few-shot examples with answers
    - The target question without an answer

    Args:
        question: The target question text.
        choices: List of 4 answer choices for the target question.
        subject: The subject name for context.
        few_shot_examples: List of few-shot example dictionaries with
            'question', 'choices', and 'answer' keys.

    Returns:
        A formatted prompt string ready for model input.
    """
    # Format subject name for display
    subject_display = subject.replace("_", " ").title()

    prompt_parts = [
        f"The following are multiple choice questions about {subject_display}."
    ]

    # Add few-shot examples
    for ex in few_shot_examples:
        prompt_parts.append("")
        prompt_parts.append(
            format_multichoice_question(
                ex["question"],
                ex["choices"],
                ex["answer"],
            )
        )

    # Add target question
    prompt_parts.append("")
    prompt_parts.append(format_multichoice_question(question, choices, answer=None))
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def get_mmmu_pro_vision_prompt(prompt_mode: str) -> str:
    """Return the configured MMMU-Pro vision prompt."""
    try:
        return MMMU_PRO_VISION_PROMPTS[prompt_mode]
    except KeyError as error:
        valid_modes = ", ".join(sorted(MMMU_PRO_VISION_PROMPTS))
        raise ValueError(
            "Invalid MMMU-Pro vision prompt_mode "
            f"{prompt_mode!r}; expected one of: {valid_modes}."
        ) from error


def extract_answer(generated_text: str) -> str | None:
    """
    Extract the answer letter (A-J) from model output.

    Args:
        generated_text: The raw text generated by the model.

    Returns:
        The extracted answer letter, or ``None`` when no valid answer is found.
    """
    text = generated_text.strip()
    if not text:
        return None

    # Prefer the last explicit declaration. This supports reasoning that mentions
    # an earlier option before concluding with a final "Answer: X" line.
    explicit_answers: list[tuple[int, str]] = []
    for pattern in _ANSWER_DECLARATION_PATTERNS:
        explicit_answers.extend(
            (match.start(), match.group(1)) for match in pattern.finditer(text)
        )
    if explicit_answers:
        _, answer = max(explicit_answers, key=lambda item: item[0])
        return answer.upper()

    # A leading answer is accepted only with answer-like punctuation. In
    # particular, plain prose such as "I think ..." must not be parsed as I.
    leading_match = _LEADING_ANSWER_PATTERN.match(text)
    if leading_match:
        return (leading_match.group(1) or leading_match.group(2)).upper()

    # Accept a bare answer when it occupies the whole response or its final
    # non-empty line.
    candidates = [text]
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines and non_empty_lines[-1] != text:
        candidates.append(non_empty_lines[-1])

    for candidate in candidates:
        bare_match = _BARE_ANSWER_PATTERN.fullmatch(candidate)
        if bare_match:
            return bare_match.group(1).upper()

    return None


def load_few_shot_examples(
    dataset: str,
    split: str,
    subject: str,
    n_shots: int = 5,
) -> list[dict[str, Any]]:
    """
    Load few-shot examples for a given MMMU subject from the 'dev' split.

    Args:
        dataset: Dataset name.
        split: Split name (e.g. 'train', 'test', 'validation').
        subject: The subject name.
        n_shots: Number of few-shot examples to load.

    Returns:
        List of example dictionaries with 'question', 'choices', and 'answer'.
    """
    if n_shots <= 0:
        return []

    ds = load_data(
        dataset=dataset,
        subject=subject,
        split=split,
        start=0,
        n_samples=n_shots,
        streaming=True,
    )

    return [get_item_mmmu(ex) for ex in ds]


def is_mmmu_pro_vision(dataset: str, subject: str) -> bool:
    return dataset == "MMMU/MMMU_Pro" and subject == "vision"


def resolve_mmmu_max_new_tokens(
    max_new_tokens: int | None,
    *,
    dataset: str,
    subject: str,
    prompt_mode: str = DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
) -> int:
    """Resolve and validate the generation budget for one MMMU subject."""
    if is_mmmu_pro_vision(dataset, subject):
        get_mmmu_pro_vision_prompt(prompt_mode)
        default = MMMU_PRO_VISION_DEFAULT_MAX_NEW_TOKENS[prompt_mode]
    else:
        default = DEFAULT_MMMU_MAX_NEW_TOKENS

    resolved = default if max_new_tokens is None else int(max_new_tokens)
    if resolved <= 0:
        raise ValueError(
            f"max_new_tokens must be a positive integer, got {max_new_tokens!r}."
        )
    return resolved


def evaluate_subject(
    model,
    processor,
    dataset: str,
    eval_split: str,
    few_shot_split: str,
    subject: str,
    device: str | torch.device,
    max_new_tokens: int | None,
    n_shots: int = 5,
    n_samples: int = -1,
    max_seq_len: int | None = None,
    temperature: float = 0.0,
    verbose: bool = True,
    prompt_mode: str = DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
) -> tuple[int, int, int]:
    """
    Evaluate model accuracy on a single MMMU subject.

    Args:
        model: Language model with generation capability.
        processor: Matching processor for the model.
        dataset: Dataset name.
        eval_split: Split name for evaluation (e.g. 'train', 'test', 'validation').
        few_shot_split: Split name for few-shot examples (e.g. 'train', 'test', 'validation').
        subject: The MMMU subject to evaluate.
        device: Device for inference.
        n_shots: Number of few-shot examples.
        n_samples: Number of test samples. Use -1 for full test set.
        max_new_tokens: Maximum tokens to generate. Use ``None`` for the
            benchmark- and prompt-mode-specific default.
        temperature: Sampling temperature.
        verbose: Whether to print detailed logs.
        prompt_mode: Prompt mode for the MMMU-Pro vision subset.

    Returns:
        A tuple of (correct_count, total_count, skipped_count).
    """
    vision_only = is_mmmu_pro_vision(dataset, subject)
    effective_max_new_tokens = resolve_mmmu_max_new_tokens(
        max_new_tokens,
        dataset=dataset,
        subject=subject,
        prompt_mode=prompt_mode,
    )
    if vision_only:
        if n_shots > 0 and verbose:
            print(
                "\n[WARNING] MMMU-Pro vision subset is evaluated image-only; "
                f"ignoring n_shots={n_shots}."
            )
        few_shot_examples: list[dict[str, Any]] = []
    else:
        few_shot_examples = load_few_shot_examples(
            dataset=dataset, split=few_shot_split, subject=subject, n_shots=n_shots
        )

    # If we take few-shot examples from the same split as evaluation examples,
    # then exclude few-shot examples from the evaluation set by adjusting start argument to load_data.
    if few_shot_examples and eval_split == few_shot_split:
        start = n_shots
    else:
        start = 0

    test_data = load_data(
        dataset=dataset,
        subject=subject,
        split=eval_split,
        start=start,
        n_samples=n_samples,
        streaming=True,
    )

    correct = 0
    total = 0
    skipped = 0

    ex: dict[str, Any]
    for ex in test_data:
        # Skip questions with multiple images
        if "image_2" in ex and ex["image_2"] is not None:
            skipped += 1
            if verbose:
                question: str = ex["question"]
                print(f"\n[WARNING]: Skipped question '{question[:100]}...'")
            continue

        item = get_item_mmmu(ex)

        if vision_only:
            prompt = get_mmmu_pro_vision_prompt(prompt_mode)
        else:
            prompt = build_few_shot_prompt(
                question=item["question"],
                choices=item["choices"],
                subject=subject,
                few_shot_examples=few_shot_examples,
            )

        try:
            if vision_only:
                generated = generate_image_only_answer(
                    model=model,
                    processor=processor,
                    image=item["image"],
                    question=prompt,
                    device=device,
                    max_new_tokens=effective_max_new_tokens,
                    max_seq_len=max_seq_len,
                    temperature=temperature,
                )
            else:
                generated = generate_answer(
                    model=model,
                    processor=processor,
                    question=prompt,
                    image=item["image"],
                    device=device,
                    max_new_tokens=effective_max_new_tokens,
                    max_seq_len=max_seq_len,
                    temperature=temperature,
                )
        except ValueError as error:
            if "Mismatch in `image` token count between text and `input_ids`." in str(
                error
            ):
                if verbose:
                    print(
                        f"\n[WARNING] prompt too long for the specified max_seq_len={max_seq_len}. Skipping."
                    )
                    print(f"Error: {error}")
                    print(f"Prompt: {prompt}")
                skipped += 1
                continue
            else:
                raise error
        except RuntimeError as error:
            if verbose:
                print(f"[ERROR]: {error}")
                print(f"Prompt: {prompt}")
            skipped += 1
            continue

        predicted = extract_answer(generated)
        gold = item["answer"].upper()

        is_correct = predicted == gold
        correct += int(is_correct)
        total += 1

        if verbose:
            print(f"\n[Sample {total}] Subject: {subject}")
            if vision_only:
                print("Q: <embedded in image>")
            else:
                print(f"Q: {item['question'][:100]}...")
            print(f"Choices: {item['choices']}")
            print(
                f"Generated: {generated}, Predicted: {predicted}, Gold: {gold}, Correct: {is_correct}"
            )

    return correct, total, skipped


def evaluate_mmmu(
    model,
    processor,
    dataset: str,
    subjects: list[str] | None = None,
    device: str | torch.device = "cuda",
    n_shots: int = 5,
    n_samples: int = -1,
    max_new_tokens: int | None = None,
    max_seq_len: int | None = None,
    temperature: float = 0.0,
    verbose: bool = True,
    prompt_mode: str = DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
) -> dict[str, tuple[int, int, int]]:
    """
    Evaluate a model on the MMMU benchmark.

    Args:
        model: Language model with generation capability.
        processor: Matching processor for the model.
        dataset: Dataset name.
        subjects: List of subjects to evaluate. Use None for all subjects.
        device: Device for inference.
        n_shots: Number of few-shot examples per subject.
        n_samples: Number of test samples per subject. Use -1 for full test sets.
        max_new_tokens: Maximum tokens to generate per question. Use ``None``
            for the benchmark- and prompt-mode-specific default.
        temperature: Sampling temperature. Use 0.0 for greedy decoding.
        verbose: Whether to print progress.
        prompt_mode: Prompt mode for the MMMU-Pro vision subset.

    Returns:
        Aggregated results dictionary in '{ subject: (correct, total, skipped) }' format.
    """
    if dataset not in MMMU_DATASETS:
        raise ValueError(f"Invalid dataset '{dataset}'")

    if subjects is None:
        subjects = MMMU_SUBJECTS[dataset]

    eval_split = "validation" if dataset == "MMMU/MMMU" else "test"
    few_shot_split = "test"

    # { subject: (correct, total) }
    results: dict[str, tuple[int, int, int]] = {}

    for i, subject in enumerate(subjects, 1):
        if verbose:
            print(f"\n[{i}/{len(subjects)}] Evaluating {subject}...")

        correct, total, skipped = evaluate_subject(
            model=model,
            processor=processor,
            dataset=dataset,
            eval_split=eval_split,
            few_shot_split=few_shot_split,
            subject=subject,
            device=device,
            n_shots=n_shots,
            n_samples=n_samples,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            temperature=temperature,
            verbose=verbose,
            prompt_mode=prompt_mode,
        )
        results[subject] = (correct, total, skipped)

        if verbose:
            accuracy = correct / total if total > 0 else 0.0
            skipped_str = f", skipped: {skipped}" if skipped > 0 else ""
            print(f"  {subject}: {accuracy:.4f} ({correct}/{total}){skipped_str}")

    return results


def print_mmmu_results(results: dict[str, Any]) -> None:
    """
    Print MMMU evaluation results in a formatted table.

    Args:
        results: Per-subject results in '{ subject: (correct, total) }' format.
    """
    subject: str
    correct: int
    total: int
    skipped: int
    print(
        f"| {'subject':<50} | {'correct':<10} | {'total':<10} | {'skipped':<10} | {'accuracy':<10} |"
    )
    print(f"| {'-'*50} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} |")
    for subject, (correct, total, skipped) in results.items():
        accuracy = correct / total
        print(
            f"| {subject:<50} | {correct:<10} | {total:<10} | {skipped:<10} | {accuracy:<10.4f} |"
        )
