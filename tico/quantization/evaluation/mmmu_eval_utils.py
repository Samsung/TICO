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
import json
import math
import re
import string
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from tico.quantization.evaluation.vlm_generation_utils import (
    positive_int,
    resolve_generation_input_budget,
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


# Prompts from MMMU-Benchmark/MMMU, mmmu-pro/prompts.yaml. The prompt names
# do not imply equivalence to any external evaluator's complete protocol.
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
STRICT_SINGLE_LETTER_PROMPT_MODE = "strict_single_letter"
MMMU_PRO_VISION_PROMPT_MODES: tuple[str, ...] = (
    STRICT_SINGLE_LETTER_PROMPT_MODE,
    "official_direct",
    "official_cot",
)
DEFAULT_MMMU_PRO_VISION_PROMPT_MODE = STRICT_SINGLE_LETTER_PROMPT_MODE
DEFAULT_MMMU_MAX_NEW_TOKENS = 16
# Per-mode default generation budgets for MMMU-Pro vision when
# ``max_new_tokens`` is None. ``None`` means the mode has no implicit budget
# and an explicit value is required. Non-vision subjects always use
# ``DEFAULT_MMMU_MAX_NEW_TOKENS``.
MMMU_PRO_VISION_DEFAULT_MAX_NEW_TOKENS: dict[str, int | None] = {
    STRICT_SINGLE_LETTER_PROMPT_MODE: 16,
    "official_direct": 256,
    "official_cot": None,
}
# Single-letter option labels; the actual choice count of each sample selects
# the valid prefix of this alphabet for both the prompt and the parser.
MMMU_MAX_CHOICES = 26
# Multiple-choice parser. v4: a bare label followed by ``-`` is recognised as
# the last declaration so that ``Answer: C-D`` is rejected as ambiguous instead
# of falling back to an earlier ``Answer: B``.
MMMU_ANSWER_PARSER_VERSION = "tico-mmmu-v4"
# ``question_type`` is absent in MMMU-Pro rows. Explicit multiple-choice
# metadata is also accepted; open-ended scoring is not supported.
MMMU_QUESTION_TYPE_MULTIPLE_CHOICE = "multiple-choice"
MMMU_QUESTION_TYPE_OPEN = "open"
MMMU_SCORING_MULTIPLE_CHOICE = "multiple_choice"
MMMU_EVALUATION_DATASETS = ("MMMU/MMMU_Pro",)

_STRICT_SINGLE_LETTER_PROMPT_PREFIX = (
    "You are a strict single-letter answering machine. "
    "Answer the multiple-choice question shown in the image. "
)

# Require a declaration separator, rather than matching the verb in
# "To answer a question ...". A parenthesised label ends at its closing
# parenthesis (``Answer: (H)HENRI`` -> H); a bare label needs a boundary so
# that ``Answer: HENRI`` yields nothing. Labels are case-insensitive;
# article-like lowercase letters followed by prose are rejected below. The
# boundary set includes every character that ``_AMBIGUOUS_TAIL_PATTERN``
# treats as a joiner (``/``, ``-``), so ``Answer: C-D`` is recognised as the
# last declaration and then rejected, rather than skipped in favour of an
# earlier ``Answer: B``.
_EXPLICIT_ANSWER_PATTERN = re.compile(
    r"\b(?i:(?:(?:final|correct)\s+)?answer)\s*"
    r"(?i::|=|\bis\b)\s*(?i:option\s+)?"
    r"(?:\(([A-Za-z])\)|([A-Za-z])(?=$|[\s.,;:!?)/-]))"
)
_BARE_ANSWER_PATTERN = re.compile(r"\s*\(?([A-Za-z])\)?[.):]?\s*")
# ``(B)Yes.`` is a leading answer; ``B. text`` still needs a space after the
# punctuation so that abbreviations are not read as labels.
_LEADING_ANSWER_PATTERN = re.compile(r"^\s*(?:\(([A-Z])\)|([A-Z])[.):](?=\s|$))")
_CHOICE_LINE_PATTERN = re.compile(
    r"(?i:(?:I\s+(?:would\s+)?(?:choose|select)\s+(?:option\s+)?|"
    r"(?:the\s+)?(?:correct\s+)?(?:option|choice)\s*(?::|=|is)?\s*))"
    r"\(?([A-Za-z])\)?[.!]?"
)
_AMBIGUOUS_TAIL_PATTERN = re.compile(r"(?i)^(?:or|and)\b|^[/-]")


def generate_answer(*args: Any, **kwargs: Any) -> str:
    """Load optional VLM dependencies only when generation is requested."""
    from tico.quantization.evaluation.vlm_eval_utils import generate_answer as impl

    return impl(*args, **kwargs)


def generate_image_only_answer(*args: Any, **kwargs: Any) -> str:
    """Load optional VLM dependencies only when generation is requested."""
    from tico.quantization.evaluation.vlm_eval_utils import (
        generate_image_only_answer as impl,
    )

    return impl(*args, **kwargs)


def mmmu_choice_labels(num_choices: int) -> str:
    """Return the valid single-letter labels ``A..`` for ``num_choices`` (1-26).

    Rejects booleans, fractions, zero, negative values and counts above
    ``MMMU_MAX_CHOICES`` instead of clamping them.
    """
    count = positive_int(num_choices, "num_choices")
    if count > MMMU_MAX_CHOICES:
        raise ValueError(
            f"num_choices must be between 1 and {MMMU_MAX_CHOICES}, got {count}."
        )
    return string.ascii_uppercase[:count]


def validate_mmmu_pro_vision_prompt_mode(prompt_mode: Any) -> str:
    if (
        not isinstance(prompt_mode, str)
        or prompt_mode not in MMMU_PRO_VISION_PROMPT_MODES
    ):
        raise ValueError(
            f"Invalid MMMU-Pro vision prompt_mode {prompt_mode!r}; "
            f"expected one of {sorted(MMMU_PRO_VISION_PROMPT_MODES)}."
        )
    return prompt_mode


def get_mmmu_pro_vision_prompt(prompt_mode: str, num_choices: int | None = None) -> str:
    """Return the image-only prompt for a mode.

    ``strict_single_letter`` requires the sample's actual ``num_choices``; only
    the last allowed label is derived from it (ten choices reproduce the
    historical ``A to J`` wording exactly). Official prompts are fixed strings
    and never mention the choice count. No question, option text or gold
    answer is injected on this path.
    """
    validate_mmmu_pro_vision_prompt_mode(prompt_mode)
    if prompt_mode == STRICT_SINGLE_LETTER_PROMPT_MODE:
        if num_choices is None:
            raise ValueError(
                "strict_single_letter requires the sample's actual num_choices."
            )
        labels = mmmu_choice_labels(num_choices)
        if len(labels) == 1:
            return _STRICT_SINGLE_LETTER_PROMPT_PREFIX + "Return only the letter A."
        return (
            _STRICT_SINGLE_LETTER_PROMPT_PREFIX
            + f"Return only one letter from A to {labels[-1]}."
        )
    if num_choices is not None:
        mmmu_choice_labels(num_choices)
    return MMMU_PRO_VISION_PROMPTS[prompt_mode]


def resolve_mmmu_max_new_tokens(
    max_new_tokens: int | None,
    *,
    dataset: str,
    subject: str,
    prompt_mode: str = DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
) -> int:
    """Resolve the generation budget; an explicit value always wins.

    ``None`` selects the per-mode default from
    ``MMMU_PRO_VISION_DEFAULT_MAX_NEW_TOKENS`` for MMMU-Pro vision and the
    16-token default elsewhere. CoT has no implicit budget. In particular, do
    not insert a 16K budget into a profile whose entire input-plus-output
    capacity is only 2048.
    """
    vision_only = is_mmmu_pro_vision(dataset, subject)
    if vision_only:
        validate_mmmu_pro_vision_prompt_mode(prompt_mode)
    if max_new_tokens is not None:
        return positive_int(max_new_tokens, "max_new_tokens")
    if not vision_only:
        return DEFAULT_MMMU_MAX_NEW_TOKENS
    default = MMMU_PRO_VISION_DEFAULT_MAX_NEW_TOKENS[prompt_mode]
    if default is None:
        raise ValueError(
            f"{prompt_mode} requires an explicit max_new_tokens budget "
            "compatible with max_seq_len."
        )
    return default


def _is_ambiguous_candidate(candidate: str, parenthesised: bool, tail: str) -> bool:
    # ``Answer: C or D`` / ``(C)/(D)`` are not a single choice. A bare
    # lowercase letter followed by prose is an article, not a label.
    if _AMBIGUOUS_TAIL_PATTERN.match(tail):
        return True
    return (
        not parenthesised
        and candidate.islower()
        and bool(tail)
        and tail[0] not in ".,;:!?"
    )


def _parse_answer(generated_text: str, num_choices: int = 10) -> tuple[str | None, str]:
    """Parse one answer label from generated text.

    Supported grammar, in priority order:

    1. the last explicit declaration ``[final|correct] answer (:|=|is) X``
       where ``X`` is ``(X)`` followed by anything or a bare letter followed by
       end, whitespace or punctuation;
    2. a bare ``X`` / ``(X)`` / ``X.`` on the last non-empty line;
    3. an unambiguous choice-only last line (``I would choose B.``);
    4. a leading ``(X)`` followed by anything, or ``X.`` followed by a space.

    A candidate followed by ``or``/``and``/``/``/``-`` is ambiguous and yields
    ``None`` without falling back to an earlier mention; ``-`` and ``/`` are
    also label boundaries so that ``Answer: C-D`` and ``Answer: C/D`` are seen
    as the last declaration. A label outside the sample's ``num_choices``
    yields ``None`` (``out_of_range``). Callers that omit ``num_choices`` get
    the historical ten-label default.
    """
    labels = mmmu_choice_labels(num_choices)
    # Markdown emphasis and code fences do not change the answer's meaning.
    text = generated_text.translate(str.maketrans("", "", "*_`")).strip()
    if not text:
        return None, "unparsed"

    answer: str | None = None
    method = "unparsed"
    match: re.Match[str] | None
    declarations = list(_EXPLICIT_ANSWER_PATTERN.finditer(text))
    if declarations:
        match = declarations[-1]
        candidate = match.group(1) or match.group(2)
        tail = text[match.end() :].split("\n", 1)[0].strip()
        if _is_ambiguous_candidate(candidate, match.group(1) is not None, tail):
            return None, "ambiguous_declaration"
        answer, method = candidate.upper(), "explicit"
    else:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        match = _BARE_ANSWER_PATTERN.fullmatch(lines[-1])
        if match:
            answer, method = match.group(1).upper(), "bare"
        else:
            match = _CHOICE_LINE_PATTERN.fullmatch(lines[-1])
            if match:
                answer, method = match.group(1).upper(), "choice_line"
            else:
                match = _LEADING_ANSWER_PATTERN.match(text)
                if match:
                    candidate = match.group(1) or match.group(2)
                    tail = text[match.end() :].split("\n", 1)[0].strip()
                    if _is_ambiguous_candidate(
                        candidate, match.group(1) is not None, tail
                    ):
                        return None, "ambiguous_leading"
                    answer, method = candidate.upper(), "leading"
    if answer is not None and answer not in labels:
        return None, "out_of_range"
    return answer, method


def extract_answer(generated_text: str, num_choices: int = 10) -> str | None:
    """Extract an answer label or None; explicit answers outrank option mentions.

    ``num_choices`` (1-26) bounds the valid labels; omitting it keeps the
    historical ten-label default, so evaluators must pass the real count.
    This is a versioned TICO parser, not a claim of official scoring parity.
    An unparseable or out-of-range answer is counted as wrong, not skipped.
    """
    return _parse_answer(generated_text, num_choices)[0]


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

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as error:
        if error.name != "datasets":
            raise
        raise RuntimeError(
            "The optional 'datasets' package is required for MMMU evaluation. "
            "Install the optional evaluation dependencies."
        ) from error

    ds: Iterable[dict[str, Any]] = load_dataset(
        path=dataset,
        name=subject,
        split=split,
        streaming=streaming,
    )

    if n_samples >= 0 or start > 0:
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
        # MMMU/MMMU rows say ``multiple-choice`` or ``open``; MMMU-Pro rows and
        # older callers have no such field, which is treated as multiple-choice.
        "question_type": ex.get("question_type"),
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

    examples: list[dict[str, Any]] = []
    for ex in ds:
        item = get_item_mmmu(ex)
        validate_mmmu_question_type(str(item["id"]), item["question_type"])
        validate_sample_choices(str(item["id"]), item["choices"], item["answer"])
        examples.append(item)
    return examples


def is_mmmu_pro_vision(dataset: str, subject: str) -> bool:
    return dataset == "MMMU/MMMU_Pro" and subject == "vision"


def normalize_mmmu_gold_label(sample_id: str, gold: Any, labels: str) -> str:
    """Return the single upper-case label that a multiple-choice gold denotes.

    The gold must be a string that, after ``strip().upper()``, is exactly one
    character contained in ``labels``. Empty strings, multi-letter strings
    (``"AB"``, ``"A,B"``), labels outside the sample's range, and non-string
    values (``None``, ``1``, ``True``) raise a ``ValueError`` naming the sample;
    they are never converted. Validation, scoring and logging all use the value
    returned here, so ``" c "`` is consistently ``"C"``.
    """
    if not isinstance(gold, str):
        raise ValueError(
            f"Sample {sample_id!r}: gold answer must be a string label, "
            f"got {type(gold).__name__}."
        )
    normalized = gold.strip().upper()
    if len(normalized) != 1 or normalized not in labels:
        raise ValueError(
            f"Sample {sample_id!r}: gold answer {gold!r} is not exactly one of "
            f"the valid labels {labels!r}."
        )
    return normalized


def validate_sample_choices(sample_id: str, choices: Any, gold: Any) -> str:
    """Return the valid labels of one multiple-choice sample, or fail first.

    Unsupported inputs (no options, more than ``MMMU_MAX_CHOICES`` options, a
    non-list option structure, or a gold answer that is not exactly one of the
    labels, see ``normalize_mmmu_gold_label``) raise a ``ValueError`` naming
    the sample before any generation. They are neither skipped nor clamped.
    Question-type validation rejects open-ended questions before this check.
    """
    if not isinstance(choices, (list, tuple)):
        raise ValueError(
            f"Sample {sample_id!r}: options must be a list, "
            f"got {type(choices).__name__}."
        )
    try:
        labels = mmmu_choice_labels(len(choices))
    except ValueError as error:
        raise ValueError(
            f"Sample {sample_id!r}: unsupported number of choices "
            f"({len(choices)}): {error}"
        ) from error
    normalize_mmmu_gold_label(sample_id, gold, labels)
    return labels


def validate_mmmu_evaluation_dataset(dataset: str) -> None:
    """Reject mixed/open-ended evaluation before loading data or generating.

    Data-loading helpers still recognize the original MMMU for other callers;
    evaluation supports the three multiple-choice MMMU-Pro configurations only.
    No automatic filtering or benchmark substitution is performed.
    """
    if dataset == "MMMU/MMMU":
        raise ValueError(
            "MMMU/MMMU mixes multiple-choice and open-ended questions. "
            "Open-ended MMMU scoring is not supported by this evaluator. "
            "Use dataset='MMMU/MMMU_Pro' with an explicit subject "
            "('vision', 'standard (4 options)', or 'standard (10 options)'). "
            "No questions were filtered and no benchmark was substituted."
        )
    if dataset not in MMMU_EVALUATION_DATASETS:
        raise ValueError(f"Invalid MMMU evaluation dataset {dataset!r}.")


def validate_mmmu_question_type(sample_id: str, question_type: Any) -> None:
    """Require multiple-choice metadata; never reinterpret or skip open rows."""
    if question_type is None or question_type == MMMU_QUESTION_TYPE_MULTIPLE_CHOICE:
        return
    if question_type == MMMU_QUESTION_TYPE_OPEN:
        raise ValueError(
            f"Sample {sample_id!r}: open-ended MMMU scoring is not supported. "
            "Only multiple-choice samples are supported; this sample was not skipped."
        )
    raise ValueError(
        f"Sample {sample_id!r}: unsupported question_type {question_type!r}; "
        f"expected {MMMU_QUESTION_TYPE_MULTIPLE_CHOICE!r} or missing metadata."
    )


def _validate_subject_options(
    *,
    dataset: str,
    subject: str,
    max_new_tokens: int | None,
    max_seq_len: int | None,
    input_max_seq_len: int | None,
    prompt_mode: str,
    temperature: float,
) -> tuple[int, int | None]:
    validate_mmmu_evaluation_dataset(dataset)
    if subject not in MMMU_SUBJECTS[dataset]:
        raise ValueError(f"Invalid subject '{subject}'")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative.")
    budget = resolve_mmmu_max_new_tokens(
        max_new_tokens, dataset=dataset, subject=subject, prompt_mode=prompt_mode
    )
    input_budget = resolve_generation_input_budget(
        max_seq_len, budget, input_max_seq_len
    )
    return budget, input_budget


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
    *,
    prompt_mode: str = DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
    input_max_seq_len: int | None = None,
    record_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, int, int]:
    """Evaluate one subject, preserving (correct, total, skipped) results.

    An explicit input cap is independent of the generation budget. Optional
    records contain raw outputs and all skip decisions for paired evaluation.
    Existing runtime-error skips are preserved; other unexpected errors fail.
    """
    budget, input_budget = _validate_subject_options(
        dataset=dataset,
        subject=subject,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        input_max_seq_len=input_max_seq_len,
        prompt_mode=prompt_mode,
        temperature=temperature,
    )
    vision_only = is_mmmu_pro_vision(dataset, subject)
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
    start = n_shots if few_shot_examples and eval_split == few_shot_split else 0
    test_data = load_data(
        dataset=dataset,
        subject=subject,
        split=eval_split,
        start=start,
        n_samples=n_samples,
        streaming=True,
    )
    correct = total = skipped = parse_failures = 0
    length_limited = eos_count = unknown_stops = 0
    effective_mode = prompt_mode if vision_only else "few_shot"
    if record_callback is not None:
        record_callback(
            {
                "record_type": "subject_start",
                "subject": subject,
                "split": eval_split,
                "start_index": start,
                "n_shots": len(few_shot_examples),
                "prompt_mode": effective_mode,
                "max_new_tokens": budget,
                "input_max_seq_len": input_budget,
                "parser_version": MMMU_ANSWER_PARSER_VERSION,
            }
        )

    for sample_index, ex in enumerate(test_data, start=start):
        record: dict[str, Any] = {
            "record_type": "sample",
            "dataset": dataset,
            "subject": subject,
            "split": eval_split,
            "sample_index": sample_index,
            "id": str(ex["id"]),
            "prompt_mode": effective_mode,
        }
        validate_mmmu_question_type(record["id"], ex.get("question_type"))
        if "image_2" in ex and ex["image_2"] is not None:
            skipped += 1
            record.update(status="skipped", skip_reason="multiple_images")
            if record_callback is not None:
                record_callback(record)
            if verbose:
                print(f"\n[WARNING] Skipped multi-image sample {record['id']}.")
            continue

        item = get_item_mmmu(ex)
        # Unsupported metadata or option structures fail here, before
        # generation, with the sample ID; they are not skipped or clamped.
        labels = validate_sample_choices(record["id"], item["choices"], item["answer"])
        num_choices = len(labels)
        gold = normalize_mmmu_gold_label(record["id"], item["answer"], labels)
        if vision_only:
            prompt = get_mmmu_pro_vision_prompt(prompt_mode, num_choices)
        else:
            prompt = build_few_shot_prompt(
                question=item["question"],
                choices=item["choices"],
                subject=subject,
                few_shot_examples=few_shot_examples,
            )
        generation: dict[str, Any] | None = {} if record_callback is not None else None
        # No tensor hashing, token-id copies or JSONL I/O on the default path.
        generation_kwargs: dict[str, Any] = {}
        if input_max_seq_len is not None:
            generation_kwargs["input_max_seq_len"] = input_max_seq_len
        if generation is not None:
            generation_kwargs["diagnostics"] = generation
        record.update(
            prompt=prompt,
            question_type=item["question_type"],
            scoring_policy=MMMU_SCORING_MULTIPLE_CHOICE,
            parser_version=MMMU_ANSWER_PARSER_VERSION,
            gold=gold,
            gold_raw=item["answer"],
            num_choices=num_choices,
            valid_labels=labels,
        )
        try:
            generate = generate_image_only_answer if vision_only else generate_answer
            generated = generate(
                model=model,
                processor=processor,
                image=item["image"],
                question=prompt,
                device=device,
                max_new_tokens=budget,
                max_seq_len=max_seq_len,
                temperature=temperature,
                **generation_kwargs,
            )
        except (ValueError, RuntimeError) as error:
            is_token_mismatch = isinstance(
                error, ValueError
            ) and "Mismatch in `image` token count between text and `input_ids`." in str(
                error
            )
            # Preserve the existing skip policy, but make every skip auditable.
            can_skip = is_token_mismatch or isinstance(error, RuntimeError)
            record.update(
                status="skipped" if can_skip else "error",
                skip_reason=(
                    "image_token_mismatch"
                    if is_token_mismatch
                    else type(error).__name__
                ),
                error=str(error),
                generation=generation,
            )
            if record_callback is not None:
                record_callback(record)
            if not can_skip:
                raise
            skipped += 1
            if verbose:
                print(f"[WARNING] Skipped {record['id']}: {error}")
            continue

        predicted, parse_method = _parse_answer(generated, num_choices)
        parse_failed = predicted is None
        is_correct = predicted == gold
        correct += int(is_correct)
        total += 1
        parse_failures += int(parse_failed)
        if generation is not None:
            length_limited += int(generation.get("length_limited") is True)
            eos_count += int(generation.get("eos_observed") is True)
            unknown_stops += int(generation.get("stop_reason", "unknown") == "unknown")
        record.update(
            status="evaluated",
            generated=generated,
            predicted=predicted,
            correct=is_correct,
            parse_method=parse_method,
            parse_failed=parse_failed,
            generation=generation,
        )
        if record_callback is not None:
            record_callback(record)
        if verbose:
            print(f"\n[Sample {total}] Subject: {subject}, ID: {record['id']}")
            print(
                "Q: <embedded in image>"
                if vision_only
                else f"Q: {item['question'][:100]}..."
            )
            print(f"Choices: {item['choices']}")
            print(
                f"Generated: {generated}, Predicted: {predicted}, "
                f"Gold: {gold}, Correct: {is_correct}, Parser: {parse_method}"
            )

    if record_callback is not None:
        record_callback(
            {
                "record_type": "subject_summary",
                "subject": subject,
                "correct": correct,
                "total": total,
                "skipped": skipped,
                "parse_failures": parse_failures,
                "parse_failure_rate": parse_failures / total if total else None,
                "length_limited": length_limited,
                "eos_count": eos_count,
                "unknown_stops": unknown_stops,
                "accuracy": correct / total if total else None,
            }
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
    *,
    prompt_mode: str = DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
    input_max_seq_len: int | None = None,
    output_jsonl: str | Path | None = None,
) -> dict[str, tuple[int, int, int]]:
    """Evaluate multiple-choice MMMU-Pro with opt-in diagnostic records.

    ``output_jsonl`` is created exclusively: existing logs are never overwritten
    or appended to. A successful run ends with a ``run_summary`` record. Failed
    runs retain the already flushed records for diagnosis, not resume.
    """
    validate_mmmu_evaluation_dataset(dataset)
    if subjects is None:
        subjects = MMMU_SUBJECTS[dataset]
    if not isinstance(subjects, list) or any(not isinstance(s, str) for s in subjects):
        raise ValueError("subjects must be a list of subject names or None.")
    if len(set(subjects)) != len(subjects):
        raise ValueError("subjects must not contain duplicates.")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < -1:
        raise ValueError("n_samples must be -1 or a non-negative integer.")
    if isinstance(n_shots, bool) or not isinstance(n_shots, int) or n_shots < 0:
        raise ValueError("n_shots must be a non-negative integer.")
    # Validate ALL subjects before opening a log, loading data or generating.
    for subject in subjects:
        _validate_subject_options(
            dataset=dataset,
            subject=subject,
            max_new_tokens=max_new_tokens,
            max_seq_len=max_seq_len,
            input_max_seq_len=input_max_seq_len,
            prompt_mode=prompt_mode,
            temperature=temperature,
        )
    if output_jsonl is not None and not isinstance(output_jsonl, (str, Path)):
        raise ValueError("output_jsonl must be a file path or None.")
    if output_jsonl == "":
        raise ValueError("output_jsonl must not be empty.")

    eval_split = "validation" if dataset == "MMMU/MMMU" else "test"
    few_shot_split = "test"
    results: dict[str, tuple[int, int, int]] = {}
    with ExitStack() as stack:
        record_callback: Callable[[dict[str, Any]], None] | None = None
        if output_jsonl is not None:
            log_path = Path(output_jsonl)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(log_path.open("x", encoding="utf-8"))

            def _write_record(record: dict[str, Any]) -> None:
                stream.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                stream.flush()

            record_callback = _write_record
            record_callback(
                {
                    "record_type": "run_start",
                    "schema_version": 2,
                    "parser_version": MMMU_ANSWER_PARSER_VERSION,
                    "dataset": dataset,
                    "subjects": subjects,
                    "n_samples": n_samples,
                    "n_shots": n_shots,
                    "max_seq_len": max_seq_len,
                    "input_max_seq_len": input_max_seq_len,
                    "max_new_tokens": max_new_tokens,
                    "prompt_mode": prompt_mode,
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "torch_version": str(torch.__version__),
                    "torch_initial_seed": torch.initial_seed(),
                    "model_class": type(model).__name__,
                    "processor_class": type(processor).__name__,
                }
            )
        try:
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
                    input_max_seq_len=input_max_seq_len,
                    record_callback=record_callback,
                )
                results[subject] = (correct, total, skipped)
                if verbose:
                    accuracy = correct / total if total else 0.0
                    print(
                        f"  {subject}: {accuracy:.4f} ({correct}/{total}), skipped: {skipped}"
                    )
        except Exception as error:
            if record_callback is not None:
                record_callback(
                    {
                        "record_type": "run_error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
            raise
        if record_callback is not None:
            record_callback({"record_type": "run_summary", "results": results})
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
        accuracy = correct / total if total else 0.0
        print(
            f"| {subject:<50} | {correct:<10} | {total:<10} | {skipped:<10} | {accuracy:<10.4f} |"
        )
