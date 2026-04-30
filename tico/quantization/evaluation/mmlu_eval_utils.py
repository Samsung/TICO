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

"""
MMLU (Massive Multitask Language Understanding) evaluation utilities.

This module provides functionality for evaluating language models on the MMLU benchmark,
which tests knowledge and reasoning across 57 academic subjects through multiple-choice
questions.

The evaluation uses 5-shot prompting by default, where the model sees 5 example questions
with correct answers before answering the target question.
"""

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


# MMLU subjects organized by domain
MMLU_SUBJECTS = {
    "STEM": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "Social Sciences": [
        "business_ethics",
        "college_macroeconomics",
        "college_microeconomics",
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_aging",
        "human_sexuality",
        "professional_accounting",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
    "Other": [
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_medicine",
        "virology",
    ],
}

# All subjects flattened
ALL_SUBJECTS = []
for domain_subjects in MMLU_SUBJECTS.values():
    ALL_SUBJECTS.extend(domain_subjects)

# Dataset configuration
DATASET_CONFIG = {
    "mmlu": {
        "test_split": "test",
        "dev_split": "dev",  # for few-shot examples
        "candidates": ["cais/mmlu", "lukaemon/mmlu"],
    }
}


def normalize_subject_name(subject: str) -> str:
    """
    Normalize a subject name to match dataset conventions.

    Args:
        subject: Subject name, possibly with spaces or different casing.

    Returns:
        Normalized subject name with underscores and lowercase.
    """
    return subject.lower().replace(" ", "_").replace("-", "_")


def get_item_mmlu(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt an MMLU-style sample to a common evaluation format.

    The returned schema is:
    {
        "subject": subject name,
        "question": question text,
        "choices": list of 4 answer choices,
        "answer": correct answer letter ("A", "B", "C", or "D")
    }

    Args:
        ex: Raw dataset example.

    Returns:
        A normalized evaluation item.
    """
    # Handle answer format: some datasets use integer (0,1,2,3), others use string ("A","B","C","D")
    answer = ex.get("answer", "")
    if isinstance(answer, int):
        # Convert integer to letter (0->A, 1->B, 2->C, 3->D)
        answer = chr(ord("A") + answer)
    elif isinstance(answer, str):
        answer = answer.upper()
    else:
        answer = str(answer).upper() if answer else ""

    return {
        "subject": ex.get("subject", ""),
        "question": ex.get("question", ""),
        "choices": ex.get("choices", [])[:4],  # Ensure exactly 4 choices
        "answer": answer,
    }


def format_mmlu_question(
    question: str,
    choices: List[str],
    answer: Optional[str] = None,
) -> str:
    """
    Format a single MMLU question with choices.

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


def build_mmlu_prompt(
    question: str,
    choices: List[str],
    subject: str,
    few_shot_examples: List[Dict[str, Any]],
) -> str:
    """
    Build a few-shot MMLU prompt.

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
            format_mmlu_question(
                ex["question"],
                ex["choices"],
                ex["answer"],
            )
        )

    # Add target question
    prompt_parts.append("")
    prompt_parts.append(format_mmlu_question(question, choices, answer=None))
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def extract_answer_letter(generated_text: str) -> Optional[str]:
    """
    Extract the answer letter (A/B/C/D) from model output.

    This function tries multiple extraction strategies:
    1. First standalone letter A/B/C/D
    2. Letter after "Answer:" pattern
    3. First occurrence of any A/B/C/D character

    Args:
        generated_text: The raw text generated by the model.

    Returns:
        The extracted letter (A/B/C/D), or None if no valid answer found.
    """
    text = generated_text.strip()

    # Strategy 1: Look for "Answer: X" pattern
    answer_match = re.search(r"Answer:\s*([A-D])", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Strategy 2: Look for standalone letter at the beginning
    first_char_match = re.match(r"^([A-D])[.\s]", text, re.IGNORECASE)
    if first_char_match:
        return first_char_match.group(1).upper()

    # Strategy 3: Look for first occurrence of A/B/C/D
    for char in text:
        if char.upper() in "ABCD":
            return char.upper()

    return None


def load_few_shot_examples(
    subject: str,
    n_shots: int = 5,
) -> List[Dict[str, Any]]:
    """
    Load few-shot examples for a given MMLU subject.

    Args:
        subject: The subject name.
        n_shots: Number of few-shot examples to load.

    Returns:
        List of example dictionaries with 'question', 'choices', and 'answer'.
    """
    from datasets import load_dataset

    config = DATASET_CONFIG["mmlu"]
    dev_split = config["dev_split"]
    candidates = config["candidates"]

    subject = normalize_subject_name(subject)

    for dataset_name in candidates:
        try:
            ds = load_dataset(
                dataset_name,
                subject,
                split=dev_split,
                streaming=True,
            )
            examples = []
            for i, ex in enumerate(ds):
                if i >= n_shots:
                    break
                examples.append(get_item_mmlu(ex))
            if examples:
                return examples
        except Exception:
            continue

    # If no examples found, return empty list
    # Evaluation will proceed with 0-shot
    return []


def load_test_data(
    subject: str,
    n_samples: int = -1,
    streaming: bool = True,
) -> Iterable[Dict[str, Any]]:
    """
    Load test data for a given MMLU subject.

    Args:
        subject: The subject name.
        n_samples: Number of samples to load. Use -1 for the full dataset.
        streaming: Whether to use streaming mode.

    Returns:
        An iterable of test examples.

    Raises:
        RuntimeError: If the dataset cannot be loaded from any candidate source.
    """
    from datasets import load_dataset

    config = DATASET_CONFIG["mmlu"]
    test_split = config["test_split"]
    candidates = config["candidates"]

    subject = normalize_subject_name(subject)

    last_err: Optional[Exception] = None
    for dataset_name in candidates:
        try:
            ds = load_dataset(
                dataset_name,
                subject,
                split=test_split,
                streaming=streaming,
            )
            if n_samples > 0:
                ds = ds.take(n_samples)
            return ds
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(
        f"Failed to load MMLU test data for subject '{subject}'. "
        f"Candidates: {candidates}. Last error: {last_err}"
    )


@torch.no_grad()
def generate_answer(
    model,
    tokenizer,
    prompt: str,
    device: str | torch.device,
    max_new_tokens: int = 1,
    temperature: float = 0.0,
) -> str:
    """
    Generate an answer for a single MMLU question.

    Args:
        model: Language model with generation capability.
        tokenizer: Matching tokenizer for the model.
        prompt: The formatted prompt string.
        device: Device for inference.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Use 0.0 for greedy decoding.

    Returns:
        The decoded model output string.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **gen_kwargs)

    # Decode only the generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, input_len:]

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_subject(
    model,
    tokenizer,
    subject: str,
    device: str | torch.device,
    n_shots: int = 5,
    n_samples: int = -1,
    max_new_tokens: int = 1,
    temperature: float = 0.0,
    verbose: bool = True,
) -> Tuple[int, int]:
    """
    Evaluate model accuracy on a single MMLU subject.

    Args:
        model: Language model with generation capability.
        tokenizer: Matching tokenizer for the model.
        subject: The MMLU subject to evaluate.
        device: Device for inference.
        n_shots: Number of few-shot examples.
        n_samples: Number of test samples. Use -1 for full test set.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        verbose: Whether to print detailed logs.

    Returns:
        A tuple of (correct_count, total_count).
    """
    few_shot_examples = load_few_shot_examples(subject, n_shots)
    test_data = load_test_data(subject, n_samples)

    correct = 0
    total = 0

    for ex in test_data:
        item = get_item_mmlu(ex)

        prompt = build_mmlu_prompt(
            question=item["question"],
            choices=item["choices"],
            subject=subject,
            few_shot_examples=few_shot_examples,
        )

        generated = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        predicted = extract_answer_letter(generated)
        gold = item["answer"].upper()

        is_correct = predicted == gold
        correct += int(is_correct)
        total += 1

        if verbose and total <= 3:
            print(f"\n[Sample {total}] Subject: {subject}")
            print(f"Q: {item['question'][:100]}...")
            print(f"Predicted: {predicted}, Gold: {gold}, Correct: {is_correct}")

    return correct, total


def aggregate_results(
    results: Dict[str, Tuple[int, int]],
) -> Dict[str, Any]:
    """
    Aggregate MMLU results by subject and domain.

    Args:
        results: Dictionary mapping subject names to (correct, total) tuples.

    Returns:
        Aggregated results dictionary with:
        - 'subjects': Per-subject accuracy
        - 'domains': Per-domain accuracy
        - 'overall': Overall accuracy
    """
    # Per-subject accuracy
    subject_accuracy = {}
    for subject, (correct, total) in results.items():
        subject_accuracy[subject] = correct / total if total > 0 else 0.0

    # Per-domain accuracy
    domain_stats: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))
    for domain, subjects in MMLU_SUBJECTS.items():
        domain_correct = 0
        domain_total = 0
        for subject in subjects:
            if subject in results:
                c, t = results[subject]
                domain_correct += c
                domain_total += t
        if domain_total > 0:
            domain_stats[domain] = (domain_correct, domain_total)

    domain_accuracy = {}
    for domain, (correct, total) in domain_stats.items():
        domain_accuracy[domain] = correct / total if total > 0 else 0.0

    # Overall accuracy
    total_correct = sum(c for c, t in results.values())
    total_count = sum(t for c, t in results.values())
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    return {
        "subjects": subject_accuracy,
        "domains": domain_accuracy,
        "overall": overall_accuracy,
        "total_correct": total_correct,
        "total_count": total_count,
    }


def evaluate_mmlu(
    model,
    tokenizer,
    subjects: Optional[List[str]] = None,
    device: str | torch.device = "cuda",
    n_shots: int = 5,
    n_samples: int = -1,
    max_new_tokens: int = 1,
    temperature: float = 0.0,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a model on the MMLU benchmark.

    Args:
        model: Language model with generation capability.
        tokenizer: Matching tokenizer for the model.
        subjects: List of subjects to evaluate. Use None for all subjects.
        device: Device for inference.
        n_shots: Number of few-shot examples per subject.
        n_samples: Number of test samples per subject. Use -1 for full test sets.
        max_new_tokens: Maximum tokens to generate per question.
        temperature: Sampling temperature. Use 0.0 for greedy decoding.
        verbose: Whether to print progress.

    Returns:
        Aggregated results dictionary with per-subject, per-domain, and overall accuracy.
    """
    if subjects is None:
        subjects = ALL_SUBJECTS

    results: Dict[str, Tuple[int, int]] = {}

    for i, subject in enumerate(subjects, 1):
        if verbose:
            print(f"[{i}/{len(subjects)}] Evaluating {subject}...")

        try:
            correct, total = evaluate_subject(
                model=model,
                tokenizer=tokenizer,
                subject=subject,
                device=device,
                n_shots=n_shots,
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                verbose=verbose,
            )
            results[subject] = (correct, total)

            if verbose:
                acc = correct / total if total > 0 else 0.0
                print(f"  {subject}: {acc:.4f} ({correct}/{total})")

        except Exception as exc:
            if verbose:
                print(f"  [Error] {subject}: {exc}")
            continue

    return aggregate_results(results)


def print_mmlu_results(results: Dict[str, Any]) -> None:
    """
    Print MMLU evaluation results in a formatted table.

    Args:
        results: Aggregated results from evaluate_mmlu().
    """
    print("\n" + "=" * 60)
    print("MMLU Evaluation Results")
    print("=" * 60)

    # Overall
    print(f"\nOverall Accuracy: {results['overall']:.4f}")
    print(f"Total: {results['total_correct']}/{results['total_count']}")

    # Per-domain
    print("\nPer-Domain Accuracy:")
    print("-" * 40)
    for domain, acc in sorted(results["domains"].items()):
        print(f"  {domain:25s}: {acc:.4f}")

    # Per-subject (top/bottom 5)
    subjects = sorted(results["subjects"].items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Subjects:")
    print("-" * 40)
    for subject, acc in subjects[:5]:
        print(f"  {subject:35s}: {acc:.4f}")

    print("\nBottom 5 Subjects:")
    print("-" * 40)
    for subject, acc in subjects[-5:]:
        print(f"  {subject:35s}: {acc:.4f}")

    print("\n" + "=" * 60)


def parse_mmlu_subjects(subjects_str: Optional[str]) -> Optional[List[str]]:
    """
    Parse a comma-separated string of MMLU subjects.

    Supports special values:
    - "all": Returns all subjects
    - "stem", "humanities", "social_sciences", "other": Returns subjects in that domain

    Args:
        subjects_str: Comma-separated subject names or special value.

    Returns:
        List of subject names, or None for all subjects.
    """
    if subjects_str is None:
        return None

    subjects_str = subjects_str.strip().lower()

    if subjects_str == "all":
        return None  # None means all subjects

    domain_map = {
        "stem": "STEM",
        "humanities": "Humanities",
        "social_sciences": "Social Sciences",
        "social_science": "Social Sciences",
        "other": "Other",
    }

    if subjects_str in domain_map:
        return MMLU_SUBJECTS[domain_map[subjects_str]]

    # Parse comma-separated list
    subjects = []
    for s in subjects_str.split(","):
        s = s.strip()
        subject = normalize_subject_name(s)
        if subject in ALL_SUBJECTS:
            subjects.append(subject)
        else:
            # Try to match partial names
            matches = [x for x in ALL_SUBJECTS if subject in x]
            if matches:
                subjects.extend(matches)
            else:
                raise ValueError(
                    f"Unknown subject: '{s}'. "
                    f"Available subjects: {ALL_SUBJECTS[:10]}..."
                )

    return list(set(subjects))  # Remove duplicates
