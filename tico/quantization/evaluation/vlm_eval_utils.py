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

import importlib
import json
import os
import random
import re
import string
import tempfile

from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import torch
from datasets import Dataset, IterableDataset, load_dataset


def normalize_answer(s: str) -> str:
    """
    Normalize an answer string for more stable exact-match evaluation.

    The normalization intentionally removes superficial formatting differences
    that should not count as semantic mismatches.

    Applied steps:
    - lowercase conversion
    - replacement of some separators with spaces
    - punctuation removal
    - article removal ("a", "an", "the")
    - whitespace collapsing

    Args:
        s: Raw answer string.

    Returns:
        A normalized answer string.
    """
    s = s.lower().strip()

    # Treat some punctuation as word separators
    s = s.replace("-", " ").replace("/", " ")

    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)

    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)

    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match(pred: str, golds: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Check whether a prediction matches any gold answer after normalization.

    Args:
        pred: Model prediction.
        golds: List of reference answers.

    Returns:
        A tuple of:
        - whether an exact match was found
        - the matched gold answer, or ``None`` if no match was found
    """
    pred_norm = normalize_answer(pred)
    for gold in golds:
        if pred_norm == normalize_answer(gold):
            return True, gold
    return False, None


def _extract_golds(answers: Any) -> List[str]:
    """
    Convert dataset-specific answer fields into a list of strings.

    Supported input patterns include:
    - None
    - a dictionary with an "answer" field
    - a list of dictionaries each containing "answer"
    - a list of plain values
    - a single scalar value

    Args:
        answers: Raw answer field from a dataset example.

    Returns:
        A list of answer strings.
    """
    if answers is None:
        return []

    if isinstance(answers, dict) and "answer" in answers:
        return [str(a) for a in answers["answer"]]

    if isinstance(answers, list):
        if answers and isinstance(answers[0], dict) and "answer" in answers[0]:
            return [str(a["answer"]) for a in answers]
        return [str(a) for a in answers]

    return [str(answers)]


# ============================================================
# Dataset adapters
# - Different VQA datasets expose answers in different formats
# - These adapters convert raw samples into a unified format:
#   { image, question, golds }
# ============================================================
def get_item_vqav2(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt a VQAv2-style sample to a common evaluation format.

    The returned schema is:

    `{"image": image, "question": question, "golds": gold_answers}`

    Args:
        ex: Raw dataset example.

    Returns:
        A normalized evaluation item.
    """
    return {
        "image": ex["image"],
        "question": ex.get("question", ""),
        "golds": _extract_golds(ex.get("answers")),
    }


def get_item_textvqa(ex: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt a TextVQA-style sample to a common evaluation format.

    TextVQA is often more sensitive to OCR degradation than generic VQA tasks,
    but the unified output schema is the same as for other supported datasets.

    Args:
        ex: Raw dataset example.

    Returns:
        A normalized evaluation item.
    """
    return {
        "image": ex["image"],
        "question": ex.get("question", ""),
        "golds": _extract_golds(ex.get("answers")),
    }


def get_item_coco(ex: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt a COCO Captioning-style sample to a common evaluation format.

    COCO Captioning differs from VQA datasets:
    - There is no question; the task is to describe the image.
    - Each image has multiple reference captions (typically 5).

    The returned schema is:

    `{"image": image, "question": question, "golds": gold_answers}`

    Args:
        ex: Raw dataset example.

    Returns:
        A normalized evaluation item.
    """
    return {
        "image": ex["image"],
        "question": ex["question"],
        "id": ex["id"],
        "image_id": ex["question_id"],
        "file_name": ex["file_name"],
        "golds": ex["answer"],
    }


def get_item_llava_bench_in_the_wild(ex: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": ex["image"],
        "question": ex["question"],
        "id": ex["question_id"],
        "image_id": ex["question_id"],  # unique evaluation key
        "file_name": ex["image_id"],  # original image filename
        "golds": [ex["gpt_answer"]],
    }


def get_item_wikitext2(ex: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt a Wikitext2 sample to a common format for text-only calibration.

    The returned schema is:

    `{"text": text}`

    Args:
        ex: Raw dataset example.

    Returns:
        A normalized text item.
    """
    return {"text": ex.get("text", "")}


def get_item_alpaca(ex: dict[str, Any]) -> dict[str, Any]:
    """
    Adapt an Alpaca-style sample to a common format for text-only calibration.

    Alpaca format has "instruction", "input", and "output" fields.
    The instruction and input are combined for the text.

    The returned schema is:

    `{"text": text}`

    Args:
        ex: Raw dataset example.

    Returns:
        A normalized text item.
    """
    instruction = ex.get("instruction", "")
    input_text = ex.get("input", "")
    if input_text:
        text = f"{instruction}\n{input_text}"
    else:
        text = instruction
    return {"text": text}


DATASETS: dict[str, dict[str, Any]] = {
    "vqav2": {
        "default_split": "validation",
        "adapter": get_item_vqav2,
        "candidates": ["HuggingFaceM4/VQAv2", "vqav2", "lmms-lab/VQAv2"],
        "is_text_only": False,
    },
    "textvqa": {
        "default_split": "validation",
        "adapter": get_item_textvqa,
        "candidates": ["textvqa", "HuggingFaceM4/TextVQA", "lmms-lab/textvqa"],
        "is_text_only": False,
    },
    "coco": {
        "default_split": "val",
        "adapter": get_item_coco,
        "candidates": [
            "lmms-lab/COCO-Caption2017",
        ],
        "is_text_only": False,
    },
    "llava_bench": {
        "default_split": "train",
        "adapter": get_item_llava_bench_in_the_wild,
        "candidates": [
            "lmms-lab/llava-bench-in-the-wild",
        ],
    },
    "wikitext2": {
        "default_split": "test",
        "adapter": get_item_wikitext2,
        "candidates": ["Salesforce/wikitext"],
        "config": "wikitext-2-raw-v1",
        "is_text_only": True,
    },
    "alpaca": {
        "default_split": "train",
        "adapter": get_item_alpaca,
        "candidates": ["tatsu-lab/alpaca"],
        "is_text_only": True,
    },
}


def build_messages(question: str) -> List[Dict[str, Any]]:
    """
    Build a chat-style multimodal message payload for a VLM prompt.

    The prompt includes:
    - one image placeholder
    - one text instruction containing the question
    - a short instruction asking for only the final answer

    Args:
        question: User question associated with the image.

    Returns:
        A list of chat-format messages compatible with processor chat templates.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        f"{question}\n"
                        "Return ONLY the final answer with no extra words."
                    ),
                },
            ],
        }
    ]


def build_prompt(processor, question: str) -> str:
    """
    Render a text prompt from a multimodal chat template.

    Args:
        processor: Hugging Face processor with `apply_chat_template` support.
        question: User question associated with the image.

    Returns:
        A rendered prompt string containing image placeholder tokens.
    """
    messages = build_messages(question)
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_vlm_inputs(
    processor,
    image,
    question: str,
    return_tensors: str = "pt",
    max_seq_len: Optional[int] = None,
):
    """
    Build processor inputs for a single image-question example.

    Args:
        processor: Hugging Face multimodal processor.
        image: Input image object accepted by the processor.
        question: User question associated with the image.
        return_tensors: Tensor format requested from the processor.
        max_seq_len: Optional maximum text sequence length. If provided,
                     text inputs are truncated to this length.

    Returns:
        A processor output object containing model-ready multimodal inputs.
    """
    prompt = build_prompt(processor, question)
    processor_kwargs: Dict[str, Any] = {
        "text": prompt,
        "images": image,
        "return_tensors": return_tensors,
    }
    if max_seq_len is not None and max_seq_len > 0:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = max_seq_len

    return processor(**processor_kwargs)


def move_inputs_to_device(inputs, device: str | torch.device):
    """
    Move tensor-valued processor inputs to the target device in-place.

    Non-tensor entries are preserved unchanged.

    Args:
        inputs: Mapping-like processor outputs.
        device: Target device.

    Returns:
        The same input container with tensor values moved to the target device.
    """
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return inputs


@torch.no_grad()
def generate_answer(
    model,
    processor,
    image,
    question: str,
    device: str | torch.device,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    max_seq_len: Optional[int] = None,
) -> str:
    """
    Generate an answer for a single image-question example.

    Args:
        model: Vision-language generation model.
        processor: Matching processor for the model.
        image: Input image.
        question: Text question for the image.
        device: Device on which generation should run.
        max_new_tokens: Maximum number of generated tokens.
        temperature: Sampling temperature. Greedy decoding is used when this
                     value is less than or equal to zero.
        max_seq_len: Optional maximum text sequence length for processor
                     tokenization.

    Returns:
        The decoded model answer string.
    """
    inputs = build_vlm_inputs(
        processor=processor,
        image=image,
        question=question,
        return_tensors="pt",
        # length of (inputs + max_new_tokens) should not exceed model's max_seq_len
        # because quantized model has precomputed static causal mask and RoPE for max_seq_len
        max_seq_len=max_seq_len - max_new_tokens,  # type: ignore[operator]
    )
    inputs = move_inputs_to_device(inputs, device)

    # Generate kwargs
    do_sample = temperature > 0.0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    out_ids = model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, input_len:]

    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_image_only_answer(
    model,
    processor,
    image,
    device: str | torch.device,
    question: str | None = None,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    max_seq_len: int | None = None,
) -> str:
    """
    Generate an answer from the image only.

    Args:
        model: Vision-language generation model.
        processor: Matching processor for the model.
        image: Input image.
        question: Optional text question.
        device: Device on which generation should run.
        max_new_tokens: Maximum number of generated tokens.
        temperature: Sampling temperature. Greedy decoding is used when this
                     value is less than or equal to zero.
        max_seq_len: Optional maximum text sequence length for processor
                     tokenization.

    Returns:
        The decoded model answer string.
    """
    content: list = [{"type": "image"}]

    if question is not None:
        content.append(
            {
                "type": "text",
                "text": question,
            }
        )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    processor_kwargs: dict[str, Any] = {
        "text": prompt,
        "images": image,
        "return_tensors": "pt",
    }
    if max_seq_len is not None and max_seq_len > 0:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = max_seq_len

    inputs = processor(**processor_kwargs)
    inputs = move_inputs_to_device(inputs, device)

    do_sample = temperature > 0.0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    out_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = out_ids[0, input_len:]

    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


class CocoResult(TypedDict):
    image_id: str
    caption: str


class CocoAnnotation(TypedDict):
    id: int
    image_id: str
    caption: str


class CocoImage(TypedDict):
    id: str
    file_name: str


# ============================================================
# Metric computation functions
# - Each function computes a specific captioning metric
# - These can be used independently or via get_coco_scores_on_dataset
# ============================================================
def _get_required_coco_eval_modules(metrics: Iterable[str]) -> list[str]:
    """Return optional Python modules needed for requested COCO metrics."""
    modules = ["pycocotools.coco"]
    metric_modules = {
        "CIDEr": "pycocoevalcap.cider.cider",
        "METEOR": "pycocoevalcap.meteor.meteor",
        "ROUGE_L": "pycocoevalcap.rouge.rouge",
    }

    for metric in metrics:
        if metric.startswith("Bleu_"):
            modules.append("pycocoevalcap.bleu.bleu")
        elif metric in metric_modules:
            modules.append(metric_modules[metric])

    return list(dict.fromkeys(modules))


def _require_coco_eval_dependencies(metrics: Iterable[str]) -> None:
    """
    Validate optional COCO evaluation dependencies before running inference.

    COCO scoring is executed after all samples have been generated. Importing
    the optional evaluation modules up front lets a missing dependency fail fast
    before expensive benchmark samples are processed.
    """
    missing: list[str] = []
    first_error: ImportError | None = None

    for module_name in _get_required_coco_eval_modules(metrics):
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            if first_error is None:
                first_error = exc
            missing.append(f"{module_name}: {exc}")

    if missing:
        missing_lines = "\n".join(f"- {item}" for item in missing)
        raise RuntimeError(
            "COCO evaluation dependencies are missing for the requested metrics. "
            "Install the optional COCO evaluation packages before running the "
            "benchmark, for example: "
            "`pip install pycocotools pycocoevalcap`.\n"
            f"Missing imports:\n{missing_lines}"
        ) from first_error


def compute_bleu_scores(
    ground_truths: dict[str, list[str]],
    predictions: dict[str, list[str]],
    bleu_metrics: list[str] = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"],
) -> dict[str, float]:
    """
    Compute BLEU scores (Bleu_1 through Bleu_4) for image captioning.

    Args:
        ground_truths: Dictionary mapping image_id to list of reference captions.
        predictions: Dictionary mapping image_id to list of predicted captions.
        bleu_metrics: List of BLEU metrics to compute. Supported: "Bleu_1", "Bleu_2",
                      "Bleu_3", "Bleu_4".

    Returns:
        Dictionary mapping metric names to scores.
    """
    from pycocoevalcap.bleu.bleu import Bleu

    bleu_scorer = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(ground_truths, predictions)
    bleu_map = {
        "Bleu_1": 0,
        "Bleu_2": 1,
        "Bleu_3": 2,
        "Bleu_4": 3,
    }
    result: dict[str, float] = {}
    for m in bleu_metrics:
        idx = bleu_map[m]
        result[m] = float(bleu_scores[idx])
    return result


def get_coco_scores_on_dataset(
    model,
    processor,
    dataset_name: str,
    ds: Iterable[dict[str, Any]],
    device: str | torch.device,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    max_seq_len: int | None = None,
    verbose: bool = True,
    log_first_n: int = 5,
    log_every_n: int = 50,
    metrics: list[str] = [
        "CIDEr",
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
    ],
) -> dict[str, float]:
    """
    Compute CIDEr, BLEU, and other captioning metrics on a dataset iterator.

    This function uses the pycocoevalcap package to compute standard captioning
    metrics including CIDEr, BLEU-1 through BLEU-4, METEOR, ROUGE-L.

    Args:
        model: Vision-language generation model.
        processor: Matching processor for the model.
        ds: Iterable dataset yielding raw examples.
        device: Device used for inference.
        max_new_tokens: Maximum number of generated tokens.
        temperature: Sampling temperature.
        max_seq_len: Optional maximum text sequence length.
        verbose: Whether to print sample predictions and progress logs.
        log_first_n: Number of early examples to print.
        log_every_n: Logging interval after the initial examples.
        metrics: List of metrics to compute. Supported values: "CIDEr", "Bleu_1",
                 "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L".
                 Defaults to CIDEr and BLEU metrics.

    Returns:
        A dictionary mapping metric names to scores (e.g., "CIDEr", "Bleu_4").
    """
    # Validate metrics
    supported_metrics = {
        "CIDEr",
        "Bleu_1",
        "Bleu_2",
        "Bleu_3",
        "Bleu_4",
        "METEOR",
        "ROUGE_L",
    }
    for m in metrics:
        if m not in supported_metrics:
            raise ValueError(
                f"Unsupported metric '{m}'. Supported metrics: {supported_metrics}"
            )

    _require_coco_eval_dependencies(metrics)

    # Collect predictions and ground truth
    results: list[CocoResult] = []
    images: list[CocoImage] = []
    annotations: list[CocoAnnotation] = []

    if "coco" in dataset_name.lower():
        get_item = get_item_coco
    elif "llava_bench" in dataset_name.lower():
        get_item = get_item_llava_bench_in_the_wild
    else:
        raise ValueError(f"Invalid dataset_name={dataset_name}")

    total_count = 0
    skipped_count = 0
    annotation_id = 0
    for i, ex in enumerate(ds, 1):
        sample: dict[str, Any] = get_item(ex)

        image: Any = sample["image"]
        question: str = sample["question"]
        sample_id: int = sample["id"]
        image_id: str = sample["image_id"]
        file_name: str = sample["file_name"]
        gold_answers: list[str] = sample["golds"]

        total_count += 1
        try:
            pred = generate_answer(
                model=model,
                processor=processor,
                image=image,
                question=question,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                max_seq_len=max_seq_len,
            )
        except (ValueError, RuntimeError) as error:
            message = str(error).lower()
            if not any(
                marker in message
                for marker in (
                    "too long",
                    "max_position_embeddings",
                    "maximum context length",
                    "sequence length",
                    "truncation",
                )
            ):
                raise

            print("[WARNING] The prompt was too long. Skipping.")
            print(f"{type(error).__name__}: {error}")
            skipped_count += 1
            continue

        # Store result
        result: CocoResult = {"image_id": image_id, "caption": pred}
        results.append(result)

        # Store ground truth
        img: CocoImage = {"id": image_id, "file_name": file_name}
        images.append(img)

        for answer in gold_answers:
            annotation_id += 1
            annotation: CocoAnnotation = {
                "id": annotation_id,
                "image_id": image_id,
                "caption": answer,
            }
            annotations.append(annotation)

        should_log = verbose and (
            i <= log_first_n or (log_every_n > 0 and i % log_every_n == 0)
        )
        if should_log:
            print("id:", sample_id)
            print("image_id:", image_id)
            print("Q:", question)
            print("pred:", repr(pred))
            print("pred_norm:", repr(normalize_answer(pred)))
            print("golds[:10]:", [repr(x) for x in gold_answers[:10]])
            print("-" * 60)

    if not results:
        raise RuntimeError(
            "No evaluation results were collected. "
            "All samples may have been skipped due to prompt length errors."
        )
    assert images
    assert annotations

    # Create temporary files for COCO evaluation
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as annotations_file:
        json.dump(
            {
                "images": images,
                "annotations": annotations,
            },
            annotations_file,
        )
        annotations_file.flush()
        annotation_path = annotations_file.name

    # Run COCO evaluation
    try:
        from pycocotools.coco import COCO

        coco = COCO(annotation_path)

        # Convert COCO objects to dictionaries expected by Cider and Bleu
        # Format: {image_id: [caption1, caption2, ...]}
        ground_truths: dict[str, list[str]] = {}
        ann_ids = coco.getAnnIds()
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            img_id = str(a["image_id"])
            if img_id not in ground_truths:
                ground_truths[img_id] = []
            ground_truths[img_id].append(a["caption"])

        # Format: {image_id: [caption]}
        res: dict[str, list[str]] = {}
        for r in results:
            img_id = str(r["image_id"])
            res[img_id] = [r["caption"]]

        all_scores: dict[str, float] = {}
        all_scores["total_count"] = total_count
        all_scores["skipped_count"] = skipped_count

        # Compute CIDEr if needed
        if "CIDEr" in metrics:
            from pycocoevalcap.cider.cider import Cider

            cider_scorer = Cider()
            cider_score, _ = cider_scorer.compute_score(ground_truths, res)
            all_scores["CIDEr"] = float(cider_score)
            if verbose:
                print(f"CIDEr: {cider_score:.4f}")

        # Compute BLEU scores if needed
        bleu_metrics = [m for m in metrics if m.startswith("Bleu_")]
        if bleu_metrics:
            bleu_scores = compute_bleu_scores(
                ground_truths, res, bleu_metrics=bleu_metrics
            )
            all_scores.update(bleu_scores)
            if verbose:
                for m, score in bleu_scores.items():
                    print(f"{m}: {score:.4f}")

        # Compute METEOR if needed
        if "METEOR" in metrics:
            from pycocoevalcap.meteor.meteor import Meteor

            meteor_scorer = Meteor()
            meteor_score, _ = meteor_scorer.compute_score(ground_truths, res)
            all_scores["METEOR"] = float(meteor_score)
            if verbose:
                print(f"METEOR: {meteor_score:.4f}")

        # Compute ROUGE_L if needed
        if "ROUGE_L" in metrics:
            from pycocoevalcap.rouge.rouge import Rouge

            rouge_scorer = Rouge()
            rouge_score, _ = rouge_scorer.compute_score(ground_truths, res)
            all_scores["ROUGE_L"] = float(rouge_score)
            if verbose:
                print(f"ROUGE_L: {rouge_score:.4f}")

        return all_scores
    finally:
        os.unlink(annotation_path)


def get_accuracy_on_dataset(
    model,
    processor,
    ds: Iterable[Dict[str, Any]],
    adapter,
    device: str | torch.device,
    max_new_tokens: int = 16,
    temperature: float = 0.0,
    max_seq_len: Optional[int] = None,
    verbose: bool = True,
    log_first_n: int = 5,
    log_every_n: int = 50,
) -> Tuple[int, int]:
    """
    Compute exact-match accuracy on a dataset iterator.

    Args:
        model: Vision-language generation model.
        processor: Matching processor for the model.
        ds: Iterable dataset yielding raw examples.
        adapter: Function that converts raw examples into the common schema
            ``{"image", "question", "golds"}``.
        device: Device used for inference.
        max_new_tokens: Maximum number of generated tokens.
        temperature: Sampling temperature.
        max_seq_len: Optional maximum text sequence length.
        verbose: Whether to print sample predictions and progress logs.
        log_first_n: Number of early examples to print.
        log_every_n: Logging interval after the initial examples.

    Returns:
        A tuple of:
        - number of exact-match hits
        - total number of evaluated examples
    """
    em_cnt = 0
    total = 0

    for i, ex in enumerate(ds, 1):
        item = adapter(ex)

        pred = generate_answer(
            model=model,
            processor=processor,
            image=item["image"],
            question=item["question"],
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_seq_len=max_seq_len,
        )

        ok, hit = exact_match(pred, item["golds"])
        em_cnt += int(ok)
        total += 1

        should_log = verbose and (
            i <= log_first_n or (log_every_n > 0 and i % log_every_n == 0)
        )
        if should_log:
            print("Q:", item["question"])
            print("pred:", repr(pred))
            print("pred_norm:", repr(normalize_answer(pred)))
            print("gold0:", repr(item["golds"][0] if item["golds"] else ""))
            print("golds[:10]:", [repr(x) for x in item["golds"][:10]])
            print("hit:", repr(hit))
            print("ok:", ok)
            print("-" * 60)

    return em_cnt, total


def get_dataset(
    dataset: str,
    n: int = 50,
    split: Optional[str] = None,
    streaming: bool = True,
):
    """
    Load a supported evaluation dataset with simple name fallback logic.

    The function tries multiple known dataset identifiers because mirrors or
    dataset aliases may differ across environments.

    Args:
        dataset: Dataset key defined in ``DATASETS``.
        n: Number of examples to take. Use a negative value to keep the full
            iterable without truncation.
        split: Optional dataset split. If omitted, the dataset default split is
            used.
        streaming: Whether to request streaming mode from ``load_dataset``.

    Returns:
        A tuple of:
        - dataset iterable
        - adapter function for the selected dataset

    Raises:
        KeyError: If the dataset key is unsupported.
        RuntimeError: If none of the candidate dataset names can be loaded.
    """
    if dataset not in DATASETS:
        raise KeyError(
            f"Unsupported dataset '{dataset}'. Available datasets: {list(DATASETS.keys())}"
        )

    meta: dict[str, Any] = DATASETS[dataset]
    adapter = meta["adapter"]
    split = split or meta["default_split"]  # type: ignore[assignment]
    candidates = meta["candidates"]
    config = meta.get("config")  # Optional config for datasets like wikitext
    assert isinstance(candidates, list)

    ds = None
    last_err: Optional[Exception] = None

    for name in candidates:
        try:
            if config:
                ds = load_dataset(
                    path=name, name=config, split=split, streaming=streaming
                )
            else:
                ds = load_dataset(path=name, split=split, streaming=streaming)
            if n >= 0:
                ds = ds.take(n)

            size_str = str(n) if n >= 0 else "all"
            stream_str = "streaming" if streaming else "non-streaming"
            config_str = f"config={config}, " if config else ""
            print(
                f"[info] Loaded dataset: {name} ({config_str}{split}, {stream_str}), size={size_str}"
            )
            break
        except Exception as exc:
            last_err = exc

    if ds is None:
        raise RuntimeError(
            f"Failed to load dataset='{dataset}', split='{split}', "
            f"candidates={candidates}. Last error: {last_err}"
        )

    return ds, adapter


def evaluate_ppl(
    model,
    tokenizer,
    ds: Dataset | IterableDataset,
    device: str | torch.device,
    stride: int = 512,
    max_seq_len: Optional[int] = None,
    show_progress: bool = True,
) -> float:
    """
    Evaluate perplexity on a text dataset.

    This function computes perplexity using a strided sliding-window approach.
    It expects a dataset that yields examples with a "text" field (e.g., wikitext2).

    Args:
        model: Language model to evaluate.
        tokenizer: Tokenizer for encoding text.
        ds: Iterable dataset yielding examples with a "text" field.
        device: Device used for evaluation.
        stride: Sliding window stride for perplexity calculation.
        max_seq_len: Maximum sequence length. Defaults to model's max_position_embeddings.
        show_progress: Whether to show progress bar.

    Returns:
        Perplexity score.
    """
    from tico.quantization.wrapq.utils.metrics import perplexity

    # Concatenate all text from the dataset
    full_text = "\n\n".join(ds["text"])

    # Encode the full text
    encodings = tokenizer(full_text, return_tensors="pt")

    # Compute perplexity
    ppl = perplexity(
        model=model,
        encodings=encodings,
        device=device,
        max_length=max_seq_len,
        stride=stride,
        show_progress=show_progress,
    )
    return ppl


def get_calib_inputs(
    dataset: str,
    processor,
    n_samples: int = 28,
    split: str = "testdev",
    max_seq_len: Optional[int] = None,
):
    """
    Build calibration inputs by preprocessing image-question pairs.

    This helper uses the same prompt and processor-input construction logic as
    evaluation so that calibration and inference stay aligned.

    Args:
        dataset: Dataset key defined in ``DATASETS``.
        processor: Hugging Face multimodal processor.
        n_samples: Number of calibration examples to prepare.
        split: Dataset split to load.
        max_seq_len: Optional maximum text sequence length.

    Returns:
        A list of processor output objects, one per example.
    """
    ds, adapter = get_dataset(dataset=dataset, n=n_samples, split=split)

    calib_inputs = []
    for ex in ds:
        item = adapter(ex)
        inputs = build_vlm_inputs(
            processor=processor,
            image=item["image"],
            question=item["question"],
            return_tensors="pt",
            max_seq_len=max_seq_len,
        )
        calib_inputs.append(inputs)

    return calib_inputs


def build_text_only_inputs(
    processor,
    text: str,
    return_tensors: str = "pt",
    max_seq_len: Optional[int] = None,
):
    """
    Build processor inputs for text-only data (no image).

    Args:
        processor: Hugging Face processor.
        text: Input text.
        return_tensors: Tensor format requested from the processor.
        max_seq_len: Optional maximum text sequence length. If provided,
                     text inputs are truncated to this length.

    Returns:
        A processor output object containing model-ready text inputs.
    """
    processor_kwargs: Dict[str, Any] = {
        "text": text,
        "return_tensors": return_tensors,
    }
    if max_seq_len is not None and max_seq_len > 0:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = max_seq_len

    return processor(**processor_kwargs)


def _build_text_calib_inputs(
    processor,
    text: str,
    n_samples: int,
    max_seq_len: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Build calibration inputs from text by sampling random fixed-length sequences.

    Tokenize all text, then sample random fixed-length sequences.

    Args:
        processor: Hugging Face processor with tokenizer.
        text: Full text to sample from.
        n_samples: Number of calibration samples to generate.
        max_seq_len: Sequence length for each sample.
        seed: Random seed for reproducible sampling.

    Returns:
        A list of processor output objects, each containing input_ids.
    """
    # Tokenize the full text
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    input_ids = tokenizer(text, return_tensors="pt").input_ids

    calib_inputs = []
    rng = random.Random(seed)

    for _ in range(n_samples):
        # Sample a random starting position
        max_start = input_ids.shape[1] - max_seq_len - 1
        if max_start <= 0:
            # If text is too short, use what we have
            start = 0
            end = input_ids.shape[1]
        else:
            start = rng.randint(0, max_start)
            end = start + max_seq_len

        sample_ids = input_ids[:, start:end]

        # Build inputs dict similar to processor output
        inputs = {"input_ids": sample_ids}
        calib_inputs.append(inputs)

    return calib_inputs


def get_mixed_calib_inputs(
    processor,
    dataset_config: Dict[str, Dict[str, Any]],
    max_seq_len: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Build calibration inputs from multiple datasets

    This function loads samples from multiple datasets and combines them into
    a single calibration set. It handles both image-text datasets (e.g. VQAv2, COCO)
    and text-only datasets (e.g. Wikitext2, Alpaca).

    For text-only datasets, it concatenates all text and samples random
    fixed-length sequences.

    For image-text datasets, it takes the first n_samples directly.

    Args:
        processor: Hugging Face processor.
        max_seq_len: maximum text sequence length.
        dataset_config: Dictionary mapping dataset names (with optional split) to
            number of samples. Format: {"dataset": {"n_samples": n_samples}} or
                                       {"dataset": {"split":"split","n_samples": n_samples}}.
            Example: {
                      "wikitext2": {"n_samples": 128},
                      "alpaca": {"split": "train", "n_samples": 32}
                      }
        seed: Random seed for reproducible sampling (used only for text-only
            datasets).

    Returns:
        A list of processor output objects from all datasets.
    """
    calib_inputs = []

    for dataset, config in dataset_config.items():
        n_samples = config.get("n_samples", None)
        split = config.get("split", None)
        if n_samples is None or n_samples <= 0:
            continue

        if dataset not in DATASETS:
            print(f"[warn] Unknown dataset '{dataset}', skipping")
            continue

        is_text_only = DATASETS[dataset].get("is_text_only", False)

        if is_text_only:
            # TODO: text only inputs should be changed with chat template

            # Loading whole dataset
            ds, adapter = get_dataset(dataset=dataset, n=-1, split=split)

            # For text-only datasets: use adapter to extract text, then sample random sequences
            if adapter is None:
                print(f"[warn] No adapter for dataset '{dataset}', skipping")
                continue

            all_texts = []
            for ex in ds:
                item = adapter(ex)
                text = item.get("text", "")
                if text.strip():
                    all_texts.append(text)

            if not all_texts:
                print(f"[warn] No text found in dataset '{dataset}', skipping")
                continue

            # Concatenate all text
            full_text = "\n\n".join(all_texts)
            print(f"[info] Tokenizing {len(full_text)} chars from '{dataset}'")

            # Sample random fixed-length sequences
            text_inputs = _build_text_calib_inputs(
                processor=processor,
                text=full_text,
                n_samples=n_samples,
                max_seq_len=max_seq_len,
                seed=seed,
            )
            calib_inputs.extend(text_inputs)
        else:
            # Loading whole dataset
            ds, adapter = get_dataset(dataset=dataset, n=n_samples, split=split)

            # For image-text datasets: take first n_samples directly
            if adapter is None:
                print(f"[warn] No adapter for dataset '{dataset}', skipping")
                continue

            for ex in ds:
                item = adapter(ex)
                inputs = build_vlm_inputs(
                    processor=processor,
                    image=item["image"],
                    question=item["question"],
                    return_tensors="pt",
                    max_seq_len=max_seq_len,
                )
                calib_inputs.append(inputs)

    if not calib_inputs:
        raise ValueError(
            "No calibration inputs were loaded. "
            "Please check --nsamples_for_qcalibration."
        )

    print(f"[info] Total calibration samples: {len(calib_inputs)}")
    return calib_inputs
