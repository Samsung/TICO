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

import json
import os
import re
import string
import tempfile

from collections.abc import Callable
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

import torch
from datasets import load_dataset


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
        "question": ex.get("question", ""),
        "golds": _extract_golds(ex.get("answer")),
    }


DATASETS = {
    "vqav2": {
        "default_split": "validation",
        "adapter": get_item_vqav2,
        "candidates": ["HuggingFaceM4/VQAv2", "vqav2", "lmms-lab/VQAv2"],
    },
    "textvqa": {
        "default_split": "validation",
        "adapter": get_item_textvqa,
        "candidates": ["textvqa", "HuggingFaceM4/TextVQA", "lmms-lab/textvqa"],
    },
    "coco": {
        "default_split": "val",
        "adapter": get_item_coco,
        "candidates": [
            "lmms-lab/COCO-Caption2017",
        ],
    },
    "wikitext2": {
        "default_split": "test",
        "adapter": None,  # Text-only dataset, no adapter needed
        "candidates": ["wikitext"],
        "config": "wikitext-2-raw-v1",
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
        max_seq_len=max_seq_len,
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


def get_coco_scores_on_dataset(
    model,
    processor,
    ds: Iterable[dict[str, Any]],
    device: str | torch.device,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    max_seq_len: int | None = None,
    verbose: bool = True,
    log_first_n: int = 5,
    log_every_n: int = 50,
) -> dict[str, float]:
    """
    Compute CIDEr, BLEU, and other captioning metrics on a dataset iterator.

    This function uses the pycocoevalcap package to compute standard captioning
    metrics including CIDEr, BLEU-1 through BLEU-4, METEOR, ROUGE-L, and SPICE.

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

    Returns:
        A dictionary mapping metric names to scores (e.g., "CIDEr", "Bleu_4").
    """
    # Collect predictions and ground truth
    results: list[CocoResult] = []
    images: list[CocoImage] = []
    annotations: list[CocoAnnotation] = []

    for i, ex in enumerate(ds, 1):
        image: Any = ex["image"]
        question: str = ex["question"]
        id: int = ex["id"]
        image_id: str = ex["question_id"]
        file_name: str = ex["file_name"]
        gold_answers: list[str] = ex["answer"]

        # Generate caption
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

        # Store result
        result: CocoResult = {"image_id": image_id, "caption": pred}
        results.append(result)

        # Store ground truth
        img: CocoImage = {"id": image_id, "file_name": file_name}
        images.append(img)

        for answer in gold_answers:
            annotation: CocoAnnotation = {
                "id": id,
                "image_id": image_id,
                "caption": answer,
            }
            annotations.append(annotation)

        should_log = verbose and (
            i <= log_first_n or (log_every_n > 0 and i % log_every_n == 0)
        )
        if should_log:
            print("id:", id)
            print("image_id:", image_id)
            print("Q:", question)
            print("pred:", repr(pred))
            print("pred_norm:", repr(normalize_answer(pred)))
            print("golds[:10]:", [repr(x) for x in gold_answers[:10]])
            print("-" * 60)

    assert results
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
        from pycocoevalcap.eval import COCOEvalCap
        from pycocotools.coco import COCO

        coco = COCO(annotation_path)
        coco_result = coco.loadRes(results)
        coco_eval = COCOEvalCap(coco, coco_result)
        coco_eval.params["id"] = coco_result.getImgIds()
        coco_eval.evaluate()

        metrics: dict[str, float] = {}
        for metric, score in coco_eval.eval.items():
            metrics[metric] = float(score)
            if verbose:
                print(f"{metric}: {score:.4f}")

        return metrics
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

    meta = DATASETS[dataset]
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
    ds: Iterable[Dict[str, Any]],
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
