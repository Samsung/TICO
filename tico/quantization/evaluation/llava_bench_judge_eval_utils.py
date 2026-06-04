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

"""Judge-based evaluation utilities for LLaVA-Bench-in-the-Wild.

LLaVA-Bench-in-the-Wild is an open-ended natural QA benchmark. These utilities
therefore generate answers from the original question without short-answer
constraints and evaluate candidate answers with a text-only judge model.

The default judge is useful for inexpensive regression checks. Scores produced
by a non-official judge are not directly comparable to official LLaVA-Bench
GPT-4-judge scores.

Static multimodal runtimes may require a small context length such as 2048. For
that case, images are resized before processor tokenization so the visual token
count fits the reserved input budget. If a sample still does not fit, the
utility raises an error instead of silently skipping the sample.
"""

from __future__ import annotations

import json
import math
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset


LLAVA_BENCH_DATASET = "lmms-lab/llava-bench-in-the-wild"
DEFAULT_JUDGE_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"


@dataclass
class LlavaBenchJudgeConfig:
    """Configuration for LLaVA-Bench answer generation and judge scoring."""

    dataset: str = LLAVA_BENCH_DATASET
    split: str = "train"
    n_samples: int = 50
    start_index: int = 0
    max_seq_len: int | None = None
    max_new_tokens: int = 512
    temperature: float = 0.0
    image_min_pixels: int | None = None
    image_max_pixels: int | None = None
    resized_height: int | None = None
    resized_width: int | None = None
    visual_token_margin: int = 256
    candidate_label: str = "candidate"
    baseline_label: str = "reference"
    candidate_answers_path: str | None = None
    baseline_answers_path: str | None = None
    answers_out: str | None = None
    reviews_out: str | None = None
    summary_out: str | None = None
    output_dir: str = "./out/llava_bench"
    regenerate: bool = False
    judge_enabled: bool = True
    judge_model_id: str = DEFAULT_JUDGE_MODEL_ID
    judge_device: str = "cuda"
    judge_dtype: str = "float16"
    judge_max_new_tokens: int = 256
    judge_temperature: float = 0.0
    judge_swap_order: bool = True
    trust_remote_code: bool = True
    hf_token: str | None = None
    cache_dir: str | None = None
    quiet: bool = False


@dataclass
class AnswerRecord:
    """One generated answer for a LLaVA-Bench question."""

    question_id: str
    image_id: str
    prompt: str
    text: str
    reference_answer: str
    model_id: str
    context: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Convert the record to a JSON-serializable dictionary."""
        return {
            "question_id": self.question_id,
            "image_id": self.image_id,
            "prompt": self.prompt,
            "text": self.text,
            "reference_answer": self.reference_answer,
            "model_id": self.model_id,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class JudgeResult:
    """Canonical judge scores for one question."""

    question_id: str
    baseline_label: str
    candidate_label: str
    score_baseline: float
    score_candidate: float
    winner: str
    reason: str
    raw_judge_output: str
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        """Convert the result to a JSON-serializable dictionary."""
        return {
            "question_id": self.question_id,
            "baseline_label": self.baseline_label,
            "candidate_label": self.candidate_label,
            "score_baseline": self.score_baseline,
            "score_candidate": self.score_candidate,
            "winner": self.winner,
            "reason": self.reason,
            "raw_judge_output": self.raw_judge_output,
            "metadata": self.metadata,
        }


def torch_dtype_from_name(name: str) -> torch.dtype:
    """Convert a simple dtype name to a torch dtype."""
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def safe_model_id(model_id: str) -> str:
    """Return a filesystem-safe model identifier."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id).strip("_")
    return safe or "model"


def resolve_output_paths(config: LlavaBenchJudgeConfig) -> LlavaBenchJudgeConfig:
    """Fill output paths that were not provided explicitly."""
    base = Path(config.output_dir)
    candidate_key = safe_model_id(config.candidate_label)
    judge_key = safe_model_id(config.judge_model_id)
    if config.answers_out is None:
        config.answers_out = str(base / f"{candidate_key}.answers.jsonl")
    if config.reviews_out is None:
        config.reviews_out = str(base / f"{candidate_key}.{judge_key}.reviews.jsonl")
    if config.summary_out is None:
        config.summary_out = str(base / f"{candidate_key}.{judge_key}.summary.json")
    return config


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write an indented JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def get_first_string(
    ex: dict[str, Any], keys: tuple[str, ...], default: str = ""
) -> str:
    """Return the first non-empty string value from a dataset example."""
    for key in keys:
        value = ex.get(key)
        if value is None:
            continue
        text = str(value)
        if text:
            return text
    return default


def normalize_dataset_example(ex: dict[str, Any]) -> dict[str, Any]:
    """Normalize one LLaVA-Bench example into a stable internal schema."""
    question_id = get_first_string(ex, ("question_id", "id"), str(uuid.uuid4()))
    image_id = get_first_string(ex, ("image_id", "file_name"), question_id)
    question = get_first_string(ex, ("question", "text", "prompt"))
    reference = get_first_string(ex, ("gpt_answer", "answer", "reference_answer"))
    context = get_first_string(
        ex,
        (
            "context",
            "caption",
            "image_caption",
            "image_context",
            "text_context",
        ),
    )
    metadata = {
        key: ex.get(key)
        for key in ("category", "image_id", "question_id")
        if key in ex and key != "image"
    }
    return {
        "image": ex.get("image"),
        "question_id": question_id,
        "image_id": image_id,
        "question": question,
        "reference_answer": reference,
        "context": context,
        "metadata": metadata,
    }


def iter_llava_bench_dataset(config: LlavaBenchJudgeConfig) -> Iterable[dict[str, Any]]:
    """Load and slice the configured LLaVA-Bench dataset split."""
    ds = load_dataset(config.dataset, split=config.split, streaming=True)
    yielded = 0
    for index, example in enumerate(ds):
        if index < config.start_index:
            continue
        if config.n_samples > 0 and yielded >= config.n_samples:
            break
        yielded += 1
        yield example


def _input_max_length(config: LlavaBenchJudgeConfig) -> int | None:
    """Return the input token budget after reserving generation tokens."""
    if config.max_seq_len is None or config.max_seq_len <= 0:
        return None
    max_length = config.max_seq_len - config.max_new_tokens
    if max_length <= 0:
        raise ValueError(
            "LLaVA-Bench max_seq_len must be larger than max_new_tokens: "
            f"max_seq_len={config.max_seq_len}, max_new_tokens={config.max_new_tokens}."
        )
    return max_length


def _effective_image_max_pixels(config: LlavaBenchJudgeConfig) -> int | None:
    """Return the image pixel cap used to keep visual tokens within context."""
    if config.image_max_pixels is not None:
        return config.image_max_pixels
    input_budget = _input_max_length(config)
    if input_budget is None:
        return None
    visual_budget = input_budget - max(0, int(config.visual_token_margin))
    if visual_budget <= 0:
        return 28 * 28
    return visual_budget * 28 * 28


def _coerce_int_attr(value: Any, default: int) -> int:
    """Convert scalar or one-element processor attributes to an integer."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return int(value[0])
    return int(value)


def _processor_patch_size(processor: Any) -> int:
    """Return the processor patch size used for image grid calculation."""
    image_processor = getattr(processor, "image_processor", None)
    return _coerce_int_attr(getattr(image_processor, "patch_size", None), 16)


def _processor_vision_factor(processor: Any) -> int:
    """Return the pixel stride of one merged visual token step."""
    image_processor = getattr(processor, "image_processor", None)
    patch_size = _coerce_int_attr(getattr(image_processor, "patch_size", None), 16)
    merge_size = _coerce_int_attr(getattr(image_processor, "merge_size", None), 2)
    return max(1, patch_size * merge_size)


def _floor_to_factor(value: float, factor: int) -> int:
    """Floor a positive value to a positive multiple of factor."""
    return max(factor, int(math.floor(value / factor)) * factor)


def _ceil_to_factor(value: float, factor: int) -> int:
    """Ceil a positive value to a positive multiple of factor."""
    return max(factor, int(math.ceil(value / factor)) * factor)


def _round_to_factor(value: float, factor: int) -> int:
    """Round a positive value to a positive multiple of factor."""
    return max(factor, int(round(value / factor)) * factor)


def _smart_resize_dims(
    *,
    height: int,
    width: int,
    factor: int,
    max_pixels: int | None,
    min_pixels: int | None,
) -> tuple[int, int]:
    """Return factor-aligned image dimensions within pixel bounds."""
    if height <= 0 or width <= 0:
        return height, width

    resized_height = _round_to_factor(height, factor)
    resized_width = _round_to_factor(width, factor)
    current_pixels = resized_height * resized_width

    if max_pixels is not None and current_pixels > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_to_factor(height / beta, factor)
        resized_width = _floor_to_factor(width / beta, factor)

    current_pixels = resized_height * resized_width
    if min_pixels is not None and current_pixels < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_to_factor(height * beta, factor)
        resized_width = _ceil_to_factor(width * beta, factor)

    return resized_height, resized_width


def _resize_image_if_needed(
    *,
    image: Any,
    processor: Any,
    config: LlavaBenchJudgeConfig,
) -> Any:
    """Resize PIL-like images before processor tokenization.

    Some Qwen3-VL processor paths do not apply per-message pixel caps before
    visual placeholders are expanded. Directly resizing PIL-like image objects
    guarantees that the image grid reflects the requested static context budget.
    """
    max_pixels = _effective_image_max_pixels(config)
    if (
        max_pixels is None
        and config.resized_height is None
        and config.resized_width is None
        and config.image_min_pixels is None
    ):
        return image

    size = getattr(image, "size", None)
    if size is None or not hasattr(image, "resize"):
        return image

    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return image

    if config.resized_height is not None and config.resized_width is not None:
        new_height = int(config.resized_height)
        new_width = int(config.resized_width)
    else:
        new_height, new_width = _smart_resize_dims(
            height=height,
            width=width,
            factor=_processor_vision_factor(processor),
            max_pixels=max_pixels,
            min_pixels=config.image_min_pixels,
        )

    if (new_width, new_height) == (width, height):
        return image

    try:
        from PIL import Image

        resample = Image.Resampling.BICUBIC
    except Exception:
        resample = 3
    return image.resize((new_width, new_height), resample=resample)


def build_vlm_messages(
    question: str,
    image: Any | None = None,
) -> list[dict[str, Any]]:
    """Build open-ended LLaVA-Bench chat messages."""
    image_content: dict[str, Any] = {"type": "image"}
    if image is not None:
        image_content["image"] = image
    return [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": question},
            ],
        }
    ]


def build_vlm_prompt(processor: Any, question: str) -> str:
    """Build the open-ended LLaVA-Bench generation prompt text."""
    messages = build_vlm_messages(question)
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"<image>\n{question}"


def move_inputs_to_device(inputs: Any, device: str | torch.device) -> Any:
    """Move tensor-valued processor inputs to the requested device."""
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(device)
    return inputs


def _input_length(inputs: Any) -> int:
    """Return the input_ids sequence length from processor outputs."""
    return int(inputs["input_ids"].shape[1])


def _image_grid_for_error(inputs: Any) -> Any:
    """Return image grid data for an error message when available."""
    image_grid = inputs.get("image_grid_thw", None)
    if hasattr(image_grid, "tolist"):
        return image_grid.tolist()
    return image_grid


def format_context_length_hint(
    config: LlavaBenchJudgeConfig,
    *,
    input_len: int | None = None,
    image_grid_thw: Any | None = None,
) -> str:
    """Return an actionable hint for static-context LLaVA-Bench failures."""
    max_pixels = _effective_image_max_pixels(config)
    parts = [
        "The LLaVA-Bench sample does not fit into the requested static multimodal context.",
        f"max_seq_len={config.max_seq_len}",
        f"max_new_tokens={config.max_new_tokens}",
        f"input_budget={_input_max_length(config)}",
        f"effective_image_max_pixels={max_pixels}",
    ]
    if input_len is not None:
        parts.append(f"input_len={input_len}")
    if image_grid_thw is not None:
        parts.append(f"image_grid_thw={image_grid_thw}")
    parts.append(
        "Lower evaluation.llava_bench.max_new_tokens or "
        "evaluation.llava_bench.image_max_pixels."
    )
    return " ".join(parts)


def _validate_context_length(inputs: Any, config: LlavaBenchJudgeConfig) -> None:
    """Raise a clear error when the processed sample cannot fit the budget."""
    max_length = _input_max_length(config)
    if max_length is None:
        return
    input_len = _input_length(inputs)
    if input_len <= max_length:
        return
    raise ValueError(
        format_context_length_hint(
            config,
            input_len=input_len,
            image_grid_thw=_image_grid_for_error(inputs),
        )
    )


def build_vlm_processor_inputs(
    *,
    processor: Any,
    image: Any,
    question: str,
    config: LlavaBenchJudgeConfig,
    max_length: int | None,
) -> Any:
    """Build processor inputs for one LLaVA-Bench sample.

    The image is resized before tokenization. Token-level truncation is not
    passed to multimodal processors because truncating after visual placeholder
    expansion can corrupt image-token alignment.
    """
    _ = max_length
    image = _resize_image_if_needed(image=image, processor=processor, config=config)
    messages = build_vlm_messages(question=question, image=image)
    common_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
    }

    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                **common_kwargs,
            )
        except TypeError:
            pass

    prompt = build_vlm_prompt(processor, question)
    processor_kwargs: dict[str, Any] = {
        "text": prompt,
        "images": image,
        "return_tensors": "pt",
    }
    max_pixels = _effective_image_max_pixels(config)
    if config.image_min_pixels is not None:
        processor_kwargs["min_pixels"] = int(config.image_min_pixels)
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = int(max_pixels)
    return processor(**processor_kwargs)


@torch.no_grad()
def generate_one_answer(
    *,
    model: Any,
    processor: Any,
    image: Any,
    question: str,
    device: str,
    config: LlavaBenchJudgeConfig,
) -> str:
    """Generate one open-ended LLaVA-Bench answer."""
    inputs = build_vlm_processor_inputs(
        processor=processor,
        image=image,
        question=question,
        config=config,
        max_length=_input_max_length(config),
    )
    _validate_context_length(inputs, config)
    if device != "auto":
        inputs = move_inputs_to_device(inputs, device)

    do_sample = config.temperature > 0.0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = config.temperature

    out_ids = model.generate(**inputs, **gen_kwargs)
    input_len = _input_length(inputs)
    gen_ids = out_ids[0, input_len:]
    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_answers(
    *,
    model: Any,
    processor: Any,
    device: str,
    config: LlavaBenchJudgeConfig,
) -> list[dict[str, Any]]:
    """Generate candidate answers and write them as JSONL."""
    if model is None or processor is None:
        raise ValueError("A model and processor are required to generate answers.")

    records: list[dict[str, Any]] = []
    for index, raw in enumerate(iter_llava_bench_dataset(config), 1):
        item = normalize_dataset_example(raw)
        answer = generate_one_answer(
            model=model,
            processor=processor,
            image=item["image"],
            question=item["question"],
            device=device,
            config=config,
        )
        record = AnswerRecord(
            question_id=item["question_id"],
            image_id=item["image_id"],
            prompt=item["question"],
            text=answer,
            reference_answer=item["reference_answer"],
            model_id=config.candidate_label,
            context=item["context"],
            metadata=item["metadata"],
        ).to_json()
        records.append(record)
        if not config.quiet:
            print(f"[{index}] question_id={item['question_id']} answer={answer!r}")

    if config.answers_out is not None:
        write_jsonl(config.answers_out, records)
    return records


def load_judge(config: LlavaBenchJudgeConfig) -> tuple[Any, Any]:
    """Load the text-only judge model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch_dtype_from_name(config.judge_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        config.judge_model_id,
        trust_remote_code=config.trust_remote_code,
        token=config.hf_token,
        cache_dir=config.cache_dir,
    )
    load_kwargs = {
        "dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
        "token": config.hf_token,
        "cache_dir": config.cache_dir,
    }
    if config.judge_device == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.judge_model_id, **load_kwargs)
    if config.judge_device != "auto":
        model = model.to(config.judge_device)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator for image-based question answering. You compare two assistant responses to the same visual question using the given textual context and reference answer as grounding. Be strict about visual correctness, relevance, factual accuracy, completeness, helpfulness, and level of detail. Do not prefer an answer only because it is longer. Do not prefer an answer because of its position. Return only valid JSON."""


def build_judge_prompt(
    *,
    question: str,
    context: str,
    reference_answer: str,
    answer_a: str,
    answer_b: str,
) -> str:
    """Build a pairwise judge prompt."""
    if not context:
        context = (
            "No separate textual image context is available. Use the question and "
            "reference answer as the grounding signal."
        )
    return f"""Compare Answer A and Answer B for the following image-based question.

Scoring rubric:
- visual correctness
- relevance to the question
- factual accuracy
- completeness
- helpfulness
- level of detail

Assign each answer a score from 1 to 10. If both answers are similarly good, give similar scores.

Return only valid JSON with exactly this schema:
{{
  "score_a": number,
  "score_b": number,
  "winner": "A" | "B" | "tie",
  "reason": "brief explanation"
}}

[Image Context]
{context}

[Question]
{question}

[Reference Answer]
{reference_answer}

[Answer A]
{answer_a}

[Answer B]
{answer_b}
"""


@torch.no_grad()
def run_judge_once(
    *,
    tokenizer: Any,
    model: Any,
    config: LlavaBenchJudgeConfig,
    prompt: str,
) -> str:
    """Run the judge model on one prompt."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = JUDGE_SYSTEM_PROMPT + "\n\n" + prompt

    inputs = tokenizer(text, return_tensors="pt")
    if config.judge_device != "auto":
        inputs = {key: value.to(config.judge_device) for key, value in inputs.items()}
    do_sample = config.judge_temperature > 0.0
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": config.judge_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = config.judge_temperature
    output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _strip_code_fence(text: str) -> str:
    """Remove a Markdown code fence around JSON when present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def clamp_score(value: Any) -> float:
    """Clamp a judge score to the inclusive 0-to-10 range."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return max(0.0, min(10.0, score))


def parse_judge_output(text: str) -> tuple[float, float, str, str]:
    """Parse judge JSON output with a conservative fallback."""
    parsed: dict[str, Any] | None = None
    cleaned = _strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
    if parsed is None:
        score_matches = re.findall(
            r"(?:score[_\s-]*[ab]|\b[ab]\b)[^0-9]{0,12}(10(?:\.0)?|[1-9](?:\.\d+)?)",
            cleaned,
            flags=re.IGNORECASE,
        )
        if len(score_matches) < 2:
            score_matches = re.findall(r"\b(10(?:\.0)?|[1-9](?:\.\d+)?)\b", cleaned)
        if len(score_matches) >= 2:
            score_a = clamp_score(score_matches[0])
            score_b = clamp_score(score_matches[1])
            winner = "A" if score_a > score_b else "B" if score_b > score_a else "tie"
            return score_a, score_b, winner, "Parsed scores with fallback regex."
        return 0.0, 0.0, "tie", "Failed to parse judge output."

    score_a = clamp_score(parsed.get("score_a", 0.0))
    score_b = clamp_score(parsed.get("score_b", 0.0))
    winner = str(parsed.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie"}:
        winner = "A" if score_a > score_b else "B" if score_b > score_a else "tie"
    reason = str(parsed.get("reason", ""))
    return score_a, score_b, winner, reason


def record_text(record: dict[str, Any]) -> str:
    """Extract an answer string from a record with common answer keys."""
    for key in ("text", "answer", "caption", "response", "prediction"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return ""


def index_answers(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index answer records by question ID."""
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = str(
            record.get("question_id")
            or record.get("id")
            or record.get("image_id")
            or ""
        )
        if question_id:
            indexed[question_id] = record
    return indexed


def synthetic_reference_baseline(candidate: dict[str, Any]) -> dict[str, Any]:
    """Create a synthetic baseline answer from the reference answer."""
    return {
        "question_id": candidate.get("question_id", ""),
        "prompt": candidate.get("prompt", ""),
        "text": candidate.get("reference_answer", ""),
        "reference_answer": candidate.get("reference_answer", ""),
        "context": candidate.get("context", ""),
        "model_id": "reference",
        "metadata": candidate.get("metadata", {}),
    }


def judge_pair(
    *,
    tokenizer: Any,
    model: Any,
    config: LlavaBenchJudgeConfig,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    swap: bool,
) -> JudgeResult:
    """Judge one baseline/candidate answer pair."""
    if swap:
        answer_a = record_text(candidate)
        answer_b = record_text(baseline)
    else:
        answer_a = record_text(baseline)
        answer_b = record_text(candidate)

    prompt = build_judge_prompt(
        question=str(candidate.get("prompt", baseline.get("prompt", ""))),
        context=str(candidate.get("context") or baseline.get("context", "")),
        reference_answer=str(
            candidate.get("reference_answer", baseline.get("reference_answer", ""))
        ),
        answer_a=answer_a,
        answer_b=answer_b,
    )
    raw = run_judge_once(tokenizer=tokenizer, model=model, config=config, prompt=prompt)
    score_a, score_b, winner_ab, reason = parse_judge_output(raw)
    if swap:
        score_baseline = score_b
        score_candidate = score_a
        winner = (
            "candidate"
            if winner_ab == "A"
            else "baseline"
            if winner_ab == "B"
            else "tie"
        )
    else:
        score_baseline = score_a
        score_candidate = score_b
        winner = (
            "baseline"
            if winner_ab == "A"
            else "candidate"
            if winner_ab == "B"
            else "tie"
        )

    return JudgeResult(
        question_id=str(candidate.get("question_id", "")),
        baseline_label=str(baseline.get("model_id", config.baseline_label)),
        candidate_label=str(candidate.get("model_id", config.candidate_label)),
        score_baseline=score_baseline,
        score_candidate=score_candidate,
        winner=winner,
        reason=reason,
        raw_judge_output=raw,
        metadata={"swapped": swap},
    )


def merge_swapped_results(first: JudgeResult, second: JudgeResult) -> JudgeResult:
    """Average canonical scores from normal and swapped judge prompts."""
    score_baseline = (first.score_baseline + second.score_baseline) / 2.0
    score_candidate = (first.score_candidate + second.score_candidate) / 2.0
    if abs(score_baseline - score_candidate) < 1e-6:
        winner = "tie"
    else:
        winner = "baseline" if score_baseline > score_candidate else "candidate"
    return JudgeResult(
        question_id=first.question_id,
        baseline_label=first.baseline_label,
        candidate_label=first.candidate_label,
        score_baseline=score_baseline,
        score_candidate=score_candidate,
        winner=winner,
        reason=(
            "Order-averaged result. Normal order reason: "
            f"{first.reason} Swapped order reason: {second.reason}"
        ),
        raw_judge_output=json.dumps(
            {"normal": first.raw_judge_output, "swapped": second.raw_judge_output},
            ensure_ascii=False,
        ),
        metadata={"swapped": True},
    )


def judge_answers(
    *,
    candidates: list[dict[str, Any]],
    config: LlavaBenchJudgeConfig,
) -> list[dict[str, Any]]:
    """Judge candidate answers against a baseline or reference answer."""
    if not config.judge_enabled:
        return []
    tokenizer, judge_model = load_judge(config)
    baseline_by_id = None
    if config.baseline_answers_path:
        baseline_by_id = index_answers(read_jsonl(config.baseline_answers_path))

    results = []
    for index, candidate in enumerate(candidates, 1):
        question_id = str(candidate.get("question_id", ""))
        if baseline_by_id is not None:
            if question_id not in baseline_by_id:
                raise KeyError(
                    f"Missing baseline answer for question_id={question_id}."
                )
            baseline = baseline_by_id[question_id]
        else:
            baseline = synthetic_reference_baseline(candidate)

        first = judge_pair(
            tokenizer=tokenizer,
            model=judge_model,
            config=config,
            baseline=baseline,
            candidate=candidate,
            swap=False,
        )
        if config.judge_swap_order:
            second = judge_pair(
                tokenizer=tokenizer,
                model=judge_model,
                config=config,
                baseline=baseline,
                candidate=candidate,
                swap=True,
            )
            result = merge_swapped_results(first, second)
        else:
            result = first
        results.append(result.to_json())
        if not config.quiet:
            print(
                f"[{index}] question_id={question_id} "
                f"baseline={result.score_baseline:.2f} "
                f"candidate={result.score_candidate:.2f} winner={result.winner}"
            )
    if config.reviews_out is not None:
        write_jsonl(config.reviews_out, results)
    return results


def summarize_answers(
    *,
    candidates: list[dict[str, Any]],
    config: LlavaBenchJudgeConfig,
) -> dict[str, Any]:
    """Summarize generated or loaded answers when judge scoring is disabled."""
    summary = {
        "count": len(candidates),
        "judge_enabled": False,
        "candidate_label": config.candidate_label,
        "answers_out": config.answers_out,
        "reviews_out": config.reviews_out,
        "summary_out": config.summary_out,
        "effective_image_max_pixels": _effective_image_max_pixels(config),
    }
    if config.summary_out is not None:
        write_json(config.summary_out, summary)
    return summary


def summarize_reviews(
    *,
    reviews: list[dict[str, Any]],
    config: LlavaBenchJudgeConfig,
) -> dict[str, Any]:
    """Summarize judge review records and write the summary JSON."""
    if not reviews:
        summary = {
            "count": 0,
            "error": "No reviews were produced.",
            "judge_model_id": config.judge_model_id,
            "answers_out": config.answers_out,
            "reviews_out": config.reviews_out,
            "summary_out": config.summary_out,
            "effective_image_max_pixels": _effective_image_max_pixels(config),
        }
        if config.summary_out is not None:
            write_json(config.summary_out, summary)
        return summary

    baseline_scores = [float(r["score_baseline"]) for r in reviews]
    candidate_scores = [float(r["score_candidate"]) for r in reviews]
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    candidate_mean = sum(candidate_scores) / len(candidate_scores)
    relative = 0.0 if baseline_mean == 0 else candidate_mean / baseline_mean * 100.0
    wins = {"baseline": 0, "candidate": 0, "tie": 0}
    for review in reviews:
        winner = str(review.get("winner", "tie"))
        wins[winner] = wins.get(winner, 0) + 1
    summary = {
        "count": len(reviews),
        "judge_enabled": True,
        "judge_model_id": config.judge_model_id,
        "baseline_label": reviews[0].get("baseline_label", config.baseline_label),
        "candidate_label": reviews[0].get("candidate_label", config.candidate_label),
        "mean_baseline_score": baseline_mean,
        "mean_candidate_score": candidate_mean,
        "candidate_relative_score": relative,
        "mean_score_delta_candidate_minus_baseline": candidate_mean - baseline_mean,
        "wins": wins,
        "swap_order": bool(config.judge_swap_order),
        "answers_out": config.answers_out,
        "reviews_out": config.reviews_out,
        "summary_out": config.summary_out,
        "effective_image_max_pixels": _effective_image_max_pixels(config),
        "note": (
            "This is a LLaVA-Bench-style judge result. It is not directly "
            "comparable to official GPT-4-judge scores unless the same judge "
            "model and rubric are used."
        ),
    }
    if config.summary_out is not None:
        write_json(config.summary_out, summary)
    return summary


def evaluate_llava_bench_with_judge(
    *,
    model: Any,
    processor: Any,
    device: str,
    config: LlavaBenchJudgeConfig,
) -> dict[str, Any]:
    """Run LLaVA-Bench answer generation and optional judge evaluation."""
    config = resolve_output_paths(config)

    if config.candidate_answers_path and not config.regenerate:
        candidates = read_jsonl(config.candidate_answers_path)
        if (
            config.answers_out is not None
            and config.answers_out != config.candidate_answers_path
        ):
            write_jsonl(config.answers_out, candidates)
    else:
        candidates = generate_answers(
            model=model,
            processor=processor,
            device=device,
            config=config,
        )

    if not config.judge_enabled:
        return summarize_answers(candidates=candidates, config=config)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reviews = judge_answers(candidates=candidates, config=config)
    return summarize_reviews(reviews=reviews, config=config)


def print_llava_bench_judge_summary(summary: dict[str, Any]) -> None:
    """Print a compact LLaVA-Bench judge summary."""
    print("\n=== LLaVA-Bench Judge Evaluation ===")
    if summary.get("count", 0) == 0:
        print(summary.get("error", "No reviews were produced."))
        return
    print(f"count                         {int(summary['count'])}")
    print(f"candidate_label               {summary.get('candidate_label')}")
    print(f"judge_enabled                 {summary.get('judge_enabled')}")
    if summary.get("judge_enabled"):
        print(f"judge_model_id                {summary.get('judge_model_id')}")
        print(f"baseline_label                {summary.get('baseline_label')}")
        print(f"mean_baseline_score           {summary['mean_baseline_score']:.3f}")
        print(f"mean_candidate_score          {summary['mean_candidate_score']:.3f}")
        print(
            f"candidate_relative_score      {summary['candidate_relative_score']:.2f}"
        )
        print(
            "mean_score_delta             "
            f"{summary['mean_score_delta_candidate_minus_baseline']:.3f}"
        )
        print(f"wins                          {summary['wins']}")
    print(f"effective_image_max_pixels    {summary.get('effective_image_max_pixels')}")
    print(f"answers_out                   {summary.get('answers_out')}")
    print(f"reviews_out                   {summary.get('reviews_out')}")
    print(f"summary_out                   {summary.get('summary_out')}")
