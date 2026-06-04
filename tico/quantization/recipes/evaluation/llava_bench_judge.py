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

"""Recipe wrapper for judge-based LLaVA-Bench evaluation."""

from pathlib import Path
from typing import Any, Mapping

import torch

from tico.quantization.evaluation.llava_bench_judge_eval_utils import (
    DEFAULT_JUDGE_MODEL_ID,
    evaluate_llava_bench_with_judge,
    LLAVA_BENCH_DATASET,
    LlavaBenchJudgeConfig,
    print_llava_bench_judge_summary,
)


def _as_bool(value: Any, default: bool = False) -> bool:
    """Convert common config values to a boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    """Convert a config value to an integer with a default."""
    if value is None:
        return default
    return int(value)


def _optional_int(value: Any) -> int | None:
    """Return an integer value or None for empty config values."""
    if value is None:
        return None
    return int(value)


def _as_float(value: Any, default: float) -> float:
    """Convert a config value to a float with a default."""
    if value is None:
        return default
    return float(value)


def _optional_str(value: Any) -> str | None:
    """Return a string value or None for empty config values."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


def build_llava_bench_judge_config(
    *,
    llava_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    default_n_samples: int,
    default_max_seq_len: int | None,
    default_device: str,
) -> LlavaBenchJudgeConfig:
    """Build a typed LLaVA-Bench judge config from recipe mappings."""
    judge_cfg = llava_cfg.get("judge", {})
    if not isinstance(judge_cfg, Mapping):
        raise TypeError("evaluation.llava_bench.judge must be a mapping.")
    output_cfg = llava_cfg.get("output", {})
    if not isinstance(output_cfg, Mapping):
        raise TypeError("evaluation.llava_bench.output must be a mapping.")

    model_id = str(model_cfg.get("name_or_path", "candidate"))
    candidate_label = str(llava_cfg.get("candidate_label", model_id))
    output_dir = str(
        llava_cfg.get("output_dir")
        or output_cfg.get("dir")
        or Path("./out/llava_bench")
    )

    judge_device = str(judge_cfg.get("device", default_device))
    if judge_device == "auto":
        resolved_judge_device = "auto"
    elif judge_device:
        resolved_judge_device = judge_device
    else:
        resolved_judge_device = "cuda" if torch.cuda.is_available() else "cpu"

    raw_max_seq_len = llava_cfg.get("max_seq_len", default_max_seq_len)
    max_seq_len = None if raw_max_seq_len is None else int(raw_max_seq_len)

    return LlavaBenchJudgeConfig(
        dataset=str(llava_cfg.get("dataset", LLAVA_BENCH_DATASET)),
        split=str(llava_cfg.get("split", "train")),
        n_samples=_as_int(llava_cfg.get("n_samples"), default_n_samples),
        start_index=_as_int(llava_cfg.get("start_index"), 0),
        max_seq_len=max_seq_len,
        max_new_tokens=_as_int(llava_cfg.get("max_new_tokens"), 512),
        temperature=_as_float(llava_cfg.get("temperature"), 0.0),
        image_min_pixels=_optional_int(llava_cfg.get("image_min_pixels")),
        image_max_pixels=_optional_int(llava_cfg.get("image_max_pixels")),
        resized_height=_optional_int(llava_cfg.get("resized_height")),
        resized_width=_optional_int(llava_cfg.get("resized_width")),
        visual_token_margin=_as_int(llava_cfg.get("visual_token_margin"), 256),
        candidate_label=candidate_label,
        baseline_label=str(llava_cfg.get("baseline_label", "reference")),
        candidate_answers_path=_optional_str(llava_cfg.get("candidate_answers")),
        baseline_answers_path=_optional_str(llava_cfg.get("baseline_answers")),
        answers_out=_optional_str(
            llava_cfg.get("answers_out") or output_cfg.get("answers")
        ),
        reviews_out=_optional_str(
            llava_cfg.get("reviews_out") or output_cfg.get("reviews")
        ),
        summary_out=_optional_str(
            llava_cfg.get("summary_out") or output_cfg.get("summary")
        ),
        output_dir=output_dir,
        regenerate=_as_bool(llava_cfg.get("regenerate"), False),
        judge_enabled=_as_bool(judge_cfg.get("enabled"), True),
        judge_model_id=str(judge_cfg.get("model_id", DEFAULT_JUDGE_MODEL_ID)),
        judge_device=resolved_judge_device,
        judge_dtype=str(judge_cfg.get("dtype", "float16")),
        judge_max_new_tokens=_as_int(judge_cfg.get("max_new_tokens"), 256),
        judge_temperature=_as_float(judge_cfg.get("temperature"), 0.0),
        judge_swap_order=_as_bool(judge_cfg.get("swap_order"), True),
        trust_remote_code=_as_bool(
            judge_cfg.get(
                "trust_remote_code",
                llava_cfg.get(
                    "trust_remote_code", model_cfg.get("trust_remote_code", True)
                ),
            ),
            True,
        ),
        hf_token=_optional_str(
            judge_cfg.get("hf_token")
            or llava_cfg.get("hf_token")
            or model_cfg.get("hf_token")
        ),
        cache_dir=_optional_str(
            judge_cfg.get("cache_dir")
            or llava_cfg.get("cache_dir")
            or model_cfg.get("cache_dir")
        ),
        quiet=not _as_bool(runtime_cfg.get("show_progress"), True),
    )


def evaluate_and_print_llava_bench_judge(
    *,
    model: Any,
    processor: Any,
    device: str,
    llava_cfg: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    default_n_samples: int,
    default_max_seq_len: int | None,
) -> dict[str, Any]:
    """Evaluate LLaVA-Bench with open-ended generation and judge scoring."""
    config = build_llava_bench_judge_config(
        llava_cfg=llava_cfg,
        model_cfg=model_cfg,
        runtime_cfg=runtime_cfg,
        default_n_samples=default_n_samples,
        default_max_seq_len=default_max_seq_len,
        default_device=device,
    )
    summary = evaluate_llava_bench_with_judge(
        model=model,
        processor=processor,
        device=device,
        config=config,
    )
    print_llava_bench_judge_summary(summary)
    return summary
