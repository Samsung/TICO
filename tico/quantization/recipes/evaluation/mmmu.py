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

from typing import Any, Mapping

from tico.quantization.evaluation.mmmu_eval_utils import (
    DEFAULT_MMMU_PRO_VISION_PROMPT_MODE,
    evaluate_mmmu,
    print_mmmu_results,
    validate_mmmu_evaluation_dataset,
)
from tico.quantization.recipes.evaluation.selection import (
    get_mapping_evaluation_config,
    should_run_mapping_evaluation,
)


def validate_mmmu_recipe_config(cfg: Mapping[str, Any]) -> None:
    """Fail unsupported selected MMMU requests before model/calibration loading."""
    eval_cfg = cfg.get("evaluation")
    if not isinstance(eval_cfg, Mapping) or not eval_cfg.get("enabled", False):
        return
    if not should_run_mapping_evaluation(eval_cfg, "mmmu"):
        return
    mmmu = get_mapping_evaluation_config(eval_cfg, "mmmu")
    # Match run_mmmu_evaluation's dataset default; never silently switch it.
    validate_mmmu_evaluation_dataset(mmmu.get("dataset") or "MMMU/MMMU")


def run_mmmu_evaluation(
    *,
    eval_cfg: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: str,
    max_seq_len: int | None,
    verbose: bool,
):
    """Select MMMU and translate recipe settings into evaluator arguments."""
    if not should_run_mapping_evaluation(eval_cfg, "mmmu"):
        return None

    mmmu = get_mapping_evaluation_config(eval_cfg, "mmmu")
    subjects = mmmu.get("subjects")
    if subjects == ["mmmu"] or subjects == "mmmu":
        subjects = None
    return evaluate_and_print_mmmu(
        model=model,
        processor=processor,
        dataset=mmmu.get("dataset") or "MMMU/MMMU",
        subjects=subjects,
        device=device,
        n_shots=int(mmmu.get("n_shots", 5)),
        n_samples=int(mmmu.get("n_samples", -1)),
        max_new_tokens=mmmu.get("max_new_tokens"),
        max_seq_len=max_seq_len,
        temperature=float(mmmu.get("temperature", 0.0)),
        verbose=bool(mmmu.get("verbose", verbose)),
        prompt_mode=mmmu.get("prompt_mode"),
        input_max_seq_len=mmmu.get("input_max_seq_len"),
        output_jsonl=mmmu.get("output_jsonl"),
    )


def evaluate_and_print_mmmu(
    *,
    model: Any,
    processor: Any,
    dataset: str,
    subjects: list[str] | None,
    device: str,
    n_shots: int,
    n_samples: int,
    max_new_tokens: int | None,
    max_seq_len: int | None,
    temperature: float,
    verbose: bool,
    prompt_mode: str | None = None,
    input_max_seq_len: int | None = None,
    output_jsonl: str | None = None,
):
    results = evaluate_mmmu(
        model=model,
        processor=processor,
        dataset=dataset,
        subjects=subjects,
        device=device,
        n_shots=n_shots,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        max_seq_len=max_seq_len,
        temperature=temperature,
        verbose=verbose,
        prompt_mode=(
            DEFAULT_MMMU_PRO_VISION_PROMPT_MODE if prompt_mode is None else prompt_mode
        ),
        input_max_seq_len=input_max_seq_len,
        output_jsonl=output_jsonl,
    )
    print_mmmu_results(results)
    return results
