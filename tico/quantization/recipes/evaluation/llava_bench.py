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

"""Shared LLaVA-Bench target selection and routing for VLM model adapters.

The ``evaluation.llava_bench`` section accepts a mapping, a boolean, or null and
selects between the judge-based evaluator and the legacy COCO-style evaluator.
VLM adapters share this policy so that a change to it is made once instead of
once per model family. The evaluators themselves stay in
``llava_bench_judge.py`` and ``vlm.py``.
"""

from typing import Any, Mapping

from tico.quantization.recipes.evaluation.llava_bench_judge import (
    evaluate_and_print_llava_bench_judge,
)
from tico.quantization.recipes.evaluation.selection import should_run_evaluation
from tico.quantization.recipes.evaluation.vlm import (
    evaluate_llava_bench,
    print_coco_score_results,
)

LLAVA_BENCH_TARGET = "llava_bench"


def run_llava_bench_evaluation(
    *,
    eval_cfg: Mapping[str, Any],
    model: Any,
    processor: Any,
    device: str,
    model_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    default_n_samples: int,
    default_max_seq_len: int | None,
) -> dict[str, Any] | None:
    """Select, route, run, and print the ``llava_bench`` evaluation target.

    Selection follows ``should_run_evaluation``: ``evaluation.selected_tasks``
    is an exclusive allow-list when present; otherwise the nested
    ``enabled`` flag (mapping) or the truthiness of the value (boolean) decides.
    Mode and type validation only happens when the target is selected.

    Routing for a selected target:

    - mapping with ``mode`` in ``{"judge", "llm_judge"}`` (default ``judge``):
      judge evaluator with the mapping passed through as ``llava_cfg``;
    - mapping with ``mode`` in ``{"legacy", "coco", "caption"}``: legacy
      COCO-style evaluator, honoring nested ``n_samples``/``max_seq_len``
      overrides;
    - ``None`` or ``False`` (explicitly selected): judge evaluator with the
      default judge configuration;
    - ``True``: legacy COCO-style evaluator with a deprecation warning.

    Returns the evaluator result, or ``None`` when the target did not run.
    """
    raw_llava_cfg = eval_cfg.get(LLAVA_BENCH_TARGET)
    llava_default_enabled = (
        bool(raw_llava_cfg.get("enabled", False))
        if isinstance(raw_llava_cfg, Mapping)
        else bool(raw_llava_cfg)
    )
    if not should_run_evaluation(
        eval_cfg,
        LLAVA_BENCH_TARGET,
        default_enabled=llava_default_enabled,
    ):
        return None

    if isinstance(raw_llava_cfg, Mapping):
        llava_cfg = raw_llava_cfg
        mode = str(llava_cfg.get("mode", "judge")).lower()
        if mode in {"judge", "llm_judge"}:
            return evaluate_and_print_llava_bench_judge(
                model=model,
                processor=processor,
                device=device,
                llava_cfg=llava_cfg,
                model_cfg=model_cfg,
                runtime_cfg=runtime_cfg,
                default_n_samples=default_n_samples,
                default_max_seq_len=default_max_seq_len,
            )
        if mode in {"legacy", "coco", "caption"}:
            llava_results = evaluate_llava_bench(
                model=model,
                processor=processor,
                device=device,
                n_samples=int(llava_cfg.get("n_samples", default_n_samples)),
                max_seq_len=llava_cfg.get(
                    "max_seq_len",
                    default_max_seq_len,
                ),
            )
            print_coco_score_results(
                "\n=== LLaVA Bench Legacy COCO-style Evaluation ===",
                llava_results,
            )
            return llava_results
        raise ValueError(
            "evaluation.llava_bench.mode must be one of "
            "{'judge', 'llm_judge', 'legacy', 'coco', 'caption'}, "
            f"got {mode!r}."
        )

    if raw_llava_cfg is None or raw_llava_cfg is False:
        return evaluate_and_print_llava_bench_judge(
            model=model,
            processor=processor,
            device=device,
            llava_cfg={},
            model_cfg=model_cfg,
            runtime_cfg=runtime_cfg,
            default_n_samples=default_n_samples,
            default_max_seq_len=default_max_seq_len,
        )

    if raw_llava_cfg is True:
        print(
            "[WARNING] evaluation.llava_bench=true uses the legacy "
            "COCO-style CIDEr/BLEU path. Prefer the nested judge config: "
            "evaluation.llava_bench.enabled=true, mode=judge."
        )
        llava_results = evaluate_llava_bench(
            model=model,
            processor=processor,
            device=device,
            n_samples=default_n_samples,
            max_seq_len=default_max_seq_len,
        )
        print_coco_score_results(
            "\n=== Llava Bench Evaluation ===",
            llava_results,
        )
        return llava_results

    raise TypeError("evaluation.llava_bench must be a mapping, boolean, or null.")
