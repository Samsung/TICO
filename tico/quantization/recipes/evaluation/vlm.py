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
from typing import Any

from tico.quantization.evaluation.vlm_eval_utils import (
    evaluate_ppl,
    get_accuracy_on_dataset,
    get_coco_scores_on_dataset,
    get_dataset,
)


def evaluate_vqa_tasks(
    *,
    model: Any,
    processor: Any,
    tasks: list[str],
    device: str,
    n_samples: int,
    max_seq_len: int | None,
) -> dict[str, tuple[int, int]]:
    results: dict[str, tuple[int, int]] = {}
    for task in tasks:
        if "vqa" not in task:
            continue
        with io.StringIO() as buffer, contextlib.redirect_stdout(
            buffer
        ), contextlib.redirect_stderr(buffer):
            dataset, adapter = get_dataset(task, n=n_samples)
            correct, total = get_accuracy_on_dataset(
                model,
                processor,
                dataset,
                adapter,
                device,
                max_seq_len=max_seq_len,
            )
        results[task] = (correct, total)
    return results


def print_vqa_results(title: str, results: dict[str, tuple[int, int]]) -> None:
    print(title)
    for task, (correct, total) in results.items():
        print(f"{task}: EM={correct / total:.4f}  (n={total})")


def evaluate_coco(
    *,
    model: Any,
    processor: Any,
    device: str,
    n_samples: int,
    max_seq_len: int | None,
) -> dict[str, float]:
    with io.StringIO() as buffer, contextlib.redirect_stdout(
        buffer
    ), contextlib.redirect_stderr(buffer):
        dataset, _ = get_dataset("coco", n=n_samples)
        return get_coco_scores_on_dataset(
            model=model,
            processor=processor,
            dataset_name="coco",
            ds=dataset,
            device=device,
            max_seq_len=max_seq_len,
        )


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
    dataset, _ = get_dataset(dataset_name, split=split, n=-1)
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
