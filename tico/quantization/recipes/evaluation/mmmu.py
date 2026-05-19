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

from typing import Any

from tico.quantization.evaluation.mmmu_eval_utils import (
    evaluate_mmmu,
    print_mmmu_results,
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
    max_new_tokens: int,
    max_seq_len: int | None,
    temperature: float,
    verbose: bool,
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
    )
    print_mmmu_results(results)
    return results
