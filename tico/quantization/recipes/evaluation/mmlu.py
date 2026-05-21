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

from tico.quantization.evaluation.mmlu_eval_utils import (
    evaluate_mmlu,
    print_mmlu_results,
)


def evaluate_and_print_mmlu(
    *,
    model: Any,
    tokenizer: Any,
    subjects: list[str],
    device: str,
    n_shots: int,
    n_samples: int,
    batch_size: int,
    max_seq_len: int,
):
    results = evaluate_mmlu(
        model=model,
        tokenizer=tokenizer,
        subjects=subjects,
        device=device,
        n_shots=n_shots,
        n_samples=n_samples,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    print_mmlu_results(results)
    return results
