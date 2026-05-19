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

from tico.quantization.evaluation.hellaswag_eval_utils import (
    evaluate_hellaswag,
    get_hellaswag_accuracy,
    print_hellaswag_results,
)


def evaluate_and_print_hellaswag(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    n_shots: int,
    n_samples: int,
    batch_size: int,
    max_seq_len: int | None,
) -> dict[str, Any]:
    results = evaluate_hellaswag(
        model=model,
        tokenizer=tokenizer,
        device=device,
        n_shots=n_shots,
        n_samples=n_samples,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
    )
    print_hellaswag_results(results)
    acc = get_hellaswag_accuracy(results)
    print(f"Accuracy: {acc['acc']:.4f}, Accuracy (norm): {acc['acc_norm']:.4f}")
    return results
