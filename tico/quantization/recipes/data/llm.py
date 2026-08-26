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

import random
from typing import Any

import torch
from datasets import load_dataset

from tico.quantization.recipes.data.dataset_usage import (
    CALIBRATION_ROLE,
    resolve_dataset_usage,
    validate_single_dataset_usage,
)


def build_wikitext_calibration_inputs(
    *,
    tokenizer: Any,
    cache_dir: str | None,
    n_samples: int,
    seq_len: int,
    seed: int,
    device: torch.device | str,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    allow_benchmark_overlap: bool = False,
) -> list[torch.Tensor]:
    """Build deterministic causal-LM calibration windows from a text corpus."""
    usage = resolve_dataset_usage(
        dataset=dataset_name,
        role=CALIBRATION_ROLE,
        split=split,
        config=dataset_config,
        consumer="build_wikitext_calibration_inputs",
        n_samples=n_samples,
    )
    validate_single_dataset_usage(
        usage,
        allow_benchmark_overlap=allow_benchmark_overlap,
    )

    if usage.config is None:
        dataset = load_dataset(
            usage.canonical_id,
            split=usage.split,
            cache_dir=cache_dir,
        )
    else:
        dataset = load_dataset(
            usage.canonical_id,
            usage.config,
            split=usage.split,
            cache_dir=cache_dir,
        )
    text = " ".join(dataset["text"])
    token_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

    if token_ids.shape[1] <= seq_len + 1:
        raise ValueError(
            f"Calibration corpus is too short for seq_len={seq_len}. "
            f"token_count={token_ids.shape[1]}"
        )

    random.seed(seed)
    samples: list[torch.Tensor] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    for _ in range(n_samples):
        i = random.randint(0, token_ids.shape[1] - seq_len - 1)
        w = token_ids[:, i : i + seq_len]
        if bos is not None:
            w = torch.cat(
                [torch.tensor([[bos]], device=w.device), w[:, : seq_len - 1]], dim=1
            )
        samples.append(w.cpu())
    return samples
