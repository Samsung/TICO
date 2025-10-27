# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

from typing import Optional

import torch
import tqdm


def perplexity(
    model: torch.nn.Module,
    encodings: torch.Tensor,
    device: torch.device | str,
    *,
    max_length: Optional[int] = None,
    stride: int = 512,
    ignore_index: int | None = -100,
    show_progress: bool = True,
) -> float:
    """
    Compute perplexity (PPL) using a "strided sliding-window"
     evaluation strategy.

    The function:
    1. Splits the token sequence into overlapping windows of length
       `max_length` (model context size).
    2. Masks tokens that were already scored in previous windows
       (`labels == -100`), so each token's negative log-likelihood (NLL)
       is counted EXACTLY once.
    3. Aggregates token-wise NLL to return corpus-level PPL.

    Parameters
    ----------
    model : torch.nn.Module
        Causal LM loaded in evaluation mode (`model.eval()`).
    encodings : torch.Tensor | transformers.BatchEncoding
        Tokenised corpus.  If a `BatchEncoding` is passed, its
        `.input_ids` field is used.  Shape must be `(1, seq_len)`.
    device : torch.device | str
        CUDA or CPU device on which to run evaluation.
    max_length : int, optional
        Context window size.  Defaults to `model.config.max_position_embeddings`.
    stride : int, default = 512
        Step size by which the sliding window advances.  Must satisfy
        `1 ≤ stride ≤ max_length`.
    ignore_index : int, default = -100
        Label value to ignore in loss computation. This should match
        the `ignore_index` used by the model's internal
        `CrossEntropyLoss`. For Hugging Face causal LMs, the
        convention is `-100`.
    show_progress : bool, default = True
        If True, displays a tqdm progess bar while evaluating.

    Returns
    -------
    float
        Corpus-level perplexity.
    """
    # -------- input preparation -------- #
    try:
        # transformers.BatchEncoding has `input_ids`
        input_ids_full = encodings.input_ids  # type: ignore[attr-defined]
    except AttributeError:  # already a tensor
        input_ids_full = encodings
    assert isinstance(input_ids_full, torch.Tensor)
    input_ids_full = input_ids_full.to(device)

    if max_length is None:
        assert hasattr(model, "config")
        assert hasattr(model.config, "max_position_embeddings")
        assert isinstance(model.config.max_position_embeddings, int)
        max_length = model.config.max_position_embeddings
    assert max_length is not None
    assert (
        1 <= stride <= max_length
    ), f"stride ({stride}) must be in [1, max_length ({max_length})]"

    seq_len = input_ids_full.size(1)
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0

    # -------- main loop -------- #
    for begin in tqdm.trange(0, seq_len, stride, desc="PPL", disable=not show_progress):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end  # fresh tokens in this window

        input_ids = input_ids_full[:, begin:end]
        target_ids = input_ids.clone()
        # mask previously-scored tokens
        target_ids[:, :-trg_len] = ignore_index  # type: ignore[assignment]

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is already averaged over non-masked labels
            neg_log_likelihood = outputs.loss

        # exact number of labels that contributed to loss
        loss_tokens = (target_ids[:, 1:] != ignore_index).sum().item()  # type: ignore[attr-defined]
        nll_sum += neg_log_likelihood * loss_tokens
        n_tokens += int(loss_tokens)

        prev_end = end
        if end == seq_len:
            break

    avg_nll: float | torch.Tensor = nll_sum / n_tokens
    if not isinstance(avg_nll, torch.Tensor):
        avg_nll = torch.tensor(avg_nll)
    assert isinstance(avg_nll, torch.Tensor)
    ppl = torch.exp(avg_nll)

    return ppl.item()
