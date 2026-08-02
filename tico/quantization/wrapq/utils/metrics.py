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

from typing import Any, Optional

import math
import torch
import torch.nn.functional as F
import tqdm
from collections.abc import Iterable


def _resolve_device(device: torch.device, model: torch.nn.Module):
    if device != "auto":
        return torch.device(device)

    for p in model.parameters():
        if p is not None:
            return p.device

    # fallback
    return torch.device("cpu")


def perplexity(
    model: torch.nn.Module,
    encodings: torch.Tensor,
    device: torch.device | str = "auto",
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
    device = _resolve_device(device, model)
    input_ids_full = input_ids_full.to(device)

    if max_length is None:
        if hasattr(model, "config"):
            assert hasattr(model, "config")
            model_config = model.config
        else:
            assert hasattr(model.wrapped, "config")
            model_config = model.wrapped.config

        if hasattr(model_config, "text_config"):
            model_config = model_config.text_config
        assert hasattr(model_config, "max_position_embeddings")
        assert isinstance(model_config.max_position_embeddings, int)
        max_length = model_config.max_position_embeddings
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


def _extract_logits(outputs: Any) -> torch.Tensor:
    """Support Hugging Face ModelOutput, tuple output, and TICO raw tensors."""
    if isinstance(outputs, torch.Tensor):
        return outputs

    if hasattr(outputs, "logits"):
        return outputs.logits

    if isinstance(outputs, (tuple, list)) and outputs:
        if isinstance(outputs[0], torch.Tensor):
            return outputs[0]

    raise TypeError(
        f"Cannot extract logits from model output of type {type(outputs)!r}"
    )


def perplexity_chat_continuation(
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    dataset: Iterable[dict[str, Any]],
    *,
    stride: int = 512,
    max_seq_len: int = 2048,
    device: torch.device | str | None = None,
    show_progress: bool = True,
) -> float:

    """
    Compute sliding-window causal PPL for raw text on Gemma4 IT models.

    Gemma4 instruction-tuned models require a proper chat-template prefix
    (user turn + model turn with empty thought channel) to produce
    meaningful perplexity.  Without this prefix the model sees raw text
    in an unexpected context and produces astronomically high PPL values.

    The dataset must yield dictionaries containing a `text` field.
    """

    if max_seq_len <= 1:
        raise ValueError("max_seq_len must be greater than 1.")

    if not 1 <= stride <= max_seq_len:
        raise ValueError("stride must satisfy 1 <= stride <= max_seq_len.")

    tokenizer = getattr(
        processor_or_tokenizer,
        "tokenizer",
        processor_or_tokenizer,
    )

    # ------------------------------------------------------------------
    # Build the Gemma4 IT prompt prefix
    # ------------------------------------------------------------------
    # Gemma4 IT models need a proper chat-template prefix to produce
    # correct predictions.  The best prefix is:
    # BOS + user turn + model turn with empty thought channel.
    #
    #   <bos><|turn>user\nContinue the following text:\n<turn|>
    #   <|turn>model\n<turn|><|channel>thought\n<channel|>
    #
    # The wikitext tokens then follow as the model's "response" and only
    # those tokens are scored (the prefix is masked with -100).
    prefix_str = (
        "<bos><|turn>user\nContinue the following text:\n<turn|>"
        "<|turn>model\n<turn|><|channel>thought\n<channel|>"
    )
    prefix_encodings = tokenizer(
        prefix_str, return_tensors="pt", add_special_tokens=False
    )
    prefix_ids = prefix_encodings["input_ids"]  # shape [1, prefix_len]
    prefix_len = prefix_ids.shape[1]

    # Reserve slots for the prefix so the window still fits within max_seq_len.
    window_size = max_seq_len - prefix_len

    if window_size <= 0:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) is too small for the Gemma4 prefix "
            f"({prefix_len} tokens).  Increase max_seq_len."
        )

    # Iteration works for both Dataset and IterableDataset.
    texts = []
    for example in dataset:
        text = example.get("text", "")
        if text is not None:
            texts.append(str(text))

    if not texts:
        raise ValueError("The dataset contains no text examples.")

    full_text = "\n\n".join(texts)

    encodings = tokenizer(
        text=full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = encodings["input_ids"]

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"Expected tokenized shape [1, sequence], got {input_ids.shape}."
        )

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Move prefix to device once
    prefix_ids = prefix_ids.to(device)

    model.eval()

    sequence_length = input_ids.shape[1]
    previous_end = 0

    total_nll = torch.zeros((), dtype=torch.float64, device=device)
    total_target_tokens = 0
    window_count = 0

    with torch.inference_mode():
        for begin in tqdm.trange(0, sequence_length, stride, desc="PPL", disable=not show_progress):

            end = min(begin + window_size, sequence_length)

            # Only tokens not evaluated by the previous window contribute.
            target_length = end - previous_end
            if target_length <= 0:
                break

            window_ids = input_ids[:, begin:end].to(device)

            labels = window_ids.clone()
            labels[:, :-target_length] = -100

            # Prepend the Gemma4 IT prefix (BOS + user turn + model turn
            # with empty thought channel) so the model sees a valid
            # instruction-tuned context.  All prefix tokens are masked (-100)
            # so they never contribute to loss.
            window_ids = torch.cat([prefix_ids, window_ids], dim=1)
            prefix_labels = torch.full(
                (1, prefix_len), -100, dtype=labels.dtype, device=device
            )
            labels = torch.cat([prefix_labels, labels], dim=1)

            # Build attention mask: all ones (prefix + content)
            attention_mask = torch.ones(
                1, window_ids.shape[1], dtype=torch.long, device=device
            )

            model_kwargs = {
                "input_ids": window_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            }

            # Prevent an implementation from returning only selected logits.
            # Remove this argument if a particular wrapper does not accept it.
            model_kwargs["logits_to_keep"] = 0

            try:
                outputs = model(**model_kwargs)
            except TypeError as exc:
                # Compatibility fallback for wrappers without logits_to_keep.
                if "logits_to_keep" not in str(exc):
                    raise
                model_kwargs.pop("logits_to_keep")
                outputs = model(**model_kwargs)

            logits = _extract_logits(outputs)

            if logits.shape[:2] != window_ids.shape:
                raise ValueError(
                    "Logit sequence shape does not match input shape: "
                    f"logits={logits.shape}, input={window_ids.shape}. "
                    "Check logits_to_keep or wrapper truncation."
                )

            # Standard causal shift: logit at t predicts token t+1.
            shift_logits = logits[:, :-1, :].float().contiguous()
            shift_labels = labels[:, 1:].contiguous()

            target_tokens = int((shift_labels != -100).sum().item())

            if target_tokens:
                window_nll = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                total_nll += window_nll.double()
                total_target_tokens += target_tokens

            window_count += 1
            previous_end = end

            if end == sequence_length:
                break

    if total_target_tokens == 0:
        raise ValueError("No target tokens were evaluated.")

    mean_nll = (total_nll / total_target_tokens).item()
    ppl = math.exp(mean_nll)

    return ppl

