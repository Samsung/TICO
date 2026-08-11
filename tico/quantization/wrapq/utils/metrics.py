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

import math
from collections.abc import Iterable
from typing import Any, Optional

import torch
import torch.nn.functional as F
import tqdm


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


_CHAT_PREFIX_INSTRUCTION = (
    "You are a text continuation engine. " "Complete the following passage exactly."
)


def _render_chat_prefix(
    processor_or_tokenizer: Any,
    tokenizer: Any,
) -> str:
    """Render the generation prefix with the model's chat template."""
    template_owner = None
    if hasattr(processor_or_tokenizer, "apply_chat_template"):
        template_owner = processor_or_tokenizer
    elif hasattr(tokenizer, "apply_chat_template"):
        template_owner = tokenizer

    if template_owner is None:
        raise ValueError(
            "chat-prefix perplexity requires a processor or tokenizer "
            "with apply_chat_template support."
        )

    text_content = _CHAT_PREFIX_INSTRUCTION
    multimodal_content = [{"type": "text", "text": text_content}]
    if hasattr(template_owner, "tokenizer"):
        content_variants: tuple[Any, ...] = (multimodal_content, text_content)
    else:
        content_variants = (text_content, multimodal_content)
    last_error: Exception | None = None
    for content in content_variants:
        messages = [{"role": "user", "content": content}]
        try:
            rendered = template_owner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError) as exc:
            last_error = exc
            continue

        if not isinstance(rendered, str):
            raise TypeError(
                "apply_chat_template(tokenize=False) must return a string, "
                f"got {type(rendered)!r}."
            )
        if not rendered:
            raise ValueError("apply_chat_template returned an empty chat prefix.")
        return rendered

    raise TypeError(
        "Failed to render chat prefix with apply_chat_template."
    ) from last_error


def perplexity_chat_prefix(
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

    **Algorithm**

    1. A fixed chat-template prefix is rendered **once** via
       :func:`_render_chat_prefix` using the model's own
       ``apply_chat_template``.  The prefix contains a user turn with
       a text-continuation instruction and a generation prompt
       (``add_generation_prompt=True``).  This places the model in
       "continuation mode" so it expects raw text to follow.
    2. The full corpus is tokenised **once** (no special tokens) and
       concatenated into a single 1-D token stream.
    3. A sliding window of ``max_seq_len - prefix_len`` content tokens
       advances by ``stride`` positions per step.
    4. For each window:
       a. The chat prefix is prepended to the content tokens.
       b. ``target_length`` identifies the newly covered content tokens,
          which always occupy the final positions of the window.
       c. The model is called with ``input_ids``, ``attention_mask``,
          ``use_cache=False``, and
          ``logits_to_keep=target_length + 1``.  This limits the LM-head
          projection to the minimal suffix required for causal alignment.
       d. The final returned logit is discarded because it predicts the
          token following the current window.  The remaining logits are
          aligned with the final ``target_length`` input IDs, and their
          summed cross-entropy is accumulated.
    5. The per-window NLL sums and token counts are accumulated, and
       the corpus-level PPL is ``exp(total_nll / total_tokens)``.

    **Key design choices**

    * The prefix is **constant** across all windows.
    * Prefix and overlapping context tokens are never used as targets.
    * Each corpus token is scored **exactly once** — in the first
      window whose target range covers it — matching the standard
      sliding-window PPL semantics.
    * ``logits_to_keep`` reduces LM-head compute and logits memory.
      Transformer hidden states are still computed for the full context.

    Parameters
    ----------
    model : torch.nn.Module
        Causal LM (original or quantized) in eval mode.
    processor_or_tokenizer : Any
        Processor or tokenizer with ``apply_chat_template`` support.
    dataset : Iterable[dict[str, Any]]
        Dataset yielding dictionaries with a ``text`` field
        (e.g. HuggingFace ``wikitext-2-raw-v1``).
    stride : int
        Sliding-window step size.  Must satisfy
        ``1 <= stride <= max_seq_len - prefix_len``.
    max_seq_len : int
        Maximum total sequence length (prefix + content) per window.
    device : torch.device | str | None
        Device for computation.  Auto-detected from model parameters
        if None.
    show_progress : bool
        Show a tqdm progress bar.

    Returns
    -------
    float
        Corpus-level perplexity.

    Raises
    ------
    ValueError
        If ``max_seq_len`` is too small for the chat prefix, if
        ``stride`` exceeds the content window size, or if no target
        tokens were evaluated.
    RuntimeError
        If the model output does not provide logits or does not honor
        ``logits_to_keep``.
    """

    if max_seq_len <= 1:
        raise ValueError("max_seq_len must be greater than 1.")

    if stride < 1:
        raise ValueError("stride must be positive.")

    tokenizer = getattr(
        processor_or_tokenizer,
        "tokenizer",
        processor_or_tokenizer,
    )

    prefix_str = _render_chat_prefix(processor_or_tokenizer, tokenizer)
    prefix_encodings = tokenizer(
        prefix_str, return_tensors="pt", add_special_tokens=False
    )
    prefix_ids = prefix_encodings["input_ids"]  # shape [1, prefix_len]
    prefix_len = prefix_ids.shape[1]

    # Reserve slots for the prefix so the window still fits within max_seq_len.
    window_size = max_seq_len - prefix_len

    if window_size <= 0:
        raise ValueError(
            f"max_seq_len ({max_seq_len}) is too small for the chat prefix "
            f"({prefix_len} tokens).  Increase max_seq_len."
        )

    if stride > window_size:
        raise ValueError(
            "stride must satisfy 1 <= stride <= max_seq_len - prefix_len "
            f"({window_size}); got stride={stride}."
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

    with torch.inference_mode():
        for begin in tqdm.trange(
            0, sequence_length, stride, desc="PPL", disable=not show_progress
        ):

            end = min(begin + window_size, sequence_length)

            # Only tokens not evaluated by the previous window contribute.
            target_length = end - previous_end
            if target_length <= 0:
                break

            content_window_ids = input_ids[:, begin:end].to(device)

            # Prepend the fixed chat prefix. Fresh targets always occupy the
            # final `target_length` positions of the resulting sequence.
            model_input_ids = torch.cat(
                [prefix_ids, content_window_ids],
                dim=1,
            )
            attention_mask = torch.ones_like(
                model_input_ids,
                dtype=torch.long,
            )

            # To score T target tokens, causal LM loss needs logits from the T
            # immediately preceding positions. Integer logits_to_keep can only
            # select a suffix, so request T + 1 positions and discard the final
            # logit, which predicts the token following the current window.
            logits_to_keep = target_length + 1
            outputs = model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )

            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError(
                    "Model output does not contain logits required for "
                    "chat-prefix perplexity."
                )

            if logits.ndim != 3 or logits.shape[1] != logits_to_keep:
                raise RuntimeError(
                    "Model did not honor logits_to_keep: expected logits with "
                    f"shape [B, {logits_to_keep}, V], got {tuple(logits.shape)}."
                )

            # The final logit predicts a token outside the current window.
            # Removing it aligns the remaining T logits with the final T input
            # token IDs, including the first fresh target's predecessor.
            target_logits = logits[:, :-1, :].contiguous()
            target_ids = model_input_ids[:, -target_length:].contiguous()

            # Hugging Face causal-LM loss upcasts logits to float before CE.
            # Match that behavior and accumulate the exact summed token NLL.
            window_nll = F.cross_entropy(
                target_logits.float().reshape(
                    -1,
                    target_logits.size(-1),
                ),
                target_ids.reshape(-1),
                reduction="sum",
            )

            total_nll += window_nll.double()
            total_target_tokens += target_length

            previous_end = end

            if end == sequence_length:
                break

    if total_target_tokens == 0:
        raise ValueError("No target tokens were evaluated.")

    mean_nll = (total_nll / total_target_tokens).item()
    ppl = math.exp(mean_nll)

    return ppl
