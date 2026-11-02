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

"""Teacher-forced cached-decode NLL/PPL on fixed held-out token chunks.

The existing perplexity helpers score every window with one uncached forward.
That never exercises the past-KV read path whose quantization policy a cached
deployment depends on. This helper scores each chunk the way a cached runtime
would produce it:

1. The prefix is prefilled once with ``use_cache=True``. Its last logit scores
   the first target token; the remaining prefix logits give a separate
   prefill-only score.
2. Each target token is then fed one at a time through the returned cache.
   The logit of step ``t`` scores target ``t + 1``, so ``T`` targets need
   ``T - 1`` decode steps and the final target is never fed.

Every chunk is a contiguous, non-overlapping token span of the same corpus
stream, so no token is scored twice and all variants see identical targets.

Context budget: with ``H`` chat-prefix tokens (0 in raw mode), ``P`` prefix
tokens and ``T`` targets, a chunk is accepted only when ``H + P + T <= M`` for
the resolved evaluation context ``M``. The true peak input/cache length is
``H + P + T - 1`` because the final target is never fed back, but the budget
deliberately keeps the whole-chunk contract and merely adds the previously
missing ``H``; the one-token slack is intentional.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn.functional as F
import tqdm


@dataclass(frozen=True)
class CachedNLLProtocol:
    """Fixed scoring budget shared by every model under comparison."""

    prefix_tokens: int
    target_tokens: int
    n_chunks: int
    start_offset: int = 0
    chat_prefix: bool = True

    def __post_init__(self) -> None:
        if self.prefix_tokens < 1:
            raise ValueError("prefix_tokens must be positive.")
        if self.target_tokens < 1:
            raise ValueError("target_tokens must be positive.")
        if self.n_chunks < 1:
            raise ValueError("n_chunks must be positive.")
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative.")

    @property
    def span(self) -> int:
        return self.prefix_tokens + self.target_tokens


def chunk_token_ranges(
    sequence_length: int, protocol: CachedNLLProtocol
) -> list[tuple[int, int]]:
    """Return ``n_chunks`` contiguous, non-overlapping ``[begin, end)`` ranges."""
    ranges = [
        (protocol.start_offset + index * protocol.span,)
        for index in range(protocol.n_chunks)
    ]
    needed = protocol.start_offset + protocol.n_chunks * protocol.span
    if needed > sequence_length:
        raise ValueError(
            "Corpus is too short for the cached NLL protocol: "
            f"{needed} tokens required (start_offset={protocol.start_offset}, "
            f"n_chunks={protocol.n_chunks}, span={protocol.span}), "
            f"got {sequence_length}."
        )
    return [(begin, begin + protocol.span) for (begin,) in ranges]


def _logits(outputs: Any, expected_positions: int) -> torch.Tensor:
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("Model output does not contain logits.")
    if (
        logits.ndim != 3
        or logits.shape[0] != 1
        or logits.shape[1] != expected_positions
    ):
        raise RuntimeError(
            f"Expected logits shaped [1, {expected_positions}, V], "
            f"got {tuple(logits.shape)}."
        )
    return logits.float()


def _cache(outputs: Any, expected_length: int) -> Any:
    """Require a real growing cache; an uncached forward cannot stand in."""
    cache = getattr(outputs, "past_key_values", None)
    get_length = getattr(cache, "get_seq_length", None)
    if not callable(get_length) or int(get_length()) != expected_length:
        raise RuntimeError(
            "Cached NLL requires past_key_values with cache length "
            f"{expected_length}; the model did not return a valid cache."
        )
    return cache


def _token_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return per-token NLL in float64 for ``logits[i]`` predicting ``targets[i]``."""
    if logits.shape[0] != targets.shape[0]:
        raise RuntimeError("Logit/target alignment mismatch.")
    if targets.numel() == 0:
        return torch.zeros((0,), dtype=torch.float64, device=logits.device)
    return F.cross_entropy(logits, targets, reduction="none").to(torch.float64)


def score_cached_chunk(
    model: torch.nn.Module,
    *,
    context_ids: torch.Tensor,
    prefix_ids: torch.Tensor,
    target_ids: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    """Score one chunk with a prefill and ``T - 1`` cached decode steps.

    ``context_ids`` are unscored template tokens (may be empty); ``prefix_ids``
    are corpus tokens scored by the prefill logits only; ``target_ids`` are the
    ``T`` corpus tokens scored through the cache.
    """
    for name, ids in (
        ("context_ids", context_ids),
        ("prefix_ids", prefix_ids),
        ("target_ids", target_ids),
    ):
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be a 1-D integer tensor.")
    if prefix_ids.numel() == 0 or target_ids.numel() == 0:
        raise ValueError("prefix_ids and target_ids must be nonempty.")

    input_ids = torch.cat((context_ids, prefix_ids)).to(device).unsqueeze(0)
    prefill_length = input_ids.shape[1]
    targets = target_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(prefill_length, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        logits = _logits(outputs, prefill_length)[0]
        cache = _cache(outputs, prefill_length)

        # Prefill-only score: corpus prefix tokens that have a predecessor.
        first_scored = max(int(context_ids.numel()), 1)
        prefill_nll = _token_nll(
            logits[first_scored - 1 : prefill_length - 1],
            input_ids[0, first_scored:],
        )
        token_nll = [_token_nll(logits[-1:], targets[:1])]
        for step in range(int(targets.numel()) - 1):
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=-1
            )
            outputs = model(
                input_ids=targets[step : step + 1].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=torch.full(
                    (1, 1), prefill_length + step, device=device, dtype=torch.long
                ),
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            step_logits = _logits(outputs, 1)[0]
            cache = _cache(outputs, prefill_length + step + 1)
            token_nll.append(_token_nll(step_logits, targets[step + 1 : step + 2]))
    per_token = torch.cat(token_nll).cpu()
    if per_token.numel() != targets.numel() or not bool(
        torch.isfinite(per_token).all()
    ):
        raise RuntimeError("Cached NLL produced a missing or non-finite target score.")
    return {
        "target_tokens": int(per_token.numel()),
        "target_nll_sum": float(per_token.sum()),
        "target_nll": [float(value) for value in per_token],
        "prefill_tokens": int(prefill_nll.numel()),
        "prefill_nll_sum": float(prefill_nll.sum()),
        "prefill_length": int(prefill_length),
    }


def chat_prefix_ids(processor_or_tokenizer: Any, tokenizer: Any) -> torch.Tensor:
    """Render and tokenize the chat prefix once; callers reuse the result."""
    from tico.quantization.wrapq.utils.metrics import _render_chat_prefix

    rendered = _render_chat_prefix(processor_or_tokenizer, tokenizer)
    encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
        raise ValueError("Chat prefix must tokenize to a nonempty [1, N] tensor.")
    return ids[0].to(torch.long)


def validate_context_budget(
    context_tokens: int, protocol: CachedNLLProtocol, max_seq_len: Optional[int]
) -> None:
    """Reject a chunk whose whole length exceeds the evaluation context.

    ``H + P + T <= M`` is enforced on the actual chat-prefix token count. The
    peak cached length is ``H + P + T - 1`` (the last target is never fed), but
    the pre-existing ``P + T`` whole-chunk contract is kept and only the missing
    ``H`` is added; the one-token slack is intentional, not an oversight.
    """
    if max_seq_len is None:
        return
    limit = int(max_seq_len)
    required = int(context_tokens) + protocol.span
    if required > limit:
        raise ValueError(
            "cached NLL chunk exceeds the evaluation context: "
            f"chat_prefix_tokens H={int(context_tokens)} + prefix_tokens P="
            f"{protocol.prefix_tokens} + target_tokens T={protocol.target_tokens} "
            f"= {required} > max_seq_len M={limit}. Reduce prefix_tokens or "
            "target_tokens, or evaluate with a larger context."
        )


def evaluate_cached_nll(
    model: torch.nn.Module,
    token_stream: torch.Tensor,
    protocol: CachedNLLProtocol,
    *,
    device: torch.device | str | None = None,
    context_ids: Optional[torch.Tensor] = None,
    max_seq_len: Optional[int] = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Score fixed chunks of ``token_stream`` (1-D) with real cached decode.

    ``context_ids`` are the exact chat-prefix tokens used for every chunk; the
    same tensor is checked against ``max_seq_len`` and fed to the model, so the
    budget check and the scoring can never disagree.
    """
    if token_stream.ndim != 1:
        raise ValueError("token_stream must be a 1-D token tensor.")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    context = (
        torch.zeros((0,), dtype=torch.long) if context_ids is None else context_ids
    )
    if context.ndim != 1:
        raise ValueError("context_ids must be a 1-D token tensor.")
    # Both checks run before the first model forward.
    validate_context_budget(int(context.numel()), protocol, max_seq_len)
    ranges = chunk_token_ranges(int(token_stream.numel()), protocol)
    model.eval()
    chunks = []
    target_nll_sum = 0.0
    target_tokens = 0
    prefill_nll_sum = 0.0
    prefill_tokens = 0
    for index, (begin, end) in enumerate(
        tqdm.tqdm(ranges, desc="cached NLL", disable=not show_progress)
    ):
        span = token_stream[begin:end].to(torch.long)
        record = score_cached_chunk(
            model,
            context_ids=context,
            prefix_ids=span[: protocol.prefix_tokens],
            target_ids=span[protocol.prefix_tokens :],
            device=device,
        )
        record.update(
            index=index,
            begin=int(begin),
            end=int(end),
            mean_target_nll=record["target_nll_sum"] / record["target_tokens"],
        )
        chunks.append(record)
        target_nll_sum += record["target_nll_sum"]
        target_tokens += record["target_tokens"]
        prefill_nll_sum += record["prefill_nll_sum"]
        prefill_tokens += record["prefill_tokens"]
    if target_tokens == 0:
        raise ValueError("No cached target tokens were scored.")
    mean_nll = target_nll_sum / target_tokens
    prefill_mean = prefill_nll_sum / prefill_tokens if prefill_tokens else None
    return {
        "protocol": asdict(protocol),
        "max_seq_len": None if max_seq_len is None else int(max_seq_len),
        "context_tokens": int(context.numel()),
        "context_token_ids": [int(value) for value in context],
        "corpus_tokens": int(token_stream.numel()),
        "scored_tokens": int(target_tokens),
        "mean_nll": float(mean_nll),
        "ppl": float(math.exp(mean_nll)),
        "prefill_only": {
            "scored_tokens": int(prefill_tokens),
            "mean_nll": None if prefill_mean is None else float(prefill_mean),
            "ppl": None if prefill_mean is None else float(math.exp(prefill_mean)),
        },
        "chunks": chunks,
    }


def normalize_text_examples(dataset: Iterable[Any], adapter: Any) -> list[str]:
    """Apply the registered dataset adapter once and collect the ``text`` fields.

    The dataset is consumed in a single pass (streaming sources cannot be
    re-iterated). Every normalized item must provide a string ``text``; a
    missing or non-string field is a schema error, never an empty substitute.
    """
    if not callable(adapter):
        raise TypeError(
            "Cached NLL requires the registered dataset adapter returned by "
            f"get_dataset(); got {type(adapter).__name__}."
        )
    texts: list[str] = []
    for index, example in enumerate(dataset):
        item = adapter(example)
        if not isinstance(item, Mapping) or "text" not in item:
            raise ValueError(
                f"Dataset adapter output for example {index} lacks the 'text' "
                f"field required by cached NLL (got {type(item).__name__})."
            )
        text = item["text"]
        if not isinstance(text, str):
            raise ValueError(
                f"Dataset adapter output for example {index} must provide a string "
                f"'text', got {type(text).__name__}."
            )
        texts.append(text)
    return texts


def tokenize_text_stream(tokenizer: Any, texts: Iterable[str]) -> torch.Tensor:
    """Tokenize a corpus once into a 1-D stream, as the PPL helpers do.

    Rows are joined with ``"\\n\\n"`` exactly as given: empty WikiText rows and
    surrounding whitespace are preserved so the token stream matches the
    existing perplexity helpers. A corpus without any non-whitespace text is
    rejected before the tokenizer or the model runs.
    """
    rows = [str(text) for text in texts]
    if not any(row.strip() for row in rows):
        raise ValueError(
            "The dataset contains no non-whitespace text after adapter "
            "normalization; refusing to score an empty corpus."
        )
    joined = "\n\n".join(rows)
    encoded = tokenizer(joined, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"Expected tokenized shape [1, sequence], got {ids.shape}.")
    return ids[0].to(torch.long)


def evaluate_text_cached_nll(
    *,
    model: torch.nn.Module,
    processor_or_tokenizer: Any,
    dataset_name: str,
    split: str,
    protocol: CachedNLLProtocol,
    device: torch.device | str,
    max_seq_len: Optional[int] = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Load a registered evaluation text dataset and score it with cached decode.

    ``max_seq_len`` is the resolved evaluation context; the chat prefix is
    rendered and tokenized once, and that tensor is used both for the
    ``H + P + T <= M`` check and for scoring.
    """
    from tico.quantization.evaluation.vlm_eval_utils import get_dataset
    from tico.quantization.recipes.data.dataset_usage import EVALUATION_ROLE

    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    context = (
        chat_prefix_ids(processor_or_tokenizer, tokenizer)
        if protocol.chat_prefix
        else None
    )
    # Fail on the budget before loading or tokenizing the corpus.
    validate_context_budget(
        0 if context is None else int(context.numel()), protocol, max_seq_len
    )
    dataset, adapter = get_dataset(
        dataset_name, role=EVALUATION_ROLE, split=split, n=-1
    )
    stream = tokenize_text_stream(tokenizer, normalize_text_examples(dataset, adapter))
    results = evaluate_cached_nll(
        model,
        stream,
        protocol,
        device=device,
        context_ids=context,
        max_seq_len=max_seq_len,
        show_progress=show_progress,
    )
    results["dataset"] = {"name": dataset_name, "split": split}
    return results


def print_cached_nll_results(results: Mapping[str, Any]) -> None:
    prefill = results["prefill_only"]
    print("\n=== Cached decode NLL ===")
    print(
        f"cached targets: {results['scored_tokens']} tokens over "
        f"{len(results['chunks'])} chunks | mean NLL {results['mean_nll']:.6f} | "
        f"PPL {results['ppl']:.4f}"
    )
    if prefill["mean_nll"] is not None:
        print(
            f"prefill-only: {prefill['scored_tokens']} tokens | "
            f"mean NLL {prefill['mean_nll']:.6f} | PPL {prefill['ppl']:.4f}"
        )
