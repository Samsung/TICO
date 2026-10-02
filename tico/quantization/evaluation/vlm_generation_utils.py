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

"""Generation budget validation and opt-in, single-example VLM diagnostics."""

import hashlib
import json
from numbers import Integral
from typing import Any, Mapping

import torch


def positive_int(value: Any, name: str) -> int:
    """Validate an integer without silently accepting booleans or fractions."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def resolve_generation_input_budget(
    max_seq_len: int | None,
    max_new_tokens: int,
    input_max_seq_len: int | None = None,
) -> int | None:
    """Resolve the input cap, optionally independent of generation length.

    ``max_seq_len`` remains the total input-plus-output budget. An explicit
    input cap is never silently reduced; an incompatible combination fails.
    Without it, the historical ``max_seq_len - max_new_tokens`` rule is used.
    """
    max_new_tokens = positive_int(max_new_tokens, "max_new_tokens")
    if max_seq_len is not None:
        max_seq_len = positive_int(max_seq_len, "max_seq_len")
    if input_max_seq_len is not None:
        input_max_seq_len = positive_int(input_max_seq_len, "input_max_seq_len")
        if (
            max_seq_len is not None
            and input_max_seq_len + max_new_tokens > max_seq_len
        ):
            raise ValueError(
                "input_max_seq_len + max_new_tokens must not exceed max_seq_len: "
                f"{input_max_seq_len} + {max_new_tokens} > {max_seq_len}."
            )
        return input_max_seq_len
    if max_seq_len is None:
        return None
    remaining = max_seq_len - max_new_tokens
    if remaining <= 0:
        raise ValueError(
            "Generation token budget must be smaller than max_seq_len: "
            f"max_seq_len={max_seq_len}, max_new_tokens={max_new_tokens}."
        )
    return remaining


def record_generation_inputs(
    diagnostics: dict[str, Any] | None,
    inputs: Mapping[str, Any],
    *,
    input_max_seq_len: int | None,
) -> None:
    """Hash tensor inputs before generation, only when explicitly requested.

    Includes shape, dtype, key and exact bytes (also for bfloat16). This does
    not claim equivalence for non-tensor inputs or model weights. Fingerprinting
    can copy tensors to CPU, so it is deliberately disabled on the normal path.
    """
    if diagnostics is None:
        return
    diagnostics.clear()
    fingerprints: dict[str, Any] = {}
    for key in sorted(inputs):
        value = inputs[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        fingerprints[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    combined = json.dumps(fingerprints, sort_keys=True, separators=(",", ":"))
    diagnostics.update(
        {
            "input_max_seq_len": input_max_seq_len,
            "input_tokens": int(inputs["input_ids"].shape[-1]),
            "input_tensors": fingerprints,
            "tensor_inputs_sha256": hashlib.sha256(combined.encode()).hexdigest(),
        }
    )
    for key in ("image_grid_thw", "video_grid_thw"):
        value = inputs.get(key)
        if torch.is_tensor(value):
            diagnostics[key] = value.detach().cpu().tolist()


def _generation_eos_ids(model: Any) -> list[int] | None:
    # Do not use tokenizer.eos_token_id: it is not necessarily the EOS used by
    # model.generate(). None means we cannot infer the generation stopping rule.
    config = getattr(model, "generation_config", None)
    if config is None:
        config = getattr(model, "config", None)
    value = getattr(config, "eos_token_id", None)
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    values = value if isinstance(value, (list, tuple)) else [value]
    if all(isinstance(v, Integral) and not isinstance(v, bool) for v in values):
        return [int(v) for v in values]
    return None


def record_generation_output(
    diagnostics: dict[str, Any] | None,
    gen_ids: torch.Tensor,
    *,
    model: Any,
    max_new_tokens: int,
) -> None:
    """Record observed length and EOS without calling every limit hit truncation."""
    if diagnostics is None:
        return
    ids = [int(token_id) for token_id in gen_ids.detach().cpu().tolist()]
    eos_ids = _generation_eos_ids(model)
    eos_position = next(
        (i for i, token_id in enumerate(ids) if eos_ids and token_id in eos_ids),
        None,
    )
    # Padding after EOS must not inflate the generated-token count.
    actual_ids = ids if eos_position is None else ids[: eos_position + 1]
    reached_limit = len(actual_ids) >= max_new_tokens
    if eos_position is not None:
        stop_reason = "eos"
        length_limited: bool | None = False
    elif eos_ids is None:
        stop_reason = "unknown"
        length_limited = None
    else:
        stop_reason = "length" if reached_limit else "other"
        length_limited = reached_limit
    diagnostics.update(
        {
            "generated_tokens": len(actual_ids),
            "generated_token_ids": actual_ids,
            "max_new_tokens": max_new_tokens,
            "eos_token_ids": eos_ids,
            "eos_observed": eos_position is not None,
            "reached_max_new_tokens": reached_limit,
            "length_limited": length_limited,
            "stop_reason": stop_reason,
        }
    )
