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

import argparse
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization import prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)


PaddingSide = Literal["left", "right"]
PromptInput = Union[str, Sequence[str]]


@dataclass
class LayerCache:
    past_k: torch.Tensor
    past_v: torch.Tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Static-shape Llama layer runtime with prefill/decode wrappers."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Maykeye/TinyLLama-v0",
        help="HF model name or local model path.",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=256,
        help="Static sequence length used by both prefill and decode runtime inputs.",
    )
    parser.add_argument(
        "--padding-side",
        type=str,
        choices=("left", "right"),
        default="right",
        help="Padding direction used for static prefill inputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device, e.g. cpu or cuda.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The capital of France is",
        help="Prompt used for verification and greedy generation.",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=6,
        help="Number of decode steps for reference verification.",
    )
    parser.add_argument(
        "--gen-steps",
        type=int,
        default=16,
        help="Maximum number of new tokens for greedy generation.",
    )
    return parser.parse_args()


def _clone_quant_layer(layer: nn.Module) -> nn.Module:
    """
    Build a quantized decoder layer wrapper.

    Export-specific specialization is handled through export adapters built
    from the wrapped layer.
    """
    return prepare(layer, PTQConfig())


def _normalize_valid_token_mask(
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    *,
    pad_token_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Build a boolean mask where True means a real prompt token.

    If `attention_mask` is omitted, padding is inferred from `pad_token_id` when
    available. When `pad_token_id` can also appear as a real token, callers
    should pass `attention_mask` explicitly.
    """
    if attention_mask is None:
        if pad_token_id is None:
            valid_token_mask = torch.ones(
                input_ids.shape,
                device=device,
                dtype=torch.bool,
            )
        else:
            valid_token_mask = input_ids.to(device).ne(int(pad_token_id))
    else:
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "attention_mask must have the same shape as input_ids. "
                f"Got attention_mask={tuple(attention_mask.shape)}, "
                f"input_ids={tuple(input_ids.shape)}."
            )
        valid_token_mask = attention_mask.to(device=device).bool()

    if torch.any(valid_token_mask.sum(dim=1) == 0):
        raise ValueError("Each batch row must contain at least one real token.")

    return valid_token_mask


def _validate_padding_layout(
    valid_token_mask: torch.Tensor,
    padding_side: PaddingSide,
) -> None:
    """
    Validate that each row uses contiguous left or right padding.
    """
    batch_size, seq_len = valid_token_mask.shape
    valid_lengths = valid_token_mask.sum(dim=1)
    positions = torch.arange(seq_len, device=valid_token_mask.device).unsqueeze(0)

    if padding_side == "right":
        expected = positions < valid_lengths.unsqueeze(1)
    else:
        expected = positions >= (seq_len - valid_lengths).unsqueeze(1)

    if not torch.equal(valid_token_mask, expected):
        raise ValueError(
            "Input padding layout does not match padding_side. "
            f"Expected contiguous {padding_side} padding for shape "
            f"(B={batch_size}, S={seq_len})."
        )


def _build_position_ids_from_valid_token_mask(
    valid_token_mask: torch.Tensor,
) -> torch.LongTensor:
    """
    Build compact absolute position IDs for padded prefill inputs.

    Real tokens receive positions 0..valid_length-1 independently of whether
    padding is placed on the left or on the right. Padding slots receive position
    0 because their logits and K/V outputs are ignored by the runtime.
    """
    position_ids = valid_token_mask.to(torch.long).cumsum(dim=1) - 1
    position_ids = torch.clamp(position_ids, min=0)
    position_ids = position_ids.masked_fill(~valid_token_mask, 0)
    return position_ids


def _last_valid_token_indices(valid_token_mask: torch.Tensor) -> torch.LongTensor:
    """
    Return the input index of the last real token for each batch row.
    """
    batch_size, seq_len = valid_token_mask.shape
    positions = torch.arange(seq_len, device=valid_token_mask.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    return positions.masked_fill(~valid_token_mask, 0).max(dim=1).values


def _gather_last_token_logits(
    logits: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Gather logits at the last real token index of each padded prefill row.
    """
    last_indices = _last_valid_token_indices(valid_token_mask).to(logits.device)
    batch_indices = torch.arange(logits.size(0), device=logits.device)
    return logits[batch_indices, last_indices, :]


def _gather_rope_by_position_ids(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    position_ids: torch.LongTensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gather RoPE tensors for per-token absolute position IDs.

    Args:
        rope_cos: RoPE cosine table with shape `(1, max_seq, head_dim)`.
        rope_sin: RoPE sine table with shape `(1, max_seq, head_dim)`.
        position_ids: Absolute position IDs with shape `(B, S)`.
        device: Destination device.
        dtype: Destination dtype.

    Returns:
        A tuple `(cos, sin)` with shape `(B, S, head_dim)`.
    """
    position_ids = position_ids.to(device=device, dtype=torch.long)
    batch_size, seq_len = position_ids.shape
    flat_positions = position_ids.reshape(-1)

    cos_table = rope_cos[0].to(device=device, dtype=dtype)
    sin_table = rope_sin[0].to(device=device, dtype=dtype)

    cos = cos_table.index_select(0, flat_positions).reshape(batch_size, seq_len, -1)
    sin = sin_table.index_select(0, flat_positions).reshape(batch_size, seq_len, -1)
    return cos, sin


def _build_rope_templates_from_config(
    config,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build full RoPE tables using the same simplified logic as the wrappers.

    Output shapes:
        cos: (1, max_seq, head_dim)
        sin: (1, max_seq, head_dim)
    """
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )

    rope_params = getattr(config, "rope_parameters", None)
    if (
        rope_params is not None
        and isinstance(rope_params, dict)
        and "rope_theta" in rope_params
    ):
        base = float(rope_params["rope_theta"])
    else:
        base = float(getattr(config, "rope_theta", 10000.0))

    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    pos = torch.arange(max_seq, dtype=torch.float32, device=device)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = emb.cos()
    sin = emb.sin()

    half_dim = head_dim // 2
    sin[..., :half_dim] = -sin[..., :half_dim]

    cos = cos.unsqueeze(0).to(dtype=dtype)
    sin = sin.unsqueeze(0).to(dtype=dtype)
    return cos, sin


def _slice_rope(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    position: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Slice one-step RoPE tensors for decode.

    Output shapes:
        cos: (B, 1, head_dim)
        sin: (B, 1, head_dim)
    """
    cos = rope_cos[:, position : position + 1, :].to(device=device, dtype=dtype)
    sin = rope_sin[:, position : position + 1, :].to(device=device, dtype=dtype)

    if batch_size != 1:
        cos = cos.expand(batch_size, -1, -1).contiguous()
        sin = sin.expand(batch_size, -1, -1).contiguous()

    return cos, sin


def _slice_prefill_rope(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Slice prefill RoPE tensors starting from position 0.

    Returns:
        cos: Tensor with shape (B, seq_len, head_dim).
        sin: Tensor with shape (B, seq_len, head_dim).
    """
    cos = rope_cos[:, :seq_len, :].to(device=device, dtype=dtype)
    sin = rope_sin[:, :seq_len, :].to(device=device, dtype=dtype)

    if batch_size != 1:
        cos = cos.expand(batch_size, -1, -1).contiguous()
        sin = sin.expand(batch_size, -1, -1).contiguous()

    return cos, sin


def _build_decode_attention_mask(
    batch_size: int,
    past_lengths: torch.LongTensor,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    """
    Build a fully static decode mask.

    Layout assumption:
        - past KV is packed into the first `past_lengths[b]` slots per batch row
        - padded past slots are masked
        - current token is appended internally by the attention module at the
          last slot

    Returned shape:
        (B, 1, max_seq)

    Valid columns per row:
        [0, 1, ..., past_lengths[b] - 1, max_seq - 1]
    Masked columns per row:
        [past_lengths[b], ..., max_seq - 2]
    """
    past_lengths = past_lengths.to(device=device, dtype=torch.long).reshape(-1)
    if past_lengths.numel() != batch_size:
        raise ValueError(
            f"Expected {batch_size} past lengths, got {past_lengths.numel()}."
        )

    mask = torch.full((batch_size, 1, max_seq), mask_value, device=device, dtype=dtype)
    past_positions = torch.arange(max_seq - 1, device=device).unsqueeze(0)
    valid_past = past_positions < past_lengths.unsqueeze(1)

    mask[:, 0, : max_seq - 1] = mask[:, 0, : max_seq - 1].masked_fill(
        valid_past,
        0.0,
    )
    mask[:, :, max_seq - 1] = 0.0
    return mask


def _build_prefill_attention_mask(
    valid_token_mask: torch.Tensor,
    position_ids: torch.LongTensor,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    """
    Build an additive causal prefill mask for padded static inputs.

    Real tokens may attend only to real keys whose compact position is less than
    or equal to the query token position. Padding keys are always masked. Padding
    queries are also masked because their outputs and K/V entries are ignored by
    the runtime.

    Args:
        valid_token_mask: Boolean tensor with shape `(B, S)`. True means a real
            token and False means padding.
        position_ids: Compact absolute position IDs with shape `(B, S)`.
        device: Device where the mask should be created.
        dtype: Output mask dtype.
        mask_value: Additive value used for masked attention entries.

    Returns:
        Additive attention mask with shape `(B, S, S)`.
    """
    valid_token_mask = valid_token_mask.to(device=device).bool()
    position_ids = position_ids.to(device=device, dtype=torch.long)

    if valid_token_mask.dim() != 2:
        raise ValueError(
            f"valid_token_mask must have shape (B, S), got {tuple(valid_token_mask.shape)}."
        )
    if tuple(position_ids.shape) != tuple(valid_token_mask.shape):
        raise ValueError(
            "position_ids must have the same shape as valid_token_mask. "
            f"Got position_ids={tuple(position_ids.shape)}, "
            f"valid_token_mask={tuple(valid_token_mask.shape)}."
        )

    batch_size, seq_len = valid_token_mask.shape

    query_valid = valid_token_mask.unsqueeze(2)
    key_valid = valid_token_mask.unsqueeze(1)
    causal = position_ids.unsqueeze(2) >= position_ids.unsqueeze(1)

    allowed = query_valid & key_valid & causal

    mask = torch.full(
        (batch_size, seq_len, seq_len),
        mask_value,
        device=device,
        dtype=dtype,
    )
    return mask.masked_fill(allowed, 0.0)


def _write_decode_delta_to_cache(
    cache: LayerCache,
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    write_positions: torch.LongTensor,
) -> None:
    """
    Write one-token decode K/V deltas into per-row cache positions.
    """
    positions = write_positions.to(device=new_k.device, dtype=torch.long).reshape(-1)
    for batch_idx, pos_tensor in enumerate(positions):
        pos = int(pos_tensor.item())
        if pos < 0 or pos >= cache.past_k.size(2):
            raise RuntimeError(
                "Decode cache write position is outside the static past buffer. "
                f"position={pos}, capacity={cache.past_k.size(2)}."
            )
        cache.past_k[batch_idx, :, pos : pos + 1, :] = new_k[batch_idx]
        cache.past_v[batch_idx, :, pos : pos + 1, :] = new_v[batch_idx]


class StaticLlamaLayerRuntime:
    """
    Hybrid runtime that uses:
        - wrapped decoder layers for prefill and decode
        - original embedding / final norm / lm_head on CPU or a chosen device

    This runtime enforces static decode shapes:
        hidden_states:       (B, 1, D)
        attention_mask:      (B, 1, max_seq)
        past_key_value:      (B, n_kv, max_seq - 1, head_dim)
        position_embeddings: (B, 1, head_dim)

    Prefill is run with static sequence length `max_seq`. Shorter prompt inputs
    are padded to `(B, max_seq)` before the wrapped prefill layers are invoked.
    Both left and right padding are accepted. The runtime packs only real prompt
    K/V entries into the external decode cache and ignores K/V entries produced
    for padding slots.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_seq: int,
        device: str = "cpu",
        layers: Optional[Sequence[nn.Module]] = None,
        padding_side: PaddingSide = "right",
    ):
        if padding_side not in ("left", "right"):
            raise ValueError(f"Unsupported padding_side: {padding_side!r}")
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.device = torch.device(device)
        self.padding_side = padding_side

        self.embed_tokens = self.model.model.embed_tokens
        self.final_norm = self.model.model.norm
        self.lm_head = self.model.lm_head
        self.layers_ref = self.model.model.layers

        if layers is None:
            self.layers = nn.ModuleList(
                [_clone_quant_layer(layer) for layer in self.layers_ref]
            ).to(self.device)
        else:
            self.layers = nn.ModuleList(layers).to(self.device)

        for layer in self.layers:
            assert hasattr(layer, "wrapped")
            assert isinstance(layer.wrapped, QuantLlamaDecoderLayer)

        self.prefill_layers = nn.ModuleList(
            [
                layer.wrapped.as_export_module("prefill", return_kv=True)
                for layer in self.layers
            ]
        ).to(self.device)
        self.decode_layers = nn.ModuleList(
            [
                layer.wrapped.as_export_module("decode", return_kv=True)
                for layer in self.layers
            ]
        ).to(self.device)

        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", None) or (
            self.hidden_size // self.config.num_attention_heads
        )

        self.rope_cos, self.rope_sin = _build_rope_templates_from_config(
            self.config,
            max_seq=self.max_seq,
            device=self.device,
            dtype=torch.float32,
        )

        self.layer_caches: List[LayerCache] = []
        self.past_lengths: Optional[torch.LongTensor] = None
        self.past_len = 0

    def reset_cache(self) -> None:
        """
        Reset all runtime KV caches.
        """
        self.layer_caches = []
        self.past_lengths = None
        self.past_len = 0

    def _allocate_empty_cache(
        self, batch_size: int, dtype: torch.dtype
    ) -> List[LayerCache]:
        """
        Allocate external static KV buffers for all layers.

        The runtime stores only past tokens in these buffers.
        The current token is always produced as a delta by the decode wrapper.
        """
        caches = []
        for _ in range(self.num_hidden_layers):
            past_k = torch.zeros(
                batch_size,
                self.num_kv_heads,
                self.max_seq - 1,
                self.head_dim,
                device=self.device,
                dtype=dtype,
            )
            past_v = torch.zeros_like(past_k)
            caches.append(LayerCache(past_k=past_k, past_v=past_v))
        return caches

    def _get_pad_token_id(self) -> int:
        """
        Return a pad token ID for constructing static prompt batches.
        """
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            return int(pad_token_id)

        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None:
            return int(eos_token_id)

        raise ValueError(
            "The tokenizer must define either pad_token_id or eos_token_id."
        )

    def _prepare_prompt_batch(
        self,
        prompt: PromptInput,
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Tokenize prompts and pad them to the static prefill length `max_seq`.

        Returns:
            A tuple `(input_ids, attention_mask)` with shape `(B, max_seq)`.
        """
        side = self.padding_side
        texts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not texts:
            raise ValueError("At least one prompt is required.")

        encoded = self.tokenizer(
            texts,
            return_tensors=None,
            add_special_tokens=True,
            padding=False,
        )
        token_rows = [
            torch.tensor(row, dtype=torch.long) for row in encoded["input_ids"]
        ]
        max_prompt_len = max(int(row.numel()) for row in token_rows)
        target_len = self.max_seq

        if target_len <= 1:
            raise ValueError(f"max_seq must be greater than 1, got {target_len}.")
        if max_prompt_len >= target_len:
            raise ValueError(
                "Prompt length must be smaller than max_seq so that at least "
                "one decode slot remains. "
                f"prompt_len={max_prompt_len}, max_seq={target_len}."
            )

        pad_token_id = self._get_pad_token_id()
        padded_rows = []
        mask_rows = []
        for row in token_rows:
            prompt_len = int(row.numel())
            pad_len = target_len - prompt_len
            pad = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((pad_len,), dtype=torch.long)
            token_mask = torch.ones((prompt_len,), dtype=torch.long)

            if side == "left":
                padded = torch.cat([pad, row], dim=0)
                mask = torch.cat([pad_mask, token_mask], dim=0)
            else:
                padded = torch.cat([row, pad], dim=0)
                mask = torch.cat([token_mask, pad_mask], dim=0)

            padded_rows.append(padded)
            mask_rows.append(mask)

        input_ids = torch.stack(padded_rows, dim=0).to(self.device)
        attention_mask = torch.stack(mask_rows, dim=0).to(self.device)
        return input_ids, attention_mask

    def _pad_prefill_tensors_to_max_seq(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.LongTensor, Optional[torch.Tensor]]:
        """
        Pad caller-provided prefill tensors to the static length `max_seq`.

        If `attention_mask` is omitted and the input is shorter than `max_seq`,
        all provided tokens are treated as real prompt tokens and only the newly
        added slots are marked as padding.
        """
        side = self.padding_side
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq:
            raise ValueError(
                f"Prefill input length must be <= max_seq. "
                f"Got seq_len={seq_len}, max_seq={self.max_seq}."
            )
        if seq_len == self.max_seq:
            if attention_mask is None:
                return input_ids, None
            return input_ids, attention_mask.to(device=self.device)

        pad_len = self.max_seq - seq_len
        pad_token_id = self._get_pad_token_id()
        pad_ids = torch.full(
            (batch_size, pad_len),
            pad_token_id,
            device=self.device,
            dtype=torch.long,
        )
        pad_mask = torch.zeros(
            (batch_size, pad_len),
            device=self.device,
            dtype=torch.long,
        )

        if attention_mask is None:
            base_mask = torch.ones(
                (batch_size, seq_len),
                device=self.device,
                dtype=torch.long,
            )
        else:
            if tuple(attention_mask.shape) != (batch_size, seq_len):
                raise ValueError(
                    "attention_mask must have the same shape as input_ids before "
                    "static prefill padding. "
                    f"Got attention_mask={tuple(attention_mask.shape)}, "
                    f"input_ids={(batch_size, seq_len)}."
                )
            base_mask = attention_mask.to(device=self.device, dtype=torch.long)

        if side == "left":
            input_ids = torch.cat([pad_ids, input_ids], dim=1)
            attention_mask = torch.cat([pad_mask, base_mask], dim=1)
        else:
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([base_mask, pad_mask], dim=1)

        return input_ids, attention_mask

    @staticmethod
    def _compact_input_rows(
        input_ids: torch.LongTensor,
        valid_token_mask: torch.Tensor,
    ) -> List[torch.LongTensor]:
        """
        Extract unpadded token rows from a padded batch.
        """
        valid_token_mask = valid_token_mask.to(device=input_ids.device).bool()
        rows = []
        for batch_idx in range(input_ids.size(0)):
            rows.append(input_ids[batch_idx, valid_token_mask[batch_idx]].clone())
        return rows

    def _pad_token_rows(
        self,
        rows: Sequence[torch.LongTensor],
    ) -> torch.LongTensor:
        """
        Pad a list of generated token rows into one tensor.
        """
        side = self.padding_side
        max_len = max(int(row.numel()) for row in rows)
        pad_token_id = self._get_pad_token_id()
        padded_rows = []

        for row in rows:
            row = row.to(device=self.device, dtype=torch.long)
            pad_len = max_len - int(row.numel())
            pad = torch.full(
                (pad_len,),
                pad_token_id,
                device=self.device,
                dtype=torch.long,
            )
            if side == "left":
                padded_rows.append(torch.cat([pad, row], dim=0))
            else:
                padded_rows.append(torch.cat([row, pad], dim=0))

        return torch.stack(padded_rows, dim=0)

    @torch.no_grad()
    def _reference_last_logits_from_sequences(
        self,
        rows: Sequence[torch.LongTensor],
    ) -> torch.Tensor:
        """
        Run the original model on compact, unpadded token rows.
        """
        logits_rows = []
        for row in rows:
            if row.numel() == 0:
                raise ValueError("Reference rows must not be empty.")
            ref_out = self.model(input_ids=row.to(self.device).unsqueeze(0))
            logits_rows.append(ref_out.logits[:, -1, :])
        return torch.cat(logits_rows, dim=0)

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run the prompt through all prefill layers and initialize decode caches.

        Args:
            input_ids: Token IDs with shape `(B, S)`. If `S < max_seq`, this
                method pads the tensor to `(B, max_seq)` before running prefill.
            attention_mask: Optional real-token mask with shape `(B, S)`, where 1
                means a real token and 0 means padding. If omitted for a short
                input, all provided tokens are treated as real tokens.

        Returns:
            Logits for the last real token of each row with shape
            `(B, vocab_size)`.
        """
        assert (
            input_ids.dim() == 2
        ), f"Expected input_ids as (B, S), got {tuple(input_ids.shape)}"
        side = self.padding_side
        input_ids, attention_mask = self._pad_prefill_tensors_to_max_seq(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        batch_size, static_seq_len = input_ids.shape
        if static_seq_len != self.max_seq:
            raise RuntimeError(
                "Static prefill input length must be exactly max_seq after padding. "
                f"Got static_seq_len={static_seq_len}, max_seq={self.max_seq}."
            )

        valid_token_mask = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        _validate_padding_layout(valid_token_mask, side)

        valid_lengths = valid_token_mask.sum(dim=1).to(
            device=self.device, dtype=torch.long
        )
        if torch.any(valid_lengths >= self.max_seq):
            raise ValueError(
                "Each prompt must leave at least one decode slot. "
                f"valid_lengths={valid_lengths.tolist()}, max_seq={self.max_seq}."
            )

        position_ids = _build_position_ids_from_valid_token_mask(valid_token_mask)
        if int(position_ids.max().item()) >= self.max_seq:
            raise ValueError(
                "Prefill position IDs must be smaller than max_seq. "
                f"max_position_id={int(position_ids.max().item())}, max_seq={self.max_seq}."
            )

        hidden_states = self.embed_tokens(input_ids.to(self.device))
        runtime_dtype = hidden_states.dtype

        self.layer_caches = self._allocate_empty_cache(batch_size, runtime_dtype)

        prefill_mask = _build_prefill_attention_mask(
            valid_token_mask=valid_token_mask,
            position_ids=position_ids,
            device=self.device,
            dtype=hidden_states.dtype,
        )
        position_embeddings = _gather_rope_by_position_ids(
            self.rope_cos,
            self.rope_sin,
            position_ids=position_ids,
            device=self.device,
            dtype=hidden_states.dtype,
        )

        for layer_idx, layer in enumerate(self.prefill_layers):
            out = layer(
                hidden_states=hidden_states,
                attention_mask=prefill_mask,
                position_embeddings=position_embeddings,
            )

            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "Expected prefill adapter output as (hidden_states, new_k, new_v)."
                )

            hidden_states, new_k, new_v = out

            assert new_k.size(2) == static_seq_len
            assert new_v.size(2) == static_seq_len

            for batch_idx in range(batch_size):
                token_indices = torch.nonzero(
                    valid_token_mask[batch_idx],
                    as_tuple=False,
                ).flatten()
                valid_len = int(valid_lengths[batch_idx].item())
                self.layer_caches[layer_idx].past_k[
                    batch_idx, :, :valid_len, :
                ] = new_k[batch_idx, :, token_indices, :]
                self.layer_caches[layer_idx].past_v[
                    batch_idx, :, :valid_len, :
                ] = new_v[batch_idx, :, token_indices, :]

        self.past_lengths = valid_lengths.clone()
        self.past_len = int(valid_lengths.max().item())

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits_last = _gather_last_token_logits(logits, valid_token_mask)
        return logits_last

    @torch.no_grad()
    def decode_one(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Run one decode step with strict static input shapes.

        Input:
            input_ids: (B, 1)

        Returns:
            logits_last: (B, vocab_size)
        """
        assert (
            input_ids.dim() == 2 and input_ids.size(1) == 1
        ), f"Decode expects input_ids as (B, 1), got {tuple(input_ids.shape)}"
        assert (
            len(self.layer_caches) == self.num_hidden_layers
        ), "Caches are not initialized. Call prefill() first."
        if self.past_lengths is None:
            raise RuntimeError(
                "Past lengths are not initialized. Call prefill() first."
            )

        batch_size = input_ids.size(0)
        if self.past_lengths.numel() != batch_size:
            raise ValueError(
                "Decode batch size must match the initialized cache batch size. "
                f"input batch={batch_size}, cache batch={self.past_lengths.numel()}."
            )
        if torch.any(self.past_lengths >= self.max_seq - 1):
            raise RuntimeError(
                "Decode cache is full. Cannot write the current token into the "
                f"static past buffer. past_lengths={self.past_lengths.tolist()}, "
                f"capacity={self.max_seq - 1}."
            )

        hidden_states = self.embed_tokens(input_ids.to(self.device))
        write_positions = self.past_lengths.clone()

        attention_mask = _build_decode_attention_mask(
            batch_size=batch_size,
            past_lengths=write_positions,
            max_seq=self.max_seq,
            device=self.device,
            dtype=hidden_states.dtype,
        )

        position_embeddings = _gather_rope_by_position_ids(
            self.rope_cos,
            self.rope_sin,
            position_ids=write_positions.view(batch_size, 1),
            device=self.device,
            dtype=hidden_states.dtype,
        )

        for layer_idx, layer in enumerate(self.decode_layers):
            cache = self.layer_caches[layer_idx]

            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=(cache.past_k, cache.past_v),
                position_embeddings=position_embeddings,
            )

            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "Expected decode adapter output as (hidden_states, new_k, new_v)."
                )

            hidden_states, new_k, new_v = out
            _write_decode_delta_to_cache(cache, new_k, new_v, write_positions)

        self.past_lengths = self.past_lengths + 1
        self.past_len = int(self.past_lengths.max().item())

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits_last = logits[:, -1, :]
        return logits_last

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: PromptInput,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Greedy generation using padded prefill once and static decode steps.
        """
        side = self.padding_side
        input_ids, attention_mask = self._prepare_prompt_batch(
            prompt,
        )

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        self.reset_cache()
        logits = self.prefill(
            input_ids,
            attention_mask=attention_mask,
        )

        generated_rows = self._compact_input_rows(input_ids, attention_mask.bool())

        for step in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            for batch_idx in range(len(generated_rows)):
                generated_rows[batch_idx] = torch.cat(
                    [generated_rows[batch_idx], next_token[batch_idx].to(self.device)],
                    dim=0,
                )

            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            if step == max_new_tokens - 1:
                break
            if self.past_lengths is None or torch.any(
                self.past_lengths >= self.max_seq - 1
            ):
                break

            logits = self.decode_one(next_token)

        return self._pad_token_rows(generated_rows)

    @torch.no_grad()
    def verify_against_reference(
        self,
        prompt: PromptInput,
        steps: int = 8,
        verbose: bool = True,
    ) -> None:
        """
        Compare runtime logits step-by-step against the full reference model.

        This verifies runtime correctness, not export correctness.
        If the wrapped layers are still FP-like, the mismatch should be tiny.
        If they were converted to quantized mode, some quantization error is expected.
        """
        side = self.padding_side
        input_ids, attention_mask = self._prepare_prompt_batch(
            prompt,
        )

        self.reset_cache()

        logits_rt = self.prefill(
            input_ids,
            attention_mask=attention_mask,
        )
        generated_rows = self._compact_input_rows(input_ids, attention_mask.bool())
        logits_ref = self._reference_last_logits_from_sequences(generated_rows)

        diff = (logits_rt - logits_ref).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        if verbose:
            print("=" * 100)
            print("Step 0: prefill last-token logits")
            print(f"padding_side = {side}")
            print(f"prefill input shape = {tuple(input_ids.shape)}")
            print(f"valid lengths = {[int(row.numel()) for row in generated_rows]}")
            print(f"mean|diff| = {mean_diff:.8f}")
            print(f" max|diff| = {max_diff:.8f}")
            print(f"PEIR    = {compute_peir(logits_rt, logits_ref) * 100:.6f} %")

        next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
        for batch_idx in range(len(generated_rows)):
            generated_rows[batch_idx] = torch.cat(
                [generated_rows[batch_idx], next_token[batch_idx].to(self.device)],
                dim=0,
            )

        for step in range(1, steps + 1):
            if self.past_lengths is None or torch.any(
                self.past_lengths >= self.max_seq - 1
            ):
                if verbose:
                    print("-" * 100)
                    print("Stopped because the static decode window is full.")
                break

            logits_rt = self.decode_one(next_token)
            logits_ref = self._reference_last_logits_from_sequences(generated_rows)

            diff = (logits_rt - logits_ref).abs()
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()

            if verbose:
                print("-" * 100)
                print(f"Step {step}: decode logits")
                print(
                    "sequence lengths = "
                    f"{[int(row.numel()) for row in generated_rows]}"
                )
                print(f"mean|diff| = {mean_diff:.8f}")
                print(f" max|diff| = {max_diff:.8f}")
                print(f"PEIR       = {compute_peir(logits_rt, logits_ref) * 100:.6f} %")

            next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
            for batch_idx in range(len(generated_rows)):
                generated_rows[batch_idx] = torch.cat(
                    [generated_rows[batch_idx], next_token[batch_idx].to(self.device)],
                    dim=0,
                )

            if max(int(row.numel()) for row in generated_rows) >= self.max_seq:
                if verbose:
                    print("-" * 100)
                    print("Stopped because the static decode window is full.")
                break

        if verbose:
            print("=" * 100)
            print("Verification finished.")

    @torch.no_grad()
    def dump_decode_inputs(
        self,
        input_id: Union[int, torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare one-step decode inputs without running the layers.

        This is useful when debugging export/runtime parity.
        """
        if self.past_lengths is None:
            raise RuntimeError(
                "Past lengths are not initialized. Call prefill() first."
            )

        if isinstance(input_id, int):
            input_ids = torch.tensor([[input_id]], device=self.device, dtype=torch.long)
        else:
            input_ids = input_id.to(device=self.device, dtype=torch.long)

        assert (
            input_ids.dim() == 2 and input_ids.size(1) == 1
        ), f"Decode expects input_ids as (B, 1), got {tuple(input_ids.shape)}"
        batch_size = input_ids.size(0)
        if batch_size != self.past_lengths.numel():
            raise ValueError(
                "Decode dump batch size must match initialized cache batch size. "
                f"input batch={batch_size}, cache batch={self.past_lengths.numel()}."
            )

        hidden_states = self.embed_tokens(input_ids)

        attention_mask = _build_decode_attention_mask(
            batch_size=batch_size,
            past_lengths=self.past_lengths,
            max_seq=self.max_seq,
            device=self.device,
            dtype=hidden_states.dtype,
        )

        position_embeddings = _gather_rope_by_position_ids(
            self.rope_cos,
            self.rope_sin,
            position_ids=self.past_lengths.view(batch_size, 1),
            device=self.device,
            dtype=hidden_states.dtype,
        )

        return hidden_states, attention_mask, position_embeddings

    @torch.no_grad()
    def dump_prefill_inputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare prefill inputs without running the layers.

        This is useful when debugging export/runtime parity.
        """
        assert (
            input_ids.dim() == 2
        ), f"Expected input_ids as (B, S), got {tuple(input_ids.shape)}"
        side = self.padding_side
        input_ids, attention_mask = self._pad_prefill_tensors_to_max_seq(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        batch_size, seq_len = input_ids.shape
        if seq_len != self.max_seq:
            raise RuntimeError(
                "Static prefill input length must be exactly max_seq after padding. "
                f"Got seq_len={seq_len}, max_seq={self.max_seq}."
            )

        valid_token_mask = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        _validate_padding_layout(valid_token_mask, side)
        position_ids = _build_position_ids_from_valid_token_mask(valid_token_mask)

        hidden_states = self.embed_tokens(input_ids.to(self.device))

        prefill_mask = _build_prefill_attention_mask(
            valid_token_mask=valid_token_mask,
            position_ids=position_ids,
            device=self.device,
            dtype=hidden_states.dtype,
        )
        position_embeddings = _gather_rope_by_position_ids(
            self.rope_cos,
            self.rope_sin,
            position_ids=position_ids,
            device=self.device,
            dtype=hidden_states.dtype,
        )

        assert position_embeddings[0].size(0) == batch_size
        return hidden_states, prefill_mask, position_embeddings


def main():
    """
    Build the runtime, verify step-by-step parity, and run greedy generation.
    """
    args = parse_args()
    torch.set_grad_enabled(False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
    ).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, legacy=False)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = args.padding_side

    model.config.max_position_embeddings = args.max_seq

    runtime = StaticLlamaLayerRuntime(
        model=model,
        tokenizer=tokenizer,
        max_seq=args.max_seq,
        device=args.device,
        padding_side=args.padding_side,
    )

    runtime.verify_against_reference(
        prompt=args.prompt,
        steps=args.verify_steps,
        verbose=True,
    )

    out_ids = runtime.generate_greedy(
        prompt=args.prompt,
        max_new_tokens=args.gen_steps,
        eos_token_id=tokenizer.eos_token_id,
    )

    print("=" * 100)
    print("Generated text:")
    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
