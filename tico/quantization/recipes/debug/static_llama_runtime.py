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

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from tico.quantization import prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)


@dataclass
class LayerCache:
    past_k: torch.Tensor
    past_v: torch.Tensor


@dataclass
class StaticLlamaRuntimeConfig:
    model: str = "Maykeye/TinyLLama-v0"
    max_seq: int = 256
    padding_side: str = "right"
    device: str = "cpu"
    prompt: str = "The capital of France is"
    verify_steps: int = 6
    gen_steps: int = 16
    run_baseline: bool = True
    run_multiturn: bool = False
    multiturn_first_prompt: str = "User: What is the capital of France?\nAssistant:"
    multiturn_second_turn: str = "\nUser: What country is that city in?\nAssistant:"
    multiturn_decode_steps_before_append: int = 4
    multiturn_decode_steps_after_append: int = 4
    append_bucket: int = 16
    compare_append_prefill_with_decode_loop: bool = True


def _clone_quant_layer(layer: nn.Module) -> nn.Module:
    return prepare(layer, PTQConfig())


def _build_rope_templates_from_config(
    config,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    return cos.unsqueeze(0).to(dtype=dtype), sin.unsqueeze(0).to(dtype=dtype)


def _normalize_valid_token_mask(
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    *,
    pad_token_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        if pad_token_id is None:
            valid = torch.ones(input_ids.shape, device=device, dtype=torch.bool)
        else:
            valid = input_ids.to(device).ne(int(pad_token_id))
    else:
        if tuple(attention_mask.shape) != tuple(input_ids.shape):
            raise ValueError(
                "attention_mask must have the same shape as input_ids. "
                f"Got attention_mask={tuple(attention_mask.shape)}, "
                f"input_ids={tuple(input_ids.shape)}."
            )
        valid = attention_mask.to(device=device).bool()

    if torch.any(valid.sum(dim=1) == 0):
        raise ValueError("Each batch row must contain at least one real token.")
    return valid


def _validate_padding_layout(valid_token_mask: torch.Tensor, padding_side: str) -> None:
    batch_size, seq_len = valid_token_mask.shape
    valid_lengths = valid_token_mask.sum(dim=1)
    positions = torch.arange(seq_len, device=valid_token_mask.device).unsqueeze(0)

    if padding_side == "right":
        expected = positions < valid_lengths.unsqueeze(1)
    elif padding_side == "left":
        expected = positions >= (seq_len - valid_lengths).unsqueeze(1)
    else:
        raise ValueError(
            f"padding_side must be 'left' or 'right'. got {padding_side!r}"
        )

    if not torch.equal(valid_token_mask, expected):
        raise ValueError(
            "Input padding layout does not match padding_side. "
            f"Expected contiguous {padding_side} padding for shape "
            f"(B={batch_size}, S={seq_len})."
        )


def _build_position_ids_from_valid_token_mask(
    valid_token_mask: torch.Tensor,
) -> torch.LongTensor:
    position_ids = valid_token_mask.to(torch.long).cumsum(dim=1) - 1
    position_ids = torch.clamp(position_ids, min=0)
    return position_ids.masked_fill(~valid_token_mask, 0)


def _last_valid_token_indices(valid_token_mask: torch.Tensor) -> torch.LongTensor:
    batch_size, seq_len = valid_token_mask.shape
    positions = torch.arange(seq_len, device=valid_token_mask.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    return positions.masked_fill(~valid_token_mask, 0).max(dim=1).values


def _gather_last_token_logits(
    logits: torch.Tensor, valid_token_mask: torch.Tensor
) -> torch.Tensor:
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
    flat_pos = position_ids.to(device=device).reshape(-1)
    cos = rope_cos.index_select(1, flat_pos).reshape(
        position_ids.size(0), position_ids.size(1), -1
    )
    sin = rope_sin.index_select(1, flat_pos).reshape(
        position_ids.size(0), position_ids.size(1), -1
    )
    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _slice_rope(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    position: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = rope_cos[:, position : position + 1, :].to(device=device, dtype=dtype)
    sin = rope_sin[:, position : position + 1, :].to(device=device, dtype=dtype)
    if batch_size != 1:
        cos = cos.expand(batch_size, -1, -1).contiguous()
        sin = sin.expand(batch_size, -1, -1).contiguous()
    return cos, sin


def _slice_rope_range(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    start: int,
    seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return RoPE tensors for an absolute position range."""
    end = start + seq_len
    if start < 0 or end > rope_cos.size(1):
        raise ValueError(
            f"RoPE range [{start}, {end}) exceeds available table length "
            f"{rope_cos.size(1)}."
        )
    cos = rope_cos[:, start:end, :].to(device=device, dtype=dtype)
    sin = rope_sin[:, start:end, :].to(device=device, dtype=dtype)
    if batch_size != 1:
        cos = cos.expand(batch_size, -1, -1).contiguous()
        sin = sin.expand(batch_size, -1, -1).contiguous()
    return cos, sin


def _compact_valid_tokens(
    input_ids: torch.LongTensor,
    valid_token_mask: torch.Tensor,
) -> torch.LongTensor:
    """Compact equally sized valid token rows into a dense token tensor."""
    valid_lengths = valid_token_mask.sum(dim=1)
    if torch.unique(valid_lengths).numel() != 1:
        raise ValueError("All batch rows must have the same valid token length.")
    return input_ids.to(valid_token_mask.device)[valid_token_mask].reshape(
        input_ids.size(0), int(valid_lengths[0].item())
    )


def _build_prefill_attention_mask(
    valid_token_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    batch_size, seq_len = valid_token_mask.shape
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    causal = torch.tril(causal).unsqueeze(0).expand(batch_size, -1, -1)
    key_valid = valid_token_mask.to(device).unsqueeze(1).expand(-1, seq_len, -1)
    query_valid = valid_token_mask.to(device).unsqueeze(2).expand(-1, -1, seq_len)
    valid = causal & key_valid & query_valid
    mask = torch.zeros(batch_size, seq_len, seq_len, device=device, dtype=dtype)
    return mask.masked_fill(~valid, mask_value)


def _build_decode_attention_mask(
    batch_size: int,
    past_len: int,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    mask = torch.full((batch_size, 1, max_seq), mask_value, device=device, dtype=dtype)
    if past_len > 0:
        mask[:, :, :past_len] = 0.0
    mask[:, :, max_seq - 1] = 0.0
    return mask


def _build_append_prefill_attention_mask(
    batch_size: int,
    past_len: int,
    actual_len: int,
    bucket_size: int,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    """Build a static additive mask for one append-prefill bucket."""
    if actual_len < 1:
        raise ValueError(f"actual_len must be positive, got {actual_len}.")
    if actual_len > bucket_size:
        raise ValueError(
            f"actual_len must not exceed bucket_size. Got actual_len={actual_len}, "
            f"bucket_size={bucket_size}."
        )
    if past_len > max_seq - bucket_size:
        raise ValueError(
            f"past_len={past_len} does not fit an append bucket of size "
            f"{bucket_size} with max_seq={max_seq}."
        )

    mask = torch.full(
        (batch_size, bucket_size, max_seq),
        mask_value,
        device=device,
        dtype=dtype,
    )
    if past_len > 0:
        mask[:, :actual_len, :past_len] = 0.0

    tail_start = max_seq - bucket_size
    causal = torch.tril(
        torch.ones(bucket_size, bucket_size, device=device, dtype=torch.bool)
    )
    tail_mask = torch.zeros(bucket_size, bucket_size, device=device, dtype=dtype)
    tail_mask = tail_mask.masked_fill(~causal, mask_value)
    mask[:, :actual_len, tail_start:max_seq] = (
        tail_mask[:actual_len, :].unsqueeze(0).expand(batch_size, -1, -1)
    )
    return mask


class StaticLlamaLayerRuntime:
    """Static-shape runtime over separately wrapped LLaMA decoder layers."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_seq: int,
        device: str = "cpu",
        padding_side: str = "right",
        layers: Optional[Sequence[nn.Module]] = None,
    ):
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
        self.append_prefill_layers = nn.ModuleList(
            [
                layer.wrapped.as_export_module("append_prefill", return_kv=True)
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
        self.layer_caches: list[LayerCache] = []
        self.past_len = 0

    def reset_cache(self) -> None:
        self.layer_caches = []
        self.past_len = 0

    def _allocate_empty_cache(
        self, batch_size: int, dtype: torch.dtype
    ) -> list[LayerCache]:
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
            caches.append(LayerCache(past_k=past_k, past_v=torch.zeros_like(past_k)))
        return caches

    @torch.no_grad()
    def prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            input_ids.dim() == 2
        ), f"Expected input_ids as (B, S), got {tuple(input_ids.shape)}"
        batch_size, seq_len = input_ids.shape
        assert seq_len == self.max_seq, (
            f"Static prefill expects padded length == max_seq. "
            f"Got seq_len={seq_len}, max_seq={self.max_seq}."
        )

        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        _validate_padding_layout(valid, self.padding_side)

        position_ids = _build_position_ids_from_valid_token_mask(valid)
        hidden_states = self.embed_tokens(input_ids.to(self.device))
        runtime_dtype = hidden_states.dtype

        self.layer_caches = self._allocate_empty_cache(batch_size, runtime_dtype)

        attn_mask = _build_prefill_attention_mask(valid, self.device, runtime_dtype)
        pos_embeds = _gather_rope_by_position_ids(
            self.rope_cos,
            self.rope_sin,
            position_ids,
            self.device,
            runtime_dtype,
        )

        for layer_idx, layer in enumerate(self.prefill_layers):
            out = layer(
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=pos_embeds,
            )
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "Expected prefill adapter output as (hidden_states, new_k, new_v)."
                )
            hidden_states, new_k, new_v = out

            valid_lengths = valid.sum(dim=1)
            for b in range(batch_size):
                length = int(valid_lengths[b].item())
                src_start = 0 if self.padding_side == "right" else self.max_seq - length
                src_end = src_start + length
                self.layer_caches[layer_idx].past_k[b, :, :length, :] = new_k[
                    b, :, src_start:src_end, :
                ]
                self.layer_caches[layer_idx].past_v[b, :, :length, :] = new_v[
                    b, :, src_start:src_end, :
                ]

        if torch.unique(valid.sum(dim=1)).numel() != 1:
            raise ValueError(
                "Static decode currently requires equal valid prompt length for all batch rows."
            )
        self.past_len = int(valid.sum(dim=1)[0].item())

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return _gather_last_token_logits(logits, valid)

    @torch.no_grad()
    def append_prefill(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        bucket_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Append a multi-token user-turn block to the existing external KV cache.

        The method mirrors the proposed static append-prefill NPU contract. The
        runtime owns the dynamic metadata (`past_len`, actual token count,
        position selection, mask construction, and compact cache write-back),
        while every decoder-layer call receives static tensor shapes based on the
        selected `bucket_size`.
        """
        assert (
            input_ids.dim() == 2
        ), f"Expected input_ids as (B, T), got {tuple(input_ids.shape)}"
        assert len(self.layer_caches) == self.num_hidden_layers, "Call prefill() first."

        batch_size = input_ids.size(0)
        if batch_size != self.layer_caches[0].past_k.size(0):
            raise ValueError(
                "append_prefill batch size must match the existing cache batch size. "
                f"Got input batch={batch_size}, cache batch="
                f"{self.layer_caches[0].past_k.size(0)}."
            )

        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        _validate_padding_layout(valid, self.padding_side)
        compact_input_ids = _compact_valid_tokens(input_ids, valid)
        actual_len = compact_input_ids.size(1)

        if bucket_size is None:
            bucket_size = actual_len
        bucket_size = int(bucket_size)
        if bucket_size < actual_len:
            raise ValueError(
                f"bucket_size must cover all valid tokens. Got bucket_size="
                f"{bucket_size}, actual_len={actual_len}."
            )
        if bucket_size < 1:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}.")

        start_pos = self.past_len
        if start_pos + actual_len > self.max_seq - 1:
            raise ValueError(
                "Not enough compact KV-cache capacity for append_prefill. "
                f"past_len={start_pos}, actual_len={actual_len}, "
                f"cache_capacity={self.max_seq - 1}."
            )
        if start_pos + bucket_size > self.max_seq:
            raise ValueError(
                "The selected append_prefill bucket does not fit the static "
                f"attention window. past_len={start_pos}, bucket_size={bucket_size}, "
                f"max_seq={self.max_seq}."
            )

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        padded_input_ids = torch.full(
            (batch_size, bucket_size),
            int(pad_token_id),
            device=self.device,
            dtype=torch.long,
        )
        padded_input_ids[:, :actual_len] = compact_input_ids.to(self.device)

        hidden_states = self.embed_tokens(padded_input_ids)
        runtime_dtype = hidden_states.dtype

        attention_mask_for_append = _build_append_prefill_attention_mask(
            batch_size=batch_size,
            past_len=start_pos,
            actual_len=actual_len,
            bucket_size=bucket_size,
            max_seq=self.max_seq,
            device=self.device,
            dtype=runtime_dtype,
        )
        position_embeddings = _slice_rope_range(
            self.rope_cos,
            self.rope_sin,
            start=start_pos,
            seq_len=bucket_size,
            batch_size=batch_size,
            device=self.device,
            dtype=runtime_dtype,
        )

        past_storage_len_for_attn = self.max_seq - bucket_size
        for layer_idx, layer in enumerate(self.append_prefill_layers):
            cache = self.layer_caches[layer_idx]
            past_key_value = (
                cache.past_k[:, :, :past_storage_len_for_attn, :],
                cache.past_v[:, :, :past_storage_len_for_attn, :],
            )
            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask_for_append,
                past_key_value=past_key_value,
                position_embeddings=position_embeddings,
            )
            if not isinstance(out, tuple) or len(out) != 3:
                raise RuntimeError(
                    "Expected append-prefill adapter output as "
                    "(hidden_states, new_k, new_v)."
                )
            hidden_states, new_k, new_v = out
            cache.past_k[:, :, start_pos : start_pos + actual_len, :] = new_k[
                :, :, :actual_len, :
            ]
            cache.past_v[:, :, start_pos : start_pos + actual_len, :] = new_v[
                :, :, :actual_len, :
            ]

        self.past_len = start_pos + actual_len
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits[:, actual_len - 1, :]

    @torch.no_grad()
    def append_prefill_chunked(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        bucket_size: int,
    ) -> torch.Tensor:
        """Append a token block using one or more static append-prefill buckets."""
        if bucket_size < 1:
            raise ValueError(f"bucket_size must be positive, got {bucket_size}.")

        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        _validate_padding_layout(valid, self.padding_side)
        compact_input_ids = _compact_valid_tokens(input_ids, valid)

        # TODO Select the smallest exported bucket satisfying
        # chunk_len <= Q and self.past_len + Q <= self.max_seq. Define the
        # context-length policy to use when no exported bucket fits.
        logits = None
        for start in range(0, compact_input_ids.size(1), bucket_size):
            chunk = compact_input_ids[:, start : start + bucket_size].to(self.device)
            chunk_mask = torch.ones_like(chunk, dtype=torch.long, device=self.device)
            logits = self.append_prefill(
                chunk,
                chunk_mask,
                bucket_size=bucket_size,
            )

        if logits is None:
            raise RuntimeError("append_prefill_chunked received no valid token.")
        return logits

    @torch.no_grad()
    def append_turn_by_decode_loop(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Append a user turn by repeatedly invoking the single-token decode path.

        This is a correctness fallback for comparing the append-prefill path
        against the existing decode contract. It is not the preferred serving
        path because it runs one graph invocation per user token.
        """
        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        _validate_padding_layout(valid, self.padding_side)
        compact_input_ids = _compact_valid_tokens(input_ids, valid)
        if self.past_len + compact_input_ids.size(1) > self.max_seq - 1:
            raise ValueError("Not enough KV-cache capacity for decode-loop append.")

        logits = None
        for i in range(compact_input_ids.size(1)):
            logits = self.decode_one(compact_input_ids[:, i : i + 1].to(self.device))
        assert logits is not None
        return logits

    @torch.no_grad()
    def decode_one(self, input_ids: torch.LongTensor) -> torch.Tensor:
        assert input_ids.dim() == 2 and input_ids.size(1) == 1
        assert len(self.layer_caches) == self.num_hidden_layers, "Call prefill() first."
        assert self.past_len < self.max_seq - 1

        batch_size = input_ids.size(0)
        hidden_states = self.embed_tokens(input_ids.to(self.device))

        attention_mask = _build_decode_attention_mask(
            batch_size,
            self.past_len,
            self.max_seq,
            self.device,
            hidden_states.dtype,
        )
        position_embeddings = _slice_rope(
            self.rope_cos,
            self.rope_sin,
            position=self.past_len,
            batch_size=batch_size,
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
            cache.past_k[:, :, self.past_len : self.past_len + 1, :] = new_k
            cache.past_v[:, :, self.past_len : self.past_len + 1, :] = new_v

        self.past_len += 1
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits[:, -1, :]

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: str,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq,
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        self.reset_cache()
        logits = self.prefill(input_ids, attention_mask)
        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        prompt_len = int(valid.sum(dim=1)[0].item())
        generated = input_ids[valid].reshape(1, prompt_len).clone()

        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id):
                break
            logits = self.decode_one(next_token)
        return generated

    @torch.no_grad()
    def verify_against_reference(
        self, prompt: str, steps: int = 8, verbose: bool = True
    ) -> None:
        batch = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq,
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.reset_cache()
        logits_rt = self.prefill(input_ids, attention_mask)
        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        compact_input_ids = input_ids[valid].reshape(input_ids.size(0), -1)
        ref_out = self.model(input_ids=compact_input_ids)
        logits_ref = ref_out.logits[:, -1, :]

        self._print_diff(
            "Step 0: prefill last-token logits", logits_rt, logits_ref, verbose
        )

        generated = compact_input_ids.clone()
        next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        for step in range(1, steps + 1):
            logits_rt = self.decode_one(next_token)
            ref_out = self.model(input_ids=generated)
            logits_ref = ref_out.logits[:, -1, :]
            self._print_diff(
                f"Step {step}: decode logits", logits_rt, logits_ref, verbose
            )

            next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if generated.size(1) >= self.max_seq - 1:
                if verbose:
                    print("Stopped because the static decode cache window is full.")
                break

    @staticmethod
    def _print_diff(
        title: str, actual: torch.Tensor, expected: torch.Tensor, verbose: bool
    ) -> None:
        if not verbose:
            return
        diff = (actual - expected).abs()
        print("=" * 100)
        print(title)
        print(f"mean|diff| = {diff.mean().item():.8f}")
        print(f" max|diff| = {diff.max().item():.8f}")
        print(f"PEIR       = {compute_peir(actual, expected) * 100:.6f} %")


def _clone_layer_caches(caches: Sequence[LayerCache]) -> list[LayerCache]:
    """Return detached copies of per-layer KV cache tensors."""
    return [
        LayerCache(past_k=cache.past_k.clone(), past_v=cache.past_v.clone())
        for cache in caches
    ]


def _tokenize_to_device(
    tokenizer: AutoTokenizer,
    text: str,
    *,
    device: torch.device,
    add_special_tokens: bool,
    padding: bool | str = False,
    truncation: bool = False,
    max_length: Optional[int] = None,
) -> tuple[torch.LongTensor, Optional[torch.Tensor]]:
    """Tokenize text and move token tensors to the requested device."""
    batch = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask


def _run_decode_steps_against_reference(
    runtime: StaticLlamaLayerRuntime,
    generated: torch.LongTensor,
    logits: torch.Tensor,
    *,
    steps: int,
    title_prefix: str,
    verbose: bool,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """Run greedy decode steps and compare each step with the reference model."""
    for step in range(1, steps + 1):
        if generated.size(1) >= runtime.max_seq - 1:
            if verbose:
                print("Stopped because the static decode cache window is full.")
            break

        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        logits = runtime.decode_one(next_token)
        ref_out = runtime.model(input_ids=generated)
        logits_ref = ref_out.logits[:, -1, :]
        runtime._print_diff(
            f"{title_prefix} step {step}",
            logits,
            logits_ref,
            verbose,
        )

    return generated, logits


def _run_multiturn_append_prefill_check(
    runtime: StaticLlamaLayerRuntime,
    cfg: StaticLlamaRuntimeConfig,
    *,
    verbose: bool = True,
) -> torch.LongTensor:
    """Validate append-prefill in a two-turn static runtime scenario.

    Returns:
        Compact token IDs containing the full generated multi-turn sequence.
    """
    if cfg.append_bucket < 1:
        raise ValueError(f"append_bucket must be positive, got {cfg.append_bucket}.")

    first_input_ids, first_attention_mask = _tokenize_to_device(
        runtime.tokenizer,
        cfg.multiturn_first_prompt,
        device=runtime.device,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=runtime.max_seq,
    )

    runtime.reset_cache()
    logits = runtime.prefill(first_input_ids, first_attention_mask)
    first_valid = _normalize_valid_token_mask(
        first_input_ids,
        first_attention_mask,
        pad_token_id=runtime.tokenizer.pad_token_id,
        device=runtime.device,
    )
    generated = _compact_valid_tokens(first_input_ids, first_valid)
    ref_out = runtime.model(input_ids=generated)
    runtime._print_diff(
        "Multi-turn step 0: first prefill last-token logits",
        logits,
        ref_out.logits[:, -1, :],
        verbose,
    )

    generated, logits = _run_decode_steps_against_reference(
        runtime,
        generated,
        logits,
        steps=cfg.multiturn_decode_steps_before_append,
        title_prefix="Multi-turn pre-append decode logits",
        verbose=verbose,
    )

    second_input_ids, second_attention_mask = _tokenize_to_device(
        runtime.tokenizer,
        cfg.multiturn_second_turn,
        device=runtime.device,
        add_special_tokens=False,
    )
    second_valid = _normalize_valid_token_mask(
        second_input_ids,
        second_attention_mask,
        pad_token_id=runtime.tokenizer.pad_token_id,
        device=runtime.device,
    )
    compact_second_input_ids = _compact_valid_tokens(second_input_ids, second_valid)

    before_append_caches = _clone_layer_caches(runtime.layer_caches)
    before_append_past_len = runtime.past_len

    logits = runtime.append_prefill_chunked(
        second_input_ids,
        second_attention_mask,
        bucket_size=cfg.append_bucket,
    )
    generated = torch.cat(
        [generated, compact_second_input_ids.to(runtime.device)], dim=1
    )
    ref_out = runtime.model(input_ids=generated)
    runtime._print_diff(
        f"Multi-turn append-prefill logits (bucket={cfg.append_bucket})",
        logits,
        ref_out.logits[:, -1, :],
        verbose,
    )

    if cfg.compare_append_prefill_with_decode_loop:
        append_prefill_caches = _clone_layer_caches(runtime.layer_caches)
        append_prefill_past_len = runtime.past_len

        runtime.layer_caches = _clone_layer_caches(before_append_caches)
        runtime.past_len = before_append_past_len
        logits_loop = runtime.append_turn_by_decode_loop(
            second_input_ids,
            second_attention_mask,
        )
        runtime._print_diff(
            "Multi-turn append-prefill vs decode-loop append logits",
            logits,
            logits_loop,
            verbose,
        )

        runtime.layer_caches = append_prefill_caches
        runtime.past_len = append_prefill_past_len

    generated, logits = _run_decode_steps_against_reference(
        runtime,
        generated,
        logits,
        steps=cfg.multiturn_decode_steps_after_append,
        title_prefix="Multi-turn post-append decode logits",
        verbose=verbose,
    )

    return generated


def run_static_llama_runtime(cfg: StaticLlamaRuntimeConfig) -> None:
    torch.set_grad_enabled(False)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        dtype=torch.float32,
    ).to(cfg.device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, legacy=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg.padding_side

    model.config.max_position_embeddings = cfg.max_seq

    runtime = StaticLlamaLayerRuntime(
        model=model,
        tokenizer=tokenizer,
        max_seq=cfg.max_seq,
        device=cfg.device,
        padding_side=cfg.padding_side,
    )

    if cfg.run_baseline:
        runtime.verify_against_reference(
            prompt=cfg.prompt,
            steps=cfg.verify_steps,
            verbose=True,
        )

        out_ids = runtime.generate_greedy(
            prompt=cfg.prompt,
            max_new_tokens=cfg.gen_steps,
            eos_token_id=tokenizer.eos_token_id,
        )

        print("=" * 100)
        print("Generated text:")
        print(
            tokenizer.decode(
                out_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )

    if cfg.run_multiturn:
        out_ids = _run_multiturn_append_prefill_check(runtime, cfg, verbose=True)
        print("=" * 100)
        print("Multi-turn generated text:")
        print(
            tokenizer.decode(
                out_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        )
