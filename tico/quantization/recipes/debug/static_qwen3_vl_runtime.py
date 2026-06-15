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
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoTokenizer

from tico.quantization import prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.evaluation.metric import compute_peir
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_decoder_layer import (
    QuantQwen3VLTextDecoderLayer,
)


@dataclass
class LayerCache:
    """Store one decoder layer's static KV cache tensors."""

    past_k: torch.Tensor
    past_v: torch.Tensor


@dataclass
class StaticQwen3VLRuntimeConfig:
    """Configuration for the text-only static Qwen3-VL runtime smoke test."""

    model: str = "Qwen/Qwen3-VL-2B-Instruct"
    max_seq: int = 2048
    padding_side: str = "right"
    device: str = "cpu"
    prompt: str = "The capital of France is"
    verify_steps: int = 6
    gen_steps: int = 16
    trust_remote_code: bool = True


def _clone_quant_layer(layer: nn.Module) -> nn.Module:
    """Build a quantized decoder layer wrapper for runtime simulation."""
    return prepare(layer, PTQConfig())


def _set_text_max_position_embeddings(model: nn.Module, max_seq: int) -> None:
    """Set Qwen3-VL text-position capacity before quant wrappers are created."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        text_config.max_position_embeddings = max_seq

    qwen_model = getattr(model, "model", None)
    language_model = getattr(qwen_model, "language_model", None)
    if language_model is None:
        raise RuntimeError("Expected model.model.language_model for Qwen3-VL.")

    language_model.config.max_position_embeddings = max_seq
    for layer in language_model.layers:
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "config"):
            layer.self_attn.config.max_position_embeddings = max_seq


def _build_qwen_text_rope_templates(
    text_model: nn.Module,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build text-only Qwen3-VL RoPE tables for static runtime positions.

    The returned tensors use shape `(1, max_seq, head_dim)` and can be gathered
    by compact text position IDs for padded prefill inputs.
    """
    hidden_size = int(text_model.config.hidden_size)
    dummy = torch.empty(1, max_seq, hidden_size, device=device, dtype=dtype)
    position_ids = (
        torch.arange(max_seq, device=device, dtype=torch.long)
        .view(1, 1, -1)
        .expand(3, 1, -1)
    )
    cos, sin = text_model.rotary_emb(dummy, position_ids)

    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
    if sin.dim() == 2:
        sin = sin.unsqueeze(0)

    if cos.size(0) != 1:
        cos = cos[:1]
    if sin.size(0) != 1:
        sin = sin[:1]

    return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _normalize_valid_token_mask(
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor],
    *,
    pad_token_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """
    Build a boolean mask where True marks real prompt tokens.

    Callers should pass `attention_mask` explicitly when the pad token can also
    appear as a real token.
    """
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
    """Validate that each batch row uses contiguous left or right padding."""
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
    """
    Build compact absolute position IDs for padded text-only prefill inputs.

    Real tokens receive positions `0..valid_length-1`. Padding slots receive
    position 0 because their logits and K/V values are ignored by the runtime.
    """
    position_ids = valid_token_mask.to(torch.long).cumsum(dim=1) - 1
    position_ids = torch.clamp(position_ids, min=0)
    return position_ids.masked_fill(~valid_token_mask, 0)


def _last_valid_token_indices(valid_token_mask: torch.Tensor) -> torch.LongTensor:
    """Return the input index of the last real token for each batch row."""
    batch_size, seq_len = valid_token_mask.shape
    positions = torch.arange(seq_len, device=valid_token_mask.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    return positions.masked_fill(~valid_token_mask, 0).max(dim=1).values


def _gather_last_token_logits(
    logits: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather logits at the last real token index of each padded prefill row."""
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
    Gather static RoPE tensors by per-token absolute position IDs.

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


def _slice_rope(
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    position: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice one absolute-position RoPE entry for static decode."""
    if position < 0 or position >= rope_cos.size(1):
        raise ValueError(
            f"position must be within [0, {rope_cos.size(1)}), got {position}."
        )

    cos = rope_cos[:, position : position + 1, :].to(device=device, dtype=dtype)
    sin = rope_sin[:, position : position + 1, :].to(device=device, dtype=dtype)
    if batch_size != 1:
        cos = cos.expand(batch_size, -1, -1).contiguous()
        sin = sin.expand(batch_size, -1, -1).contiguous()
    return cos, sin


def _build_prefill_attention_mask(
    valid_token_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    """
    Build a static additive causal + padding mask for prefill.

    Returns a tensor with shape `(B, 1, S, S)`.
    """
    batch_size, seq_len = valid_token_mask.shape
    causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    causal = torch.tril(causal).unsqueeze(0).expand(batch_size, -1, -1)
    key_valid = valid_token_mask.to(device).unsqueeze(1).expand(-1, seq_len, -1)
    query_valid = valid_token_mask.to(device).unsqueeze(2).expand(-1, -1, seq_len)
    valid = causal & key_valid & query_valid
    mask = torch.zeros(batch_size, 1, seq_len, seq_len, device=device, dtype=dtype)
    return mask.masked_fill(~valid.unsqueeze(1), mask_value)


def _build_decode_attention_mask(
    batch_size: int,
    past_len: int,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    """
    Build a static additive decode mask for a `(max_seq - 1) + 1` KV layout.

    The CPU cache stores valid past tokens at the beginning of the fixed cache.
    The decode adapter appends the current token after the fixed past tensor, so
    the current key is located at index `max_seq - 1` inside the attention matmul.
    """
    if past_len < 0 or past_len >= max_seq:
        raise ValueError(f"past_len must be within [0, {max_seq}), got {past_len}.")

    mask = torch.full(
        (batch_size, 1, 1, max_seq),
        mask_value,
        device=device,
        dtype=dtype,
    )
    if past_len > 0:
        mask[:, :, :, :past_len] = 0.0
    mask[:, :, :, max_seq - 1] = 0.0
    return mask


class StaticQwen3VLTextLayerRuntime:
    """
    Static-shape text-only runtime over separately wrapped Qwen3-VL decoder layers.

    CPU responsibilities:
      - token embedding,
      - RoPE lookup,
      - attention-mask construction,
      - KV-cache allocation and updates,
      - final norm and LM head.

    NPU-simulated responsibilities:
      - dense decoder-layer prefill and decode adapters.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
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

        self.qwen_model = self.model.model
        self.text_model = self.qwen_model.language_model
        self.embed_tokens = self.text_model.embed_tokens
        self.final_norm = self.text_model.norm
        self.lm_head = self.model.lm_head
        self.rotate_lm_head = getattr(self.model, "rotate_lm_head", None)
        self.layers_ref = self.text_model.layers

        if layers is None:
            self.layers = nn.ModuleList(
                [_clone_quant_layer(layer) for layer in self.layers_ref]
            ).to(self.device)
        else:
            self.layers = nn.ModuleList(layers).to(self.device)

        for layer in self.layers:
            assert hasattr(layer, "wrapped")
            assert isinstance(layer.wrapped, QuantQwen3VLTextDecoderLayer)

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

        self.config = self.text_model.config
        self.hidden_size = self.config.hidden_size
        self.num_hidden_layers = self.config.num_hidden_layers
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", None) or (
            self.hidden_size // self.config.num_attention_heads
        )

        self.rope_cos, self.rope_sin = _build_qwen_text_rope_templates(
            self.text_model,
            max_seq=self.max_seq,
            device=self.device,
            dtype=torch.float32,
        )
        self.layer_caches: list[LayerCache] = []
        self.past_len = 0

    def reset_cache(self) -> None:
        """Clear CPU-managed KV caches."""
        self.layer_caches = []
        self.past_len = 0

    def _allocate_empty_cache(
        self,
        batch_size: int,
        dtype: torch.dtype,
    ) -> list[LayerCache]:
        """Allocate static `(max_seq - 1)` past-cache tensors for every layer."""
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
        """Run static prefill and return last-real-token logits."""
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
        if self.rotate_lm_head is not None:
            hidden_states = self.rotate_lm_head(hidden_states)
        logits = self.lm_head(hidden_states)
        return _gather_last_token_logits(logits, valid)

    @torch.no_grad()
    def decode_one(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Run one static decode step and return logits for the current token."""
        assert input_ids.dim() == 2 and input_ids.size(1) == 1
        assert len(self.layer_caches) == self.num_hidden_layers, "Call prefill() first."
        assert self.past_len < self.max_seq

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
                position_embeddings=position_embeddings,
                past_key_values=(cache.past_k, cache.past_v),
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
        if self.rotate_lm_head is not None:
            hidden_states = self.rotate_lm_head(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits[:, -1, :]

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: str,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Generate tokens greedily with the static prefill/decode runtime."""
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
        self,
        prompt: str,
        steps: int = 8,
        verbose: bool = True,
    ) -> None:
        """Compare static runtime logits with the HF reference model."""
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
            "Step 0: prefill last-token logits",
            logits_rt,
            logits_ref,
            verbose,
        )

        generated = compact_input_ids.clone()
        next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        for step in range(1, steps + 1):
            logits_rt = self.decode_one(next_token)
            ref_out = self.model(input_ids=generated)
            logits_ref = ref_out.logits[:, -1, :]
            self._print_diff(
                f"Step {step}: decode logits",
                logits_rt,
                logits_ref,
                verbose,
            )

            next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            if generated.size(1) >= self.max_seq:
                if verbose:
                    print("Stopped because the static decode window is full.")
                break

    @staticmethod
    def _print_diff(
        title: str,
        actual: torch.Tensor,
        expected: torch.Tensor,
        verbose: bool,
    ) -> None:
        """Print absolute-difference and PEIR metrics for two logits tensors."""
        if not verbose:
            return
        diff = (actual - expected).abs()
        print("=" * 100)
        print(title)
        print(f"mean|diff| = {diff.mean().item():.8f}")
        print(f" max|diff| = {diff.max().item():.8f}")
        print(f"PEIR       = {compute_peir(actual, expected) * 100:.6f} %")


def run_static_qwen3_vl_runtime(cfg: StaticQwen3VLRuntimeConfig) -> None:
    """Load a Qwen3-VL model and run the text-only static runtime smoke test."""
    torch.set_grad_enabled(False)

    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model,
        dtype=torch.float32,
        trust_remote_code=cfg.trust_remote_code,
    ).to(cfg.device)
    _set_text_max_position_embeddings(model, cfg.max_seq)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = cfg.padding_side

    runtime = StaticQwen3VLTextLayerRuntime(
        model=model,
        tokenizer=tokenizer,
        max_seq=cfg.max_seq,
        device=cfg.device,
        padding_side=cfg.padding_side,
    )

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
    print(tokenizer.decode(out_ids[0], skip_special_tokens=True))


def _parse_args() -> StaticQwen3VLRuntimeConfig:
    """Parse command-line arguments for the static runtime smoke test."""
    parser = argparse.ArgumentParser(
        description="Run the text-only static Qwen3-VL layer runtime."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=StaticQwen3VLRuntimeConfig.model,
        help="HF model name or local model path.",
    )
    parser.add_argument(
        "--max-seq",
        type=int,
        default=StaticQwen3VLRuntimeConfig.max_seq,
        help="Static sequence length used by prefill and decode.",
    )
    parser.add_argument(
        "--padding-side",
        type=str,
        choices=("left", "right"),
        default=StaticQwen3VLRuntimeConfig.padding_side,
        help="Padding direction for static prefill inputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=StaticQwen3VLRuntimeConfig.device,
        help="Execution device, such as cpu or cuda.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=StaticQwen3VLRuntimeConfig.prompt,
        help="Prompt used for verification and greedy generation.",
    )
    parser.add_argument(
        "--verify-steps",
        type=int,
        default=StaticQwen3VLRuntimeConfig.verify_steps,
        help="Number of decode steps for reference verification.",
    )
    parser.add_argument(
        "--gen-steps",
        type=int,
        default=StaticQwen3VLRuntimeConfig.gen_steps,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code when loading the model and tokenizer.",
    )
    args = parser.parse_args()

    return StaticQwen3VLRuntimeConfig(
        model=args.model,
        max_seq=args.max_seq,
        padding_side=args.padding_side,
        device=args.device,
        prompt=args.prompt,
        verify_steps=args.verify_steps,
        gen_steps=args.gen_steps,
        trust_remote_code=not args.no_trust_remote_code,
    )


if __name__ == "__main__":
    run_static_qwen3_vl_runtime(_parse_args())
