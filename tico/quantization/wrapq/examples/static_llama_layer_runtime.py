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
import copy
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

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
        help="Static maximum sequence length for decode runtime.",
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
    Build a wrapped decoder layer using the wrapper.

    The wrapper keeps a single HF-compatible forward for both prefill and decode.
    Export-specific specialization is handled later through export adapters.
    """
    return prepare(layer, PTQConfig())


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


def _build_decode_attention_mask(
    batch_size: int,
    past_len: int,
    max_seq: int,
    device: torch.device,
    dtype: torch.dtype,
    mask_value: float = -120.0,
) -> torch.Tensor:
    """
    Build a fully static decode mask.

    Layout assumption:
        - past KV occupies the first `past_len` slots inside the static past buffer
        - padded past slots are masked
        - current token is appended internally by the attention module at the last slot

    Returned shape:
        (B, 1, max_seq)

    Valid columns:
        [0, 1, ..., past_len - 1, max_seq - 1]
    Masked columns:
        [past_len, ..., max_seq - 2]
    """
    mask = torch.full((batch_size, 1, max_seq), mask_value, device=device, dtype=dtype)

    if past_len > 0:
        mask[:, :, :past_len] = 0.0

    mask[:, :, max_seq - 1] = 0.0
    return mask


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
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        max_seq: int,
        device: str = "cpu",
        layers: Optional[Sequence[nn.Module]] = None,
    ):
        self.model = model.eval().to(device)
        self.tokenizer = tokenizer
        self.max_seq = max_seq
        self.device = torch.device(device)

        self.embed_tokens = self.model.model.embed_tokens
        self.final_norm = self.model.model.norm
        self.lm_head = self.model.lm_head
        self.layers_ref = self.model.model.layers

        if layers is None:
            self.layers = nn.ModuleList(
                [_clone_quant_layer(layer) for layer in self.layers_ref]
            ).to(self.device)
            for layer in self.layers:
                layer.wrapped.return_type = "tuple"
        else:
            self.layers = nn.ModuleList(layers).to(self.device)

        for layer in self.layers:
            assert hasattr(layer, "wrapped")
            assert isinstance(layer.wrapped, QuantLlamaDecoderLayer)

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
        self.past_len = 0

    def reset_cache(self) -> None:
        """
        Reset all runtime KV caches.
        """
        self.layer_caches = []
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

    @torch.no_grad()
    def prefill(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Run the prompt through all prefill layers and initialize static decode caches.

        Input:
            input_ids: (B, L)

        Returns:
            logits_last: (B, vocab_size)
        """
        assert (
            input_ids.dim() == 2
        ), f"Expected input_ids as (B, L), got {tuple(input_ids.shape)}"
        batch_size, prompt_len = input_ids.shape
        assert prompt_len < self.max_seq, (
            f"Prompt length must be < max_seq so that decode still has one current slot. "
            f"Got prompt_len={prompt_len}, max_seq={self.max_seq}"
        )

        hidden_states = self.embed_tokens(input_ids.to(self.device))
        runtime_dtype = hidden_states.dtype

        self.layer_caches = self._allocate_empty_cache(batch_size, runtime_dtype)

        for layer_idx, layer in enumerate(self.layers):
            out = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                position_embeddings=None,
                past_key_value=None,
                use_cache=True,
            )

            if not isinstance(out, tuple) or len(out) != 2:
                raise RuntimeError(
                    f"Expected unified decoder layer output as "
                    f"(hidden_states, present_key_value) when use_cache=True. Got len(out) = {len(out)}."
                )

            hidden_states, present_key_value = out
            present_k, present_v = present_key_value

            assert present_k.size(2) == prompt_len
            assert present_v.size(2) == prompt_len

            self.layer_caches[layer_idx].past_k[:, :, :prompt_len, :] = present_k
            self.layer_caches[layer_idx].past_v[:, :, :prompt_len, :] = present_v

        self.past_len = prompt_len

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits_last = logits[:, -1, :]
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
        assert (
            self.past_len < self.max_seq
        ), f"Decode position overflow: past_len={self.past_len}, max_seq={self.max_seq}"

        batch_size = input_ids.size(0)
        hidden_states = self.embed_tokens(input_ids.to(self.device))

        attention_mask = _build_decode_attention_mask(
            batch_size=batch_size,
            past_len=self.past_len,
            max_seq=self.max_seq,
            device=self.device,
            dtype=hidden_states.dtype,
        )

        position_embeddings = _slice_rope(
            self.rope_cos,
            self.rope_sin,
            position=self.past_len,
            batch_size=batch_size,
            device=self.device,
            dtype=hidden_states.dtype,
        )

        for layer_idx, layer in enumerate(self.layers):
            cache = self.layer_caches[layer_idx]

            out = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_value=(cache.past_k, cache.past_v),
                position_embeddings=position_embeddings,
                use_cache=True,
            )

            if not isinstance(out, tuple) or len(out) != 2:
                raise RuntimeError(
                    "Expected unified decoder layer output as "
                    "(hidden_states, present_key_value) when use_cache=True."
                )

            hidden_states, present_key_value = out
            new_k, new_v = present_key_value

            cache.past_k[:, :, self.past_len : self.past_len + 1, :] = new_k
            cache.past_v[:, :, self.past_len : self.past_len + 1, :] = new_v

        self.past_len += 1

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        logits_last = logits[:, -1, :]
        return logits_last

    @torch.no_grad()
    def generate_greedy(
        self,
        prompt: str,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Greedy generation using prefill once and then decode-only static steps.
        """
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = batch["input_ids"].to(self.device)

        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        self.reset_cache()
        logits = self.prefill(input_ids)

        generated = input_ids.clone()

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
        """
        Compare runtime logits step-by-step against the full reference model.

        This verifies runtime correctness, not export correctness.
        If the wrapped layers are still FP-like, the mismatch should be tiny.
        If they were converted to quantized mode, some quantization error is expected.
        """
        batch = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = batch["input_ids"].to(self.device)

        self.reset_cache()

        logits_rt = self.prefill(input_ids)
        ref_out = self.model(input_ids=input_ids)
        logits_ref = ref_out.logits[:, -1, :]

        diff = (logits_rt - logits_ref).abs()
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        if verbose:
            print("=" * 100)
            print("Step 0: prefill last-token logits")
            print(f"mean|diff| = {mean_diff:.8f}")
            print(f" max|diff| = {max_diff:.8f}")
            print(f"PEIR    = {compute_peir(logits_rt, logits_ref) * 100:.6f} %")

        generated = input_ids.clone()
        next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        for step in range(1, steps + 1):
            logits_rt = self.decode_one(next_token)

            ref_out = self.model(input_ids=generated)
            logits_ref = ref_out.logits[:, -1, :]

            diff = (logits_rt - logits_ref).abs()
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()

            if verbose:
                print("-" * 100)
                print(f"Step {step}: decode logits")
                print(f"sequence length = {generated.size(1)}")
                print(f"mean|diff| = {mean_diff:.8f}")
                print(f" max|diff| = {max_diff:.8f}")
                print(f"PEIR       = {compute_peir(logits_rt, logits_ref) * 100:.6f} %")

            next_token = torch.argmax(logits_rt, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if generated.size(1) >= self.max_seq:
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
        input_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare one-step decode inputs without running the layers.

        This is useful when debugging export/runtime parity.
        """
        x = torch.tensor([[input_id]], device=self.device, dtype=torch.long)
        hidden_states = self.embed_tokens(x)

        attention_mask = _build_decode_attention_mask(
            batch_size=1,
            past_len=self.past_len,
            max_seq=self.max_seq,
            device=self.device,
            dtype=hidden_states.dtype,
        )

        position_embeddings = _slice_rope(
            self.rope_cos,
            self.rope_sin,
            position=self.past_len,
            batch_size=1,
            device=self.device,
            dtype=hidden_states.dtype,
        )

        return hidden_states, attention_mask, position_embeddings


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

    model.config.max_position_embeddings = args.max_seq

    runtime = StaticLlamaLayerRuntime(
        model=model,
        tokenizer=tokenizer,
        max_seq=args.max_seq,
        device=args.device,
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
