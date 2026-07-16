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

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from transformers.cache_utils import Cache


_FAKE_QUANT_META_KERNELS_REGISTERED = False


def register_fake_quant_meta_kernels_for_dynamic_export() -> None:
    """
    Register fake kernels required by torch.export dynamic-shape tracing.

    Dynamic-shape export uses FakeTensor/Meta tensors for shape propagation.
    Some fake-quantization aten operators used by PTQ observers do not provide
    fake/meta kernels in PyTorch, even though TICO can decompose those operators
    after torch.export succeeds.

    These fake kernels only describe output shapes for tracing. They do not
    replace the real fake-quantization operators in the exported graph.
    """
    global _FAKE_QUANT_META_KERNELS_REGISTERED

    if _FAKE_QUANT_META_KERNELS_REGISTERED:
        return

    def _already_registered(exc: RuntimeError) -> bool:
        msg = str(exc)
        return (
            "already has a fake impl" in msg
            or "already registered" in msg
            or "register_fake" in msg
            and "already" in msg
        )

    try:

        @torch.library.register_fake(
            "aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams"
        )
        def _fake_quantize_per_tensor_affine_cachemask_tensor_qparams_fake(
            self: torch.Tensor,
            scale: torch.Tensor,
            zero_point: torch.Tensor,
            fake_quant_enabled: torch.Tensor,
            quant_min: int,
            quant_max: int,
        ):
            return (
                torch.empty_like(self),
                torch.empty_like(self, dtype=torch.bool),
            )

    except RuntimeError as e:
        if not _already_registered(e):
            raise

    try:

        @torch.library.register_fake("aten::fake_quantize_per_tensor_affine_cachemask")
        def _fake_quantize_per_tensor_affine_cachemask_fake(
            self: torch.Tensor,
            scale: float,
            zero_point: int,
            quant_min: int,
            quant_max: int,
        ):
            return (
                torch.empty_like(self),
                torch.empty_like(self, dtype=torch.bool),
            )

    except RuntimeError as e:
        if not _already_registered(e):
            raise

    try:

        @torch.library.register_fake("aten::fake_quantize_per_channel_affine_cachemask")
        def _fake_quantize_per_channel_affine_cachemask_fake(
            self: torch.Tensor,
            scale: torch.Tensor,
            zero_point: torch.Tensor,
            axis: int,
            quant_min: int,
            quant_max: int,
        ):
            return (
                torch.empty_like(self),
                torch.empty_like(self, dtype=torch.bool),
            )

    except RuntimeError as e:
        if not _already_registered(e):
            raise

    _FAKE_QUANT_META_KERNELS_REGISTERED = True


def make_token_embedding_example_input(
    qmodel: torch.nn.Module,
    max_seq_len: int,
) -> torch.Tensor:
    """
    Create an example token-id tensor for dynamic token embedding export.
    """
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

    return torch.randint(
        low=0,
        high=int(qmodel.config.vocab_size),
        size=(1, max_seq_len),
        dtype=torch.long,
        device="cpu",
    )


def make_token_embedding_dynamic_shapes(max_seq_len: int):
    """
    Build a torch.export dynamic-shape spec for token embedding.

    Batch dimension is fixed to 1. The sequence dimension `S` is dynamic and
    bounded by `1 <= S <= max_seq_len`.

    Returns:
        A dynamic-shape spec matching the positional input tuple
        `(input_ids,)`. Returns `None` when `max_seq_len` is 1 because
        there is no useful dynamic range.
    """
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")

    if max_seq_len == 1:
        return None

    seq_dim = torch.export.Dim(
        "token_embedding_seq_len",
        min=1,
        max=int(max_seq_len),
    )
    return {"input_ids": {1: seq_dim}}


class LlamaTokenEmbeddingExportAdapter(torch.nn.Module):
    """
    Export adapter for the token embedding stage.

    The adapter maps token IDs to decoder hidden states. When SpinQuant-style
    embedding rotation is present, it is included so the output can be fed
    directly into the first separately exported decoder layer.

    Input contract:
        input_ids: Tensor with shape `(1, S)` where `S` is dynamic.

    Return contract:
        hidden_states: Tensor with shape `(1, S, hidden_size)`.
    """

    def __init__(self, qmodel: torch.nn.Module):
        super().__init__()
        llama_model = qmodel.model.wrapped
        self.embed_tokens = llama_model.embed_tokens
        self.rotate_embedding = getattr(llama_model, "rotate_embedding", None)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return decoder hidden states for the given token IDs."""
        hidden_states = self.embed_tokens(input_ids)
        if self.rotate_embedding is not None:
            hidden_states = self.rotate_embedding(hidden_states)
        return hidden_states


class LlamaLMHeadExportAdapter(torch.nn.Module):
    """
    Export adapter for the final normalization and LM head stage.

    The adapter consumes the output of the last separately exported decoder
    layer and returns vocabulary logits. It includes the final model norm and
    the optional SpinQuant LM-head rotation to preserve full-model semantics.

    Input contract:
        hidden_states: Tensor with shape `(1, 1, hidden_size)`.

    Return contract:
        logits: Tensor with shape `(1, 1, vocab_size)`.
    """

    def __init__(self, qmodel: torch.nn.Module):
        super().__init__()
        llama_model = qmodel.model.wrapped
        self.norm = llama_model.norm
        self.rotate_lm_head = getattr(qmodel, "rotate_lm_head", None)
        self.lm_head = qmodel.lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return vocabulary logits for a single decoder hidden state."""
        hidden_states = self.norm(hidden_states)
        if self.rotate_lm_head is not None:
            hidden_states = self.rotate_lm_head(hidden_states)
        return self.lm_head(hidden_states)


class LlamaAttentionPrefillExportAdapter(nn.Module):
    """
    Export adapter for prefill attention.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.
    """

    def __init__(self, wrapped, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Run prefill attention with export-friendly cache semantics.

        When `return_kv=True`, the wrapped attention module is asked to return
        only the newly produced K/V tensors instead of the full present cache.
        """
        outputs = self.wrapped(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[2]
        return hidden, new_k, new_v


class LlamaAttentionDecodeExportAdapter(nn.Module):
    """
    Export adapter for decode attention.

    Input contract
    --------------
    - hidden_states:        (B, 1, D)
    - position_embeddings:  (B, 1, head_dim)
    - attention_mask:       (B, 1, K)
    - past_key_value:       (B, num_kv_heads, K - 1, head_dim)


    Return contract
    ---------------
    - return_kv=True:
        (hidden_states, new_key, new_value)
      where new_key/new_value are the KV delta for the current token.

    - return_kv=False:
        hidden_states
    """

    def __init__(self, wrapped, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        """
        Run decode attention with delta-only cache output.

        The wrapped attention module still builds the full present K/V for
        attention computation, but only the newly produced K/V tensors are
        returned to the caller.
        """
        outputs = self.wrapped(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[2]
        return hidden, new_k, new_v


class LlamaDecoderLayerPrefillExportAdapter(nn.Module):
    """
    Export adapter for prefill.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.
    """

    def __init__(self, wrapped, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        # TODO: Replace this shared-state mutation with a per-call return-type override.
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ):
        """
        Run the decoder layer prefill path and return delta-only K/V tensors
        when caching is enabled.
        """
        outputs = self.wrapped(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=None,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[1]
        return hidden, new_k, new_v


class LlamaDecoderLayerDecodeExportAdapter(nn.Module):
    """
    Export adapter for cache-aware decoder-layer execution.

    The adapter is query-length agnostic. Single-token decode uses `Q = 1`,
    while append-prefill uses a fixed multi-token block `Q > 1`. In both cases,
    the runtime supplies static attention masks, RoPE position embeddings, and
    past-cache tensors.

    Input contract
    --------------
    - hidden_states:     (B, Q, D)
    - attention_mask:    additive mask, typically (B, Q, K)
    - past_key / past_value:
                        (B, num_kv_heads, K-Q, head_dim)
    - cos / sin:         (B, Q, head_dim)

    Return contract
    ---------------
    - return_kv=True:
        (hidden_states, new_key, new_value)
      where new_key/new_value are the delta KV tensors for the current block.

    - return_kv=False:
        hidden_states
    """

    def __init__(self, wrapped, *, return_kv: bool = True):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ):
        """
        Run a cache-aware decoder-layer block and return delta-only K/V tensors
        when caching is enabled.
        """
        outputs = self.wrapped(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=self.return_kv,
            cache_output_mode="delta",
            **kwargs,
        )

        hidden = outputs[0]

        if not self.return_kv:
            return hidden

        new_k, new_v = outputs[1]
        return hidden, new_k, new_v


class QuantLlamaForCausalLMPrefillExportAdapter(nn.Module):
    """
    Export adapter for prefill mode of QuantLLamaForCausalLM.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.

    Input contract
    --------------
    - input_ids: (B, S) input token ids

    Return contract
    ---------------
    - return_kv=True:
        (logits, past_key_values) where logits have shape (B, S) and
        past_key_values is a list of KV tensors for the next token generation.
    - return_kv=False:
        logits ~ (B, S, vocab_size)
    """

    def __init__(self, wrapped, *, return_kv: bool = False):
        super().__init__()
        self.wrapped = wrapped
        self.wrapped.return_type = "tuple"
        self.return_kv = return_kv

    def forward(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ):
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] | None = (
            None if self.return_kv is False else []
        )
        outputs = self.wrapped(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=self.return_kv,
            return_dict=True,
            **kwargs,
        )

        logits = outputs.logits
        if self.return_kv is False:
            return logits

        past_key_values = outputs.past_key_values
        return (logits, past_key_values)


class QuantLlamaForCausalLMDecodeExportAdapter(nn.Module):
    """
    Export adapter for decode mode of QuantLLamaForCausalLM.

    This adapter keeps a minimal accelerator-friendly signature while reusing the
    wrapper unchanged.

    Input contract
    --------------
    - input_ids: (B, S) input token ids
    - past_key_values: optional list of past key/value tensors, each of shape
      (B, num_kv_heads, S-1, head_dim) when provided.

    Return contract
    ---------------
    - Returns a tuple (logits, new_key_values) where logits have (B, S)-shape and
      new_key_values is a list of KV tensors for the next token.
    """

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ):
        outputs = self.wrapped(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            cache_output_mode="delta",
            **kwargs,
        )

        logits = outputs.logits
        new_key_values = outputs.past_key_values

        return (logits, new_key_values)
