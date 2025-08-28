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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from tico.experimental.quantization.ptq.quant_config import QuantConfig
from tico.experimental.quantization.ptq.wrappers.llama.quant_attn import (
    QuantLlamaAttention,
)
from tico.experimental.quantization.ptq.wrappers.llama.quant_mlp import QuantLlamaMLP
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper
from tico.experimental.quantization.ptq.wrappers.quant_module_base import (
    QuantModuleBase,
)
from tico.experimental.quantization.ptq.wrappers.registry import try_register


@try_register("transformers.models.llama.modeling_llama.LlamaDecoderLayer")
class QuantLlamaDecoderLayer(QuantModuleBase):
    """
    Quant-aware drop-in replacement for HF `LlamaDecoderLayer`.
    Signature and return-value are identical to the original.

    ▸ Attention & MLP blocks are replaced by their quantized counterparts
    ▸ LayerNorms remain FP32 (no fake-quant)
    ▸ A "static" causal mask is pre-built in `__init__` to avoid
      dynamic boolean-to-float casts inside `forward`.

    Notes on the causal mask
    ------------------------
    Building a boolean mask "inside" `forward` would introduce
    non-deterministic dynamic ops that an integer-only accelerator cannot
    fuse easily.  Therefore we:

    1. Pre-compute a full upper-triangular mask of size
       `[1, 1, max_seq, max_seq]` in `__init__`.
    2. In `forward`, if the caller passes `attention_mask=None`, we
       slice the pre-computed template to the current sequence length.
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[QuantConfig] = None,
        fp_name: Optional[str] = None,
        return_type: Optional[str] = None,
    ):
        """
        Q) Why do we need `return_type`?
        A) Different versions of `transformers` wrap the decoder output in
            different containers: a plain Tensor or a tuple.
        """
        self.return_type = return_type
        if self.return_type is None:
            import transformers

            v = tuple(map(int, transformers.__version__.split(".")[:2]))
            self.return_type = "tensor" if v >= (4, 54) else "tuple"
        assert self.return_type is not None
        super().__init__(qcfg, fp_name=fp_name)

        # Child QuantConfigs -------------------------------------------------
        attn_cfg = qcfg.child("self_attn") if qcfg else None
        mlp_cfg = qcfg.child("mlp") if qcfg else None

        # Quantized sub-modules ---------------------------------------------
        assert hasattr(fp_layer, "self_attn") and isinstance(
            fp_layer.self_attn, torch.nn.Module
        )
        assert hasattr(fp_layer, "mlp") and isinstance(fp_layer.mlp, torch.nn.Module)
        self.self_attn = PTQWrapper(
            fp_layer.self_attn, qcfg=attn_cfg, fp_name=f"{fp_name}.self_attn"
        )
        self.mlp = PTQWrapper(fp_layer.mlp, qcfg=mlp_cfg, fp_name=f"{fp_name}.mlp")

        # LayerNorms remain FP (copied from fp_layer to keep weights)
        assert hasattr(fp_layer, "input_layernorm") and isinstance(
            fp_layer.input_layernorm, torch.nn.Module
        )
        assert hasattr(fp_layer, "post_attention_layernorm") and isinstance(
            fp_layer.post_attention_layernorm, torch.nn.Module
        )
        self.input_layernorm = fp_layer.input_layernorm
        self.post_attention_layernorm = fp_layer.post_attention_layernorm

        # Static causal mask template ---------------------------------------
        assert hasattr(fp_layer.self_attn, "config") and hasattr(
            fp_layer.self_attn.config, "max_position_embeddings"
        )
        assert isinstance(fp_layer.self_attn.config.max_position_embeddings, int)
        max_seq = fp_layer.self_attn.config.max_position_embeddings
        mask = torch.full((1, 1, max_seq, max_seq), float("-120"))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

    def _slice_causal(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Return `[1,1,L,L]` causal mask slice on *device*."""
        assert isinstance(self.causal_mask_template, torch.Tensor)
        return self.causal_mask_template[..., :seq_len, :seq_len].to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional["Cache"] = None,  # type: ignore[name-defined]
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor] | torch.Tensor:
        if output_attentions:
            raise NotImplementedError(
                "QuantLlamaDecoderLayer does not support output attention yet."
            )
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if attention_mask is None or attention_mask.dtype == torch.bool:
            L = hidden_states.size(1)
            attention_mask = self._slice_causal(L, hidden_states.device)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # ─── MLP block ─────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if self.return_type == "tuple":
            return (hidden_states,)
        elif self.return_type == "tensor":
            return hidden_states
        else:
            raise RuntimeError("Invalid return type.")

    # No local observers; just recurse into children
    def _all_observers(self):
        yield from self.self_attn._all_observers()
        yield from self.mlp._all_observers()
