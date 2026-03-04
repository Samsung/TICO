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

from typing import Optional, Tuple

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register(
    "transformers.models.llama.modeling_llama.LlamaDecoderLayer",
    variant="decode",
)
class QuantLlamaDecoderLayerDecode(QuantModuleBase):
    """
    Decode-only quant wrapper for HF `LlamaDecoderLayer` with fully static shapes.

    Design constraints
    ------------------
    - Fully static tensor shapes at runtime:
        hidden_states:        (B, 1, D)
        position_embeddings:
          cos:                (B, 1, head_dim)
          sin:                (B, 1, head_dim)
        attention_mask:       (B, 1, max_seq_len)      # additive mask: 0 or negative (e.g., -120)
        past_key_value:
          past_key:           (B, num_kv_heads, max_seq_len - 1, head_dim)  # already RoPE-applied
          past_value:         (B, num_kv_heads, max_seq_len - 1, head_dim)

    - No dynamic attention mask construction inside `forward`.
      The host/runtime must provide the additive mask of the final static width.

    - Correct RoPE behavior:
      We only apply RoPE to the current token (seq=1) using provided (cos, sin).
      Past K is assumed to already be RoPE-applied.

    KV cache update model
    ---------------------
    The attention wrapper returns only the delta KV for the current token:
        new_key:             (B, num_kv_heads, 1, head_dim)
        new_value:           (B, num_kv_heads, 1, head_dim)

    The host/runtime is responsible for writing this delta into the external KV cache buffer.
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
        return_type: Optional[str] = None,
    ):
        """
        `return_type` exists for HF compatibility across transformer versions.

        Policy:
        - If use_cache=True: always return (hidden_states, present_key_value)
        - Else: return either (hidden_states,) or hidden_states depending on HF version
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
        input_ln_cfg = qcfg.child("input_layernorm") if qcfg else None
        post_ln_cfg = qcfg.child("post_attention_layernorm") if qcfg else None

        # Force decode variant for children when config exists
        if attn_cfg is None:
            attn_cfg = PTQConfig(wrapper_variant="decode", strict_wrap=True)
        else:
            attn_cfg.wrapper_variant = "decode"

        # Quantized sub-modules ---------------------------------------------
        assert hasattr(fp_layer, "self_attn") and isinstance(
            fp_layer.self_attn, nn.Module
        )
        assert hasattr(fp_layer, "mlp") and isinstance(fp_layer.mlp, nn.Module)
        assert hasattr(fp_layer, "input_layernorm") and isinstance(
            fp_layer.input_layernorm, nn.Module
        )
        assert hasattr(fp_layer, "post_attention_layernorm") and isinstance(
            fp_layer.post_attention_layernorm, nn.Module
        )

        self.self_attn = PTQWrapper(
            fp_layer.self_attn,
            qcfg=attn_cfg,
            fp_name=f"{fp_name}.self_attn" if fp_name else None,
        )
        self.mlp = PTQWrapper(
            fp_layer.mlp,
            qcfg=mlp_cfg,
            fp_name=f"{fp_name}.mlp" if fp_name else None,
        )

        self.input_layernorm = PTQWrapper(
            fp_layer.input_layernorm,
            qcfg=input_ln_cfg,
            fp_name=f"{fp_name}.input_layernorm" if fp_name else None,
        )
        self.post_attention_layernorm = PTQWrapper(
            fp_layer.post_attention_layernorm,
            qcfg=post_ln_cfg,
            fp_name=f"{fp_name}.post_attention_layernorm" if fp_name else None,
        )

        self.obs_mlp_residual_out = self._make_obs("mlp_residual_out")

        # Cache static shape constants from config for assert/debugging.
        cfg = getattr(fp_layer.self_attn, "config", None)
        assert cfg is not None and hasattr(cfg, "max_position_embeddings")
        assert isinstance(cfg.max_position_embeddings, int)
        self.max_seq = cfg.max_position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, 1, D)
        attention_mask: Optional[torch.Tensor] = None,  # (B, 1, max_seq)
        position_ids: Optional[torch.LongTensor] = None,  # ignored in decode wrapper
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.LongTensor] = None,  # ignored
        position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,  # REQUIRED
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] | Tuple[
        torch.Tensor
    ] | torch.Tensor:
        if output_attentions:
            raise NotImplementedError(
                "QuantLlamaDecoderLayerDecode does not support output_attentions."
            )

        # Enforce decode contract: fully static single-token step.
        assert hidden_states.dim() == 3 and hidden_states.size(1) == 1, (
            "Decode expects hidden_states with shape (B, 1, D); "
            f"got {tuple(hidden_states.shape)}"
        )

        # No dynamic mask creation is allowed here.
        assert (
            attention_mask is not None
        ), "Decode expects an additive attention_mask input."
        assert attention_mask.dim() == 3 and attention_mask.size(1) == 1, (
            "Decode expects attention_mask with shape (B, 1, max_seq); "
            f"got {tuple(attention_mask.shape)}"
        )
        assert attention_mask.size(2) == self.max_seq, (
            f"Decode expects attention_mask width == max_seq ({self.max_seq}); "
            f"got {attention_mask.size(2)}"
        )

        # RoPE tables for the current token must be provided by the host/runtime.
        assert (
            position_embeddings is not None
        ), "Decode expects position_embeddings=(cos,sin)."
        cos, sin = position_embeddings
        assert cos.shape[:2] == (hidden_states.size(0), 1) and sin.shape[:2] == (
            hidden_states.size(0),
            1,
        ), (
            "Decode expects cos/sin with shape (B, 1, head_dim). "
            f"got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )

        # Past KV cache must be fully static and provided externally.
        assert (
            past_key_value is not None
        ), "Decode expects past_key_value=(past_k,past_v)."
        past_k, past_v = past_key_value
        assert past_k.dim() == 4 and past_v.dim() == 4, (
            "Decode expects past_k/past_v to be 4D tensors; "
            f"got past_k={tuple(past_k.shape)}, past_v={tuple(past_v.shape)}"
        )
        assert past_k.size(0) == hidden_states.size(0) and past_v.size(
            0
        ) == hidden_states.size(
            0
        ), "Batch mismatch between hidden_states and past_key_value."
        assert (
            past_k.size(2) == self.max_seq - 1 and past_v.size(2) == self.max_seq - 1
        ), (
            f"Decode expects past sequence length == max_seq-1 ({self.max_seq - 1}); "
            f"got past_k={past_k.size(2)}, past_v={past_v.size(2)}"
        )

        # ─── Attention block ────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_out = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )

        if use_cache:
            hidden_states_attn, present_key_value = attn_out  # delta KV
        else:
            hidden_states_attn = attn_out
            present_key_value = None

        hidden_states = residual + hidden_states_attn

        # ─── MLP block ─────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # Residual add after MLP.
        hidden_states = residual + hidden_states
        hidden_states = self._fq(hidden_states, self.obs_mlp_residual_out)

        # Return type policy:
        # - If use_cache: always return (hidden_states, present_key_value)
        # - Else: match HF compatibility as configured
        if use_cache:
            assert present_key_value is not None
            return hidden_states, present_key_value

        if self.return_type == "tuple":
            return (hidden_states,)
        if self.return_type == "tensor":
            return hidden_states
        raise RuntimeError("Invalid return_type configuration.")

    def _all_observers(self):
        # No local observers other than the final residual observer; recurse into children.
        yield from self.self_attn._all_observers()
        yield from self.mlp._all_observers()
        yield self.obs_mlp_residual_out
