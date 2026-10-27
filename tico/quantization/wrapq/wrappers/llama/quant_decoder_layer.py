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

from typing import cast, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn

from transformers.cache_utils import Cache

if TYPE_CHECKING:
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    from tico.quantization.wrapq.wrappers.llama.quant_attention import (
        QuantLlamaAttention,
    )


from tico.quantization.config.llama_attention import get_llama_attention_options
from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.utils.utils import join_name
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaDecoderLayerDecodeExportAdapter,
    LlamaDecoderLayerPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.llama.position import (
    build_causal_mask,
    combine_binary_attention_mask,
    is_binary_attention_mask,
    make_rotary_embedding,
    position_embeddings as make_position_embeddings,
)
from tico.quantization.wrapq.wrappers.llama.static_inputs import (
    capture_rope_state,
    prepare_llama_static_inputs,
)
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.llama.modeling_llama.LlamaDecoderLayer")
class QuantLlamaDecoderLayer(QuantModuleBase):
    """
    Unified quant wrapper for HF `LlamaDecoderLayer`.

    Design goals
    ------------
    - Keep a single HF-compatible forward for both prefill and decode.
    - Preserve calibration/runtime semantics in the main wrapper.
    - Move export-specific static contracts to thin export adapters.
    - Share the same Llama attention implementation profile used by
      `QuantLlamaAttention` for RoPE table generation.
    """

    def __init__(
        self,
        fp_layer: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
        return_type: Optional[str] = None,
        layer_idx: Optional[int] = None,
        rotary_emb: Optional[nn.Module] = None,
    ):
        """
        Initialize the quantized decoder-layer wrapper.

        `return_type` is needed because different versions of
        `transformers` wrap decoder output in different containers: a plain
        tensor or a tuple. Pass the original model-level ``rotary_emb`` for
        customized rotary state; otherwise the complete layer config is used.
        """
        fp_layer = cast("LlamaDecoderLayer", fp_layer)
        self.return_type = return_type
        if self.return_type is None:
            import transformers

            v = tuple(map(int, transformers.__version__.split(".")[:2]))
            self.return_type = "tensor" if v >= (4, 54) else "tuple"
        assert self.return_type is not None

        super().__init__(qcfg, fp_name=fp_name)

        self.attn_options = get_llama_attention_options(self.qcfg)
        self.layer_idx = layer_idx

        attn_cfg = qcfg.child("self_attn") if qcfg else None
        mlp_cfg = qcfg.child("mlp") if qcfg else None
        input_ln_cfg = qcfg.child("input_layernorm") if qcfg else None
        post_ln_cfg = qcfg.child("post_attention_layernorm") if qcfg else None

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
            fp_name=join_name(fp_name, "self_attn"),
        )
        if hasattr(self.self_attn, "wrapped") and hasattr(
            self.self_attn.wrapped, "layer_idx"
        ):
            cast("QuantLlamaAttention", self.self_attn.wrapped).layer_idx = layer_idx
        self.mlp = PTQWrapper(
            fp_layer.mlp,
            qcfg=mlp_cfg,
            fp_name=join_name(fp_name, "mlp"),
        )
        self.input_layernorm = PTQWrapper(
            fp_layer.input_layernorm,
            qcfg=input_ln_cfg,
            fp_name=join_name(fp_name, "input_layernorm"),
        )
        self.post_attention_layernorm = PTQWrapper(
            fp_layer.post_attention_layernorm,
            qcfg=post_ln_cfg,
            fp_name=join_name(fp_name, "post_attention_layernorm"),
        )

        self.obs_mlp_residual_out = self._make_obs("mlp_residual_out")
        self.obs_attn_mask = self._make_obs("attn_mask")
        self.obs_cos = self._make_obs("cos")
        self.obs_sin = self._make_obs("sin")

        cfg = fp_layer.self_attn.config
        assert hasattr(cfg, "max_position_embeddings")
        assert isinstance(cfg.max_position_embeddings, int)
        self.max_seq = cfg.max_position_embeddings
        source = (
            rotary_emb
            if rotary_emb is not None
            else getattr(fp_layer, "rotary_emb", None)
        )
        self.rotary_emb = make_rotary_embedding(cfg, source)
        capture_rope_state(self, cfg, self.rotary_emb)

    def _build_static_export_inputs(self, max_seq_len: int):
        """Support direct layer export without calling eager input builders."""
        return prepare_llama_static_inputs(self, max_seq_len)

    def _build_causal_mask(
        self,
        *,
        q_len: int,
        k_len: int,
        past_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build the [1, Q, K] mask for eager execution or an explicit export shape."""
        prepared = getattr(self, "_static_export_inputs", None)
        if prepared is not None:
            return prepared.mask(q_len, k_len, past_len, device=device)
        if torch.compiler.is_compiling():
            raise ValueError(
                "Static layer export requires an explicit mask or prepared_static_export()."
            )
        return build_causal_mask(
            q_len=q_len,
            k_len=k_len,
            past_len=past_len,
            device=device,
            fill_value=float(self.qcfg.attention_mask_fill_value),
        )

    def _slice_rope(
        self,
        *,
        start: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate pretrained RoPE for the requested range without a full table."""
        prepared = getattr(self, "_static_export_inputs", None)
        if prepared is not None:
            return prepared.rope(start, seq_len, device=device, dtype=dtype)
        if torch.compiler.is_compiling():
            raise ValueError(
                "Static layer export requires explicit RoPE or prepared_static_export()."
            )
        positions = torch.arange(start, start + seq_len, device=device).unsqueeze(0)
        return make_position_embeddings(
            self,
            self.self_attn.wrapped.config,
            torch.empty((), device=device, dtype=dtype),
            positions,
            pre_negated_sin=self.attn_options.rope == "pre_negated_sin",
        )

    def _get_past_len(
        self,
        past_key_value: Optional[Cache | Tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        """
        Return the cached sequence length for this layer.
        """
        if past_key_value is None:
            return 0
        if isinstance(past_key_value, Cache):
            return past_key_value.get_seq_length()

        past_k, past_v = past_key_value
        if past_k is None or past_v is None:
            return 0

        return int(past_k.shape[2])

    def _normalize_position_embeddings(
        self,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        past_key_value: Optional[Cache | Tuple[torch.Tensor, torch.Tensor]],
        position_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return position embeddings that match this wrapper's RoPE convention.
        """
        prepared = getattr(self, "_static_export_inputs", None)
        if position_embeddings is None and prepared is not None:
            if position_ids is not None:
                raise ValueError(
                    "Provide explicit RoPE tensors for custom static positions."
                )
            cos, sin = prepared.rope(
                self._get_past_len(past_key_value),
                hidden_states.size(1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        elif position_embeddings is None:
            if torch.compiler.is_compiling():
                raise ValueError(
                    "Static layer export requires explicit RoPE or prepared_static_export()."
                )
            if position_ids is None:
                past_len = self._get_past_len(past_key_value)
                position_ids = torch.arange(
                    past_len,
                    past_len + hidden_states.size(1),
                    device=hidden_states.device,
                ).unsqueeze(0)
            cos, sin = make_position_embeddings(
                self,
                self.self_attn.wrapped.config,
                hidden_states,
                position_ids,
                pre_negated_sin=self.attn_options.rope == "pre_negated_sin",
            )
        else:
            cos, sin = position_embeddings
        return self._fq(cos, self.obs_cos), self._fq(sin, self.obs_sin)

    def _normalize_attention_mask(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache | Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return an additive attention mask usable with per-head logits.

        Supported cases:
        - None: build a causal mask for the current query and key lengths.
        - bool/integer mask: combine nonzero keep values with the causal mask.
        - additive mask: use as-is.
        """
        q_len = hidden_states.size(1)
        past_len = self._get_past_len(past_key_value)
        k_len = past_len + q_len

        if attention_mask is None:
            mask = self._build_causal_mask(
                q_len=q_len, k_len=k_len, past_len=past_len, device=device
            )
            return self._fq(mask, self.obs_attn_mask)

        if is_binary_attention_mask(attention_mask):
            causal_mask = self._build_causal_mask(
                q_len=q_len, k_len=k_len, past_len=past_len, device=device
            )
            mask = combine_binary_attention_mask(
                causal_mask,
                attention_mask,
                fill_value=float(self.qcfg.attention_mask_fill_value),
            )
            return self._fq(mask, self.obs_attn_mask)

        return self._fq(attention_mask, self.obs_attn_mask)

    @staticmethod
    def _unpack_attn_outputs(
        attn_out,
        *,
        output_attentions: bool,
        use_cache: bool,
    ) -> tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[Cache | Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Normalize attention outputs into hidden, attention, and cache values.
        """
        if not isinstance(attn_out, tuple):
            return attn_out, None, None

        idx = 0
        hidden_states_attn = attn_out[idx]
        idx += 1

        attn_weights = None
        if output_attentions:
            attn_weights = attn_out[idx]
        idx += 1

        present_key_value = None
        if use_cache:
            present_key_value = attn_out[idx]

        return hidden_states_attn, attn_weights, present_key_value

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache | Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        return_type: Optional[str] = None,
        **kwargs,
    ):
        """
        Run the quantized decoder layer.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attention_mask = self._normalize_attention_mask(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            device=hidden_states.device,
        )
        position_embeddings = self._normalize_position_embeddings(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            position_ids=position_ids,
        )

        attn_out = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states_attn, attn_weights, present_key_value = self._unpack_attn_outputs(
            attn_out,
            output_attentions=bool(output_attentions),
            use_cache=bool(use_cache),
        )

        hidden_states = residual + hidden_states_attn

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self._fq(residual + hidden_states, self.obs_mlp_residual_out)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)  # type: ignore[assignment]
        if use_cache:
            outputs += (present_key_value,)  # type: ignore[assignment]

        result_type = self.return_type if return_type is None else return_type
        if result_type == "tuple":
            return outputs
        if result_type == "tensor":
            return hidden_states
        raise RuntimeError("Invalid return_type configuration.")

    def _all_observers(self):
        yield from (
            self.obs_attn_mask,
            self.obs_cos,
            self.obs_sin,
            self.obs_mlp_residual_out,
        )

    def as_export_module(self, mode: ExportMode = "prefill", *, return_kv: bool = True):
        """
        Return a decoder-layer export adapter for the requested mode.
        """
        if mode == "prefill":
            return LlamaDecoderLayerPrefillExportAdapter(self, return_kv=return_kv)
        if mode in ("decode", "append_prefill"):
            return LlamaDecoderLayerDecodeExportAdapter(self, return_kv=return_kv)
        raise ValueError(f"Unsupported export mode: {mode!r}")
