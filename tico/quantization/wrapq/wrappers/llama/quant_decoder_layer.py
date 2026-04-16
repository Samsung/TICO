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

from tico.quantization.config.ptq import ExportMode, PTQConfig
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    LlamaDecoderLayerDecodeExportAdapter,
    LlamaDecoderLayerPrefillExportAdapter,
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
        self.obs_attn_mask = self._make_obs("attn_mask")
        self.obs_cos = self._make_obs("cos")
        self.obs_sin = self._make_obs("sin")

        cfg = fp_layer.self_attn.config
        assert hasattr(cfg, "max_position_embeddings")
        assert isinstance(cfg.max_position_embeddings, int)
        self.max_seq = cfg.max_position_embeddings
        # Static causal mask template
        mask = torch.full((1, 1, self.max_seq, self.max_seq), float("-120"))
        mask.triu_(1)
        self.register_buffer("causal_mask_template", mask, persistent=False)

        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        rotary = getattr(fp_layer, "rotary_emb", None)
        # Static RoPE (position_embeddings) templates
        # `rotray_emb` is normally inside `LlamaModel`.
        # 1) inv_freq, scaling
        if rotary is not None and hasattr(rotary, "inv_freq"):
            inv_freq = rotary.inv_freq.detach().float()
            attn_scaling = float(getattr(rotary, "attention_scaling", 1.0))
        else:
            rope_params = getattr(cfg, "rope_parameters", None)
            if (
                rope_params is not None
                and isinstance(rope_params, dict)
                and "rope_theta" in rope_params
            ):
                base = float(rope_params["rope_theta"])
            else:
                base = float(getattr(cfg, "rope_theta", 10000.0))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
            )
            attn_scaling = 1.0
        # 2) Create cos/sin: [max_seq, head_dim]
        pos = torch.arange(self.max_seq, dtype=torch.float32)  # [max_seq]
        freqs = torch.outer(pos, inv_freq)  # [max_seq, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq, head_dim]
        cos_t = emb.cos() * attn_scaling
        sin_t = emb.sin() * attn_scaling
        half_dim = head_dim // 2
        sin_t[..., :half_dim] = -sin_t[..., :half_dim]
        self.register_buffer(
            "rope_cos_template", cos_t.unsqueeze(0), persistent=False
        )  # [1, max_seq, head_dim]
        self.register_buffer(
            "rope_sin_template", sin_t.unsqueeze(0), persistent=False
        )  # [1, max_seq, head_dim]

    def _slice_rope(
        self,
        *,
        start: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert isinstance(self.rope_cos_template, torch.Tensor)
        assert isinstance(self.rope_sin_template, torch.Tensor)
        end = start + seq_len
        cos = self.rope_cos_template[:, start:end, :].to(device=device, dtype=dtype)
        sin = self.rope_sin_template[:, start:end, :].to(device=device, dtype=dtype)
        return cos, sin

    def _normalize_position_embeddings(
        self,
        *,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_embeddings is None:
            q_len = hidden_states.size(1)
            past_len = 0 if past_key_value is None else int(past_key_value[0].shape[2])
            cos, sin = self._slice_rope(
                start=past_len,
                seq_len=q_len,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            cos, sin = position_embeddings
        return self._fq(cos, self.obs_cos), self._fq(sin, self.obs_sin)

    def _normalize_attention_mask(
        self,
        *,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return an additive attention mask usable with per-head logits.

        Supported cases:
        - None: build a causal mask slice.
        - bool mask: convert to additive mask using 0 / -120.
        - additive mask: use as-is.
        """
        q_len = hidden_states.size(1)
        past_len = 0 if past_key_value is None else int(past_key_value[0].shape[2])
        k_len = past_len + q_len

        if attention_mask is None:
            assert isinstance(self.causal_mask_template, torch.Tensor)
            mask = self.causal_mask_template[..., :q_len, :k_len].to(device)
            mask = mask.squeeze(0)
            return self._fq(mask, self.obs_attn_mask)

        if attention_mask.dtype == torch.bool:
            additive = torch.zeros_like(attention_mask, dtype=torch.float32)
            additive = additive.masked_fill(~attention_mask, float("-120"))
            return self._fq(additive, self.obs_attn_mask)

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
        Optional[Tuple[torch.Tensor, torch.Tensor]],
    ]:
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
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
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

        if self.return_type == "tuple" or use_cache is True:
            return outputs
        if self.return_type == "tensor":
            return hidden_states
        raise RuntimeError("Invalid return_type configuration.")

    def _all_observers(self):
        yield from (self.obs_attn_mask, self.obs_cos, self.obs_sin)
        yield from self.self_attn._all_observers()
        yield from self.mlp._all_observers()
        yield self.obs_mlp_residual_out

    def as_export_module(self, mode: ExportMode = "prefill", *, return_kv: bool = True):
        if mode == "prefill":
            return LlamaDecoderLayerPrefillExportAdapter(self, return_kv=return_kv)
        if mode == "decode":
            return LlamaDecoderLayerDecodeExportAdapter(self, return_kv=return_kv)
        raise ValueError(f"Unsupported export mode: {mode!r}")
