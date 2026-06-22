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

"""PTQ wrapper for the dense Gemma4 E2B text model."""

from collections import UserDict
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.utils.utils import get_model_arg, join_name
from tico.quantization.wrapq.wrappers.gemma4.utils import assert_gemma4_e2b_no_moe
from tico.quantization.wrapq.wrappers.ptq_wrapper import PTQWrapper
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from tico.quantization.wrapq.wrappers.registry import try_register


@try_register("transformers.models.gemma4.modeling_gemma4.Gemma4TextModel")
class QuantGemma4TextModel(QuantModuleBase):
    """Quantization wrapper for the dense Gemma4 E2B text model.

    Full and sliding causal masks are built from bounded static templates owned
    by this wrapper. The forward path does not call Hugging Face mask factories.
    Masked entries use ``PTQConfig.attention_mask_fill_value`` instead of the
    minimum finite value of the activation dtype, which keeps affine observer
    ranges usable around unmasked attention logits.

    The static template capacity can be selected with
    ``PTQConfig.model_args["text"]["max_seq"]``. When it is omitted, the
    wrapper uses the smaller of the model context limit and 2048 tokens to avoid
    allocating an unexpectedly large quadratic mask.

    The E2B scope intentionally rejects MoE. Cache allocation and writes,
    sampling, and CPU/NPU orchestration remain runtime responsibilities.
    """

    force_export: bool = False
    _DEFAULT_STATIC_MAX_SEQ = 2048
    _SUPPORTED_LAYER_TYPES = frozenset(("full_attention", "sliding_attention"))

    def __init__(
        self,
        fp_model: nn.Module,
        *,
        qcfg: Optional[PTQConfig] = None,
        fp_name: Optional[str] = None,
    ):
        assert_gemma4_e2b_no_moe(fp_model)
        super().__init__(qcfg, fp_name=fp_name)

        self.module = fp_model
        self.config = fp_model.config
        self.padding_idx = getattr(fp_model, "padding_idx", None)
        self.vocab_size = getattr(fp_model, "vocab_size", self.config.vocab_size)
        self.unique_layer_types = tuple(sorted(set(self.config.layer_types)))
        self.hidden_size_per_layer_input = int(
            getattr(fp_model, "hidden_size_per_layer_input", 0) or 0
        )

        unsupported = set(self.unique_layer_types) - self._SUPPORTED_LAYER_TYPES
        if unsupported:
            raise ValueError(
                "Unsupported Gemma4 text layer types: " f"{sorted(unsupported)}."
            )

        self.embed_tokens = PTQWrapper(
            fp_model.embed_tokens,
            qcfg=qcfg.child("embed_tokens") if qcfg else None,
            fp_name=join_name(fp_name, "embed_tokens"),
        )
        self.layers = nn.ModuleList(
            [
                PTQWrapper(
                    layer,
                    qcfg=qcfg.child("layers").child(str(i)) if qcfg else None,
                    fp_name=join_name(fp_name, f"layers.{i}"),
                )
                for i, layer in enumerate(fp_model.layers)
            ]
        )
        self.norm = PTQWrapper(
            fp_model.norm,
            qcfg=qcfg.child("norm") if qcfg else None,
            fp_name=join_name(fp_name, "norm"),
        )
        self.rotary_emb = fp_model.rotary_emb
        self.gradient_checkpointing = bool(
            getattr(fp_model, "gradient_checkpointing", False)
        )

        self.embed_tokens_per_layer: Optional[nn.Module] = None
        self.per_layer_model_projection: Optional[nn.Module] = None
        self.per_layer_projection_norm: Optional[nn.Module] = None
        self.per_layer_input_scale = 1.0
        self.per_layer_model_projection_scale = 1.0
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = PTQWrapper(
                fp_model.embed_tokens_per_layer,
                qcfg=qcfg.child("embed_tokens_per_layer") if qcfg else None,
                fp_name=join_name(fp_name, "embed_tokens_per_layer"),
            )
            self.per_layer_model_projection = PTQWrapper(
                fp_model.per_layer_model_projection,
                qcfg=qcfg.child("per_layer_model_projection") if qcfg else None,
                fp_name=join_name(fp_name, "per_layer_model_projection"),
            )
            self.per_layer_projection_norm = PTQWrapper(
                fp_model.per_layer_projection_norm,
                qcfg=qcfg.child("per_layer_projection_norm") if qcfg else None,
                fp_name=join_name(fp_name, "per_layer_projection_norm"),
            )
            self.per_layer_input_scale = float(fp_model.per_layer_input_scale)
            self.per_layer_model_projection_scale = float(
                fp_model.per_layer_model_projection_scale
            )

        self.static_max_seq = self._resolve_static_max_seq()
        self._register_static_templates(fp_model)

        mk = self._make_obs
        self.obs_inputs_embeds = mk("inputs_embeds")
        self.obs_attention_masks = nn.ModuleDict(
            {
                layer_type: mk(f"attention_mask_{layer_type}")
                for layer_type in self.unique_layer_types
            }
        )
        self.obs_position_cos = nn.ModuleDict(
            {
                layer_type: mk(f"position_embeddings_{layer_type}_cos")
                for layer_type in self.unique_layer_types
            }
        )
        self.obs_position_sin = nn.ModuleDict(
            {
                layer_type: mk(f"position_embeddings_{layer_type}_sin")
                for layer_type in self.unique_layer_types
            }
        )

        self.obs_per_layer_token_inputs = None
        self.obs_per_layer_projection = None
        self.obs_per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            self.obs_per_layer_token_inputs = mk("per_layer_token_inputs")
            self.obs_per_layer_projection = mk("per_layer_projection")
            self.obs_per_layer_inputs = mk("per_layer_inputs")

    def _resolve_static_max_seq(self) -> int:
        """Resolve the capacity of static mask, position, and RoPE templates."""
        configured = get_model_arg(self.qcfg, "text", "max_seq", default=None)
        if configured is None:
            configured = get_model_arg(self.qcfg, "max_seq", default=None)

        model_capacity = int(self.config.max_position_embeddings)
        max_seq = (
            min(model_capacity, self._DEFAULT_STATIC_MAX_SEQ)
            if configured is None
            else int(configured)
        )
        if max_seq <= 0:
            raise ValueError(f"Gemma4 text max_seq must be positive, got {max_seq}.")
        if max_seq > model_capacity:
            raise ValueError(
                "Gemma4 text max_seq exceeds max_position_embeddings: "
                f"max_seq={max_seq}, model_capacity={model_capacity}."
            )
        return max_seq

    @staticmethod
    def _mask_template_name(layer_type: str) -> str:
        """Return the mask-template buffer name for a layer type."""
        return f"{layer_type}_attention_mask_template"

    @staticmethod
    def _cos_template_name(layer_type: str) -> str:
        """Return the cosine-template buffer name for a layer type."""
        return f"{layer_type}_cos_template"

    @staticmethod
    def _sin_template_name(layer_type: str) -> str:
        """Return the sine-template buffer name for a layer type."""
        return f"{layer_type}_sin_template"

    def _build_full_attention_mask_template(self, device: torch.device) -> torch.Tensor:
        """Build a bounded full causal mask template."""
        mask = torch.full(
            (1, 1, self.static_max_seq, self.static_max_seq),
            float(self.qcfg.attention_mask_fill_value),
            dtype=torch.float32,
            device=device,
        )
        return mask.triu_(1)

    def _build_sliding_attention_mask_template(
        self, device: torch.device
    ) -> torch.Tensor:
        """Build a bounded causal sliding-window mask template."""
        window = int(getattr(self.config, "sliding_window", 0) or 0)
        if window <= 0:
            raise ValueError(
                "Gemma4 sliding_attention requires a positive sliding_window, "
                f"got {window}."
            )

        query = torch.arange(self.static_max_seq, device=device).view(-1, 1)
        key = torch.arange(self.static_max_seq, device=device).view(1, -1)
        keep = (key <= query) & (key > query - window)

        mask = torch.full(
            (self.static_max_seq, self.static_max_seq),
            float(self.qcfg.attention_mask_fill_value),
            dtype=torch.float32,
            device=device,
        )
        mask.masked_fill_(keep, 0.0)
        return mask.unsqueeze(0).unsqueeze(0)

    def _register_static_templates(self, fp_model: nn.Module) -> None:
        """Register bounded masks, position ids, and per-layer-type RoPE tables."""
        embedding_weight = fp_model.embed_tokens.weight
        device = embedding_weight.device
        dtype = embedding_weight.dtype

        position_ids = torch.arange(
            self.static_max_seq, dtype=torch.long, device=device
        ).unsqueeze(0)
        self.register_buffer("position_ids_template", position_ids, persistent=False)

        for layer_type in self.unique_layer_types:
            if layer_type == "full_attention":
                mask = self._build_full_attention_mask_template(device)
            elif layer_type == "sliding_attention":
                mask = self._build_sliding_attention_mask_template(device)
            else:
                raise AssertionError(f"Unexpected layer type: {layer_type!r}.")
            self.register_buffer(
                self._mask_template_name(layer_type), mask, persistent=False
            )

        dummy = torch.empty(0, device=device, dtype=dtype)
        with torch.no_grad():
            for layer_type in self.unique_layer_types:
                cos, sin = self.rotary_emb(dummy, position_ids, layer_type)
                if cos.shape[:2] != (1, self.static_max_seq):
                    raise RuntimeError(
                        "Unexpected Gemma4 RoPE template shape: "
                        f"layer_type={layer_type!r}, shape={tuple(cos.shape)}."
                    )
                if sin.shape != cos.shape:
                    raise RuntimeError(
                        "Gemma4 RoPE sine and cosine shapes differ: "
                        f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}."
                    )
                self.register_buffer(
                    self._cos_template_name(layer_type), cos, persistent=False
                )
                self.register_buffer(
                    self._sin_template_name(layer_type), sin, persistent=False
                )

    @staticmethod
    def _is_torch_export_context() -> bool:
        """Return whether execution is inside a torch compile or export context."""
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        return bool(is_compiling()) if callable(is_compiling) else False

    def _requires_static_inputs(self) -> bool:
        """Return whether precomputed RoPE tables should be used."""
        return bool(self.force_export or self._is_torch_export_context())

    @staticmethod
    def _unwrap_fp_module(module: nn.Module) -> nn.Module:
        """Return the floating-point module hidden behind PTQWrapper layers."""
        wrapped = getattr(module, "wrapped", module)
        return getattr(wrapped, "module", wrapped)

    @staticmethod
    def _past_seen_tokens(past_key_values: Any) -> int:
        """Return the number of tokens already stored in the cache."""
        if past_key_values is None:
            return 0
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        return int(get_seq_length()) if callable(get_seq_length) else 0

    @staticmethod
    def _normalize_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
        """Normalize position ids to shape ``(B, S)``."""
        if position_ids.dim() == 1:
            return position_ids.unsqueeze(0)
        if position_ids.dim() != 2:
            raise ValueError(
                "Gemma4 position_ids must have rank 1 or 2, "
                f"got shape={tuple(position_ids.shape)}."
            )
        return position_ids

    def _make_position_ids(
        self,
        *,
        inputs_embeds: torch.Tensor,
        past_key_values: Any,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Create or validate position ids for the current model step."""
        batch_size, seq_len = inputs_embeds.shape[:2]
        if position_ids is not None:
            position_ids = self._normalize_position_ids(position_ids).to(
                device=inputs_embeds.device
            )
            if position_ids.size(-1) != seq_len:
                raise ValueError(
                    "position_ids length does not match inputs_embeds: "
                    f"position_ids={position_ids.size(-1)}, seq_len={seq_len}."
                )
            if position_ids.size(0) not in (1, batch_size):
                raise ValueError(
                    "position_ids batch must be 1 or match inputs_embeds: "
                    f"position_ids={position_ids.size(0)}, batch={batch_size}."
                )
            return position_ids.expand(batch_size, -1)

        start = self._past_seen_tokens(past_key_values)
        end = start + seq_len
        if end > self.static_max_seq:
            raise ValueError(
                "Gemma4 position range exceeds static_max_seq: "
                f"end={end}, static_max_seq={self.static_max_seq}."
            )
        return (
            self.position_ids_template[:, start:end]
            .to(device=inputs_embeds.device)
            .expand(batch_size, -1)
        )

    def _slice_mask_template(
        self,
        layer_type: str,
        *,
        q_len: int,
        kv_len: int,
        past_seen_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Slice a bounded static mask for the current query and KV spans."""
        row_end = past_seen_tokens + q_len
        if row_end > self.static_max_seq or kv_len > self.static_max_seq:
            raise ValueError(
                "Gemma4 attention span exceeds static_max_seq: "
                f"q_end={row_end}, kv_len={kv_len}, "
                f"static_max_seq={self.static_max_seq}."
            )
        template = getattr(self, self._mask_template_name(layer_type))
        return template[..., past_seen_tokens:row_end, :kv_len].to(
            device=device, dtype=dtype
        )

    @staticmethod
    def _normalize_explicit_mask_shape(
        mask: torch.Tensor,
        *,
        batch_size: int,
        q_len: int,
        kv_len: int,
    ) -> torch.Tensor:
        """Normalize an explicit mask to shape ``(B, 1, Q, K)``."""
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError(
                "Explicit Gemma4 masks must have rank 3 or 4, "
                f"got shape={tuple(mask.shape)}."
            )
        if mask.size(0) not in (1, batch_size):
            raise ValueError(
                "Explicit mask batch must be 1 or match inputs: "
                f"mask_batch={mask.size(0)}, batch={batch_size}."
            )
        if mask.size(1) != 1:
            raise ValueError(
                "Per-head Gemma4 masks are unsupported; expected head dim 1, "
                f"got shape={tuple(mask.shape)}."
            )
        if mask.size(-1) < kv_len:
            raise ValueError(
                "Explicit mask is shorter than the KV span: "
                f"mask_k={mask.size(-1)}, kv_len={kv_len}."
            )
        if mask.size(-1) > kv_len:
            mask = mask[..., :kv_len]
        if mask.size(-2) < q_len:
            raise ValueError(
                "Explicit mask has fewer query rows than required: "
                f"mask_q={mask.size(-2)}, q_len={q_len}."
            )
        if mask.size(-2) > q_len:
            mask = mask[..., -q_len:, :]
        return mask

    def _bounded_additive_mask(
        self,
        mask: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map an additive mask to zero or the configured finite fill value."""
        mask = mask.to(device=device, dtype=dtype)
        fill = torch.full_like(mask, float(self.qcfg.attention_mask_fill_value))
        return torch.where(mask < 0, fill, torch.zeros_like(mask))

    def _normalize_attention_mask_for_layer(
        self,
        layer_type: str,
        attention_mask: Optional[torch.Tensor],
        *,
        inputs_embeds: torch.Tensor,
        past_key_values: Any,
    ) -> torch.Tensor:
        """Build one bounded additive mask directly from static templates."""
        batch_size, q_len = inputs_embeds.shape[:2]
        past_seen_tokens = self._past_seen_tokens(past_key_values)
        kv_len = past_seen_tokens + q_len
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        causal_mask = self._slice_mask_template(
            layer_type,
            q_len=q_len,
            kv_len=kv_len,
            past_seen_tokens=past_seen_tokens,
            device=device,
            dtype=dtype,
        )
        if attention_mask is None:
            return causal_mask

        if attention_mask.dim() == 2:
            if attention_mask.size(0) != batch_size:
                raise ValueError(
                    "2D attention-mask batch does not match inputs: "
                    f"mask_batch={attention_mask.size(0)}, batch={batch_size}."
                )
            if attention_mask.size(1) == q_len and past_seen_tokens > 0:
                prefix = torch.ones(
                    batch_size,
                    past_seen_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat((prefix, attention_mask), dim=-1)
            if attention_mask.size(1) != kv_len:
                raise ValueError(
                    "2D attention-mask length does not match KV length: "
                    f"mask_len={attention_mask.size(1)}, kv_len={kv_len}."
                )

            keep = attention_mask.to(device=device) != 0
            padding_mask = torch.zeros(
                batch_size, 1, 1, kv_len, device=device, dtype=dtype
            ).masked_fill(
                ~keep[:, None, None, :],
                float(self.qcfg.attention_mask_fill_value),
            )
            return torch.clamp(
                causal_mask + padding_mask,
                min=float(self.qcfg.attention_mask_fill_value),
                max=0.0,
            )

        explicit = self._normalize_explicit_mask_shape(
            attention_mask,
            batch_size=batch_size,
            q_len=q_len,
            kv_len=kv_len,
        )
        if torch.is_floating_point(explicit):
            return self._bounded_additive_mask(explicit, device=device, dtype=dtype)

        keep = explicit.to(device=device).bool()
        return torch.zeros(keep.shape, device=device, dtype=dtype).masked_fill(
            ~keep, float(self.qcfg.attention_mask_fill_value)
        )

    def _create_attention_mask_mapping(
        self,
        *,
        attention_mask: Any,
        inputs_embeds: torch.Tensor,
        past_key_values: Any,
    ) -> dict[str, torch.Tensor]:
        """Create bounded masks for every layer type without HF mask factories."""
        if isinstance(attention_mask, Mapping):
            missing = [
                layer_type
                for layer_type in self.unique_layer_types
                if layer_type not in attention_mask
            ]
            if missing:
                raise KeyError(
                    "Gemma4 mask mapping is missing layer types: " f"{missing}."
                )
            return {
                layer_type: self._normalize_attention_mask_for_layer(
                    layer_type,
                    attention_mask[layer_type],
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                )
                for layer_type in self.unique_layer_types
            }

        # The complete TextModel is not the static NPU export unit. Preserve the
        # existing export contract and require CPU-provided masks in that mode.
        if self._requires_static_inputs():
            raise NotImplementedError(
                "QuantGemma4TextModel static/export mode requires static masks "
                "as a dict keyed by layer type."
            )

        return {
            layer_type: self._normalize_attention_mask_for_layer(
                layer_type,
                attention_mask,
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
            )
            for layer_type in self.unique_layer_types
        }

    def _observe_attention_mask_mapping(
        self, mapping: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Observe each bounded attention mask."""
        return {
            layer_type: self._fq(mask, self.obs_attention_masks[layer_type])
            for layer_type, mask in mapping.items()
        }

    def _make_position_embeddings(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        use_static_templates: bool,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Create or slice observed RoPE tables for each layer type."""
        batch_size, seq_len = hidden_states.shape[:2]
        if use_static_templates:
            start = self._past_seen_tokens(past_key_values)
            end = start + seq_len
            if end > self.static_max_seq:
                raise ValueError(
                    "Gemma4 RoPE span exceeds static_max_seq: "
                    f"end={end}, static_max_seq={self.static_max_seq}."
                )

        outputs = {}
        for layer_type in self.unique_layer_types:
            if use_static_templates:
                cos = getattr(self, self._cos_template_name(layer_type))[
                    :, start:end, :
                ]
                sin = getattr(self, self._sin_template_name(layer_type))[
                    :, start:end, :
                ]
                cos = cos.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                ).expand(batch_size, -1, -1)
                sin = sin.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                ).expand(batch_size, -1, -1)
            else:
                cos, sin = self.rotary_emb(hidden_states, position_ids, layer_type)
            outputs[layer_type] = (
                self._fq(cos, self.obs_position_cos[layer_type]),
                self._fq(sin, self.obs_position_sin[layer_type]),
            )
        return outputs

    def _reverse_input_ids_from_embeddings(
        self, inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Recover input ids from exact main embeddings for PLE compatibility."""
        embedding = self._unwrap_fp_module(self.embed_tokens)
        weight = embedding.weight.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        )
        scale = getattr(embedding, "embed_scale", None)
        if scale is not None:
            weight = weight * scale.to(
                device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        with torch.no_grad():
            matches = (inputs_embeds[:, :, None, :] == weight[None, None, :, :]).all(
                dim=-1
            )
            indices = matches.nonzero()
            expected = inputs_embeds.shape[0] * inputs_embeds.shape[1]
            if indices.size(0) != expected:
                raise RuntimeError(
                    "Gemma4 PLE could not recover input_ids from inputs_embeds. "
                    "Provide input_ids or explicit per_layer_inputs."
                )
            try:
                return indices[:, 2].view(inputs_embeds.shape[:2])
            except RuntimeError as exc:
                raise RuntimeError(
                    "Gemma4 PLE recovered a non-rectangular input-id layout."
                ) from exc

    def get_per_layer_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the token-identity component of per-layer embeddings."""
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("Per-layer embeddings are disabled for this config.")
        if self.embed_tokens_per_layer is None:
            raise RuntimeError("Gemma4 PLE embedding is not initialized.")
        if input_ids is None:
            if inputs_embeds is None:
                raise ValueError("inputs_embeds is required when input_ids is None.")
            input_ids = self._reverse_input_ids_from_embeddings(inputs_embeds)

        result = self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        if self.obs_per_layer_token_inputs is None:
            raise RuntimeError("Gemma4 PLE token observer is not initialized.")
        return self._fq(result, self.obs_per_layer_token_inputs)

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project and combine Gemma4 context-aware per-layer inputs."""
        if not self.hidden_size_per_layer_input:
            raise RuntimeError("Per-layer projection is disabled for this config.")
        projection = self.per_layer_model_projection
        projection_norm = self.per_layer_projection_norm
        if projection is None or projection_norm is None:
            raise RuntimeError("Gemma4 PLE projection modules are not initialized.")
        if self.obs_per_layer_projection is None or self.obs_per_layer_inputs is None:
            raise RuntimeError("Gemma4 PLE observers are not initialized.")

        projected = projection(inputs_embeds) * self.per_layer_model_projection_scale
        projected = projected.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        projected = projection_norm(projected)
        projected = self._fq(projected, self.obs_per_layer_projection)
        if per_layer_inputs is None:
            return self._fq(projected, self.obs_per_layer_inputs)
        combined = (projected + per_layer_inputs) * self.per_layer_input_scale
        return self._fq(combined, self.obs_per_layer_inputs)

    @staticmethod
    def _unwrap_layer_output(output: Any) -> tuple[torch.Tensor, Optional[Any]]:
        """Extract hidden states and optional cache output from a layer result."""
        if not isinstance(output, tuple):
            return output, None
        if not output:
            raise RuntimeError("Gemma4 decoder layer returned an empty tuple.")
        return output[0], output[1] if len(output) > 1 else None

    @staticmethod
    def _output_cls():
        """Return the Hugging Face Gemma4 text output class."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextModelOutputWithPast,
        )

        return Gemma4TextModelOutputWithPast

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Any = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: bool = True,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        **kwargs,
    ):
        """Run Gemma4 text inference with direct bounded mask construction."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds."
            )
        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError(
                "You cannot specify per_layer_inputs if input_ids is provided."
            )
        if per_layer_inputs is not None and not self.hidden_size_per_layer_input:
            raise ValueError("per_layer_inputs was provided, but PLE is disabled.")

        return_shared_kv_states = bool(kwargs.pop("return_shared_kv_states", False))
        output_hidden_states = bool(output_hidden_states)
        output_attentions = bool(output_attentions)
        position_ids_were_provided = position_ids is not None

        if use_cache is None:
            use_cache = bool(getattr(self.config, "use_cache", False))
        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            inputs_embeds = self._fq(inputs_embeds, self.obs_inputs_embeds)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            else:
                if self.obs_per_layer_token_inputs is None:
                    raise RuntimeError("Gemma4 PLE token observer is not initialized.")
                per_layer_inputs = self._fq(
                    per_layer_inputs, self.obs_per_layer_token_inputs
                )
            per_layer_inputs = self.project_per_layer_inputs(
                inputs_embeds, per_layer_inputs
            )

        position_ids = self._make_position_ids(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        mask_mapping = self._create_attention_mask_mapping(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
        )
        mask_mapping = self._observe_attention_mask_mapping(mask_mapping)

        hidden_states = inputs_embeds
        position_embeddings = self._make_position_embeddings(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_static_templates=(
                self._requires_static_inputs() and not position_ids_were_provided
            ),
        )

        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())
        if shared_kv_states is None:
            shared_kv_states = UserDict()

        all_hidden_states = () if output_hidden_states else None
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

            layer_type = self.config.layer_types[i]
            per_layer_input = (
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            )
            layer_output = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=mask_mapping[layer_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=bool(use_cache),
                **kwargs,
            )
            hidden_states, cache_output = self._unwrap_layer_output(layer_output)
            if cache_output is not None and hasattr(cache_output, "get_seq_length"):
                past_key_values = cache_output

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)  # type: ignore[operator]

        attentions = None
        if not return_dict:
            output: tuple[Any, ...] = (hidden_states,)
            if use_cache:
                output = output + (past_key_values,)
            if output_hidden_states:
                output = output + (all_hidden_states,)
            if output_attentions:
                output = output + (attentions,)
            if return_shared_kv_states:
                output = output + (shared_kv_states,)
            return output

        return self._output_cls()(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=attentions,
            shared_kv_states=(shared_kv_states if return_shared_kv_states else None),
        )

    def _all_observers(self) -> Iterable:
        """Return observers owned directly by this wrapper."""
        observers = [
            self.obs_inputs_embeds,
            *tuple(self.obs_attention_masks.values()),
            *tuple(self.obs_position_cos.values()),
            *tuple(self.obs_position_sin.values()),
        ]
        observers.extend(
            observer
            for observer in (
                self.obs_per_layer_token_inputs,
                self.obs_per_layer_projection,
                self.obs_per_layer_inputs,
            )
            if observer is not None
        )
        return tuple(observers)
