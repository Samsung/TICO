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

"""LLaMA static input preparation independent of eager rotary execution."""

import copy
import math
from typing import Any

import torch
from torch import nn

from tico.quantization.wrapq.utils.static_export import validate_static_length
from tico.quantization.wrapq.wrappers.llama.position import make_rotary_embedding


ROPE_CONTRACT_VERSION = 2


def capture_rope_state(owner: nn.Module, config: Any, source=None) -> None:
    """Freeze the canonical, fully configured rotary implementation for export.

    A layer without a model-level source reconstructs the Hugging Face rotary
    implementation from the complete config, including RoPE scaling. Full-model
    wrappers pass their model-level source to every layer. This snapshot is never
    called directly: preparation evaluates a copy, so eager execution and dynamic
    frequency updates cannot change future exports.
    """
    snapshot = make_rotary_embedding(config, source).eval()
    owner.add_module("_export_rotary_emb", snapshot)
    owner._export_rope_contract_version = ROPE_CONTRACT_VERSION
    # Old table/snapshot names must not override the canonical source after an
    # explicit migration. Quantized parameters and observer state are untouched.
    for name in (
        "rope_cos_template", "rope_sin_template", "_export_rope_inv_freq",
        "_export_rope_attention_scaling",
    ):
        if hasattr(owner, name):
            delattr(owner, name)


def _owner_config(owner: nn.Module) -> Any:
    """Resolve configuration on a model or a standalone wrapped decoder."""
    config = getattr(owner, "config", None)
    return config if config is not None else owner.self_attn.wrapped.config


def _canonical_snapshot(owner: nn.Module) -> nn.Module:
    """Require the unified contract rather than reusing legacy unscaled tables."""
    snapshot = getattr(owner, "_export_rotary_emb", None)
    if (
        getattr(owner, "_export_rope_contract_version", None) != ROPE_CONTRACT_VERSION
        or not isinstance(snapshot, nn.Module)
    ):
        raise ValueError(
            "This LLaMA checkpoint predates the unified RoPE contract. Regenerate "
            "it, or explicitly call migrate_llama_rope_state(model) before eager "
            "evaluation/export. Legacy decoder tables may omit RoPE scaling and "
            "must not silently take precedence over the model-level source."
        )
    return snapshot


def _install_rope_source(owner: nn.Module, config: Any, source: nn.Module) -> None:
    """Install independent eager and static copies without touching quantization."""
    owner.rotary_emb = make_rotary_embedding(config, source)
    capture_rope_state(owner, config, source)


def migrate_llama_rope_state(model: nn.Module, *, rotary_emb=None) -> dict[str, Any]:
    """Explicitly migrate loaded LLaMA wrappers to one model-level RoPE source.

    Unified checkpoints reuse their immutable source. Older checkpoints rebuild
    from their recorded complete config, unless the caller supplies the original
    model-level rotary module (required for customized, non-configurable RoPE).
    Mutable eager modules and old tables are deliberately not used as provenance.
    No model weights, qparams, observer statistics, modes, or config are changed.
    Discard external KV caches created before migration and re-evaluate quality;
    this operation fixes positional semantics, not the old calibration data.
    """
    from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
        QuantLlamaDecoderLayer,
    )
    from tico.quantization.wrapq.wrappers.llama.quant_model import QuantLlamaModel

    modules = list(model.modules())
    for module in modules:
        if getattr(getattr(module, "_mode", None), "name", "") == "CALIB":
            raise ValueError("Finish calibration before migrating LLaMA RoPE state.")
        if getattr(module, "_static_export_inputs", None) is not None:
            raise ValueError("Cannot migrate RoPE while static export is prepared.")
    full_models = [m for m in modules if isinstance(m, QuantLlamaModel)]
    covered = {id(child) for m in full_models for child in m.modules()}
    roots = full_models + [
        m for m in modules
        if isinstance(m, QuantLlamaDecoderLayer) and id(m) not in covered
    ]
    if not roots:
        raise ValueError("No QuantLlamaModel or QuantLlamaDecoderLayer was found.")
    if rotary_emb is not None and len(roots) != 1:
        raise ValueError("An explicit rotary source requires exactly one LLaMA root.")

    # Resolve every source before modifying the model, so a bad config fails early.
    plans = []
    for owner in roots:
        config = _owner_config(owner)
        if rotary_emb is not None:
            source, origin = rotary_emb, "explicit_model_rotary"
        elif getattr(owner, "_export_rope_contract_version", None) == ROPE_CONTRACT_VERSION:
            source, origin = _canonical_snapshot(owner), "unified_snapshot"
        else:
            source, origin = None, "recorded_config"
        source = make_rotary_embedding(config, source)
        children = (
            [layer.wrapped for layer in owner.layers]
            if isinstance(owner, QuantLlamaModel) else []
        )
        if any(not isinstance(child, QuantLlamaDecoderLayer) for child in children):
            raise TypeError("Expected quantized LLaMA decoder layers for migration.")
        plans.append((owner, config, source, origin, children))

    result = []
    for owner, config, source, origin, children in plans:
        _install_rope_source(owner, config, source)
        for child in children:
            _install_rope_source(child, config, source)
        result.append({"source": origin, "decoder_layers": len(children) or 1})
    return {
        "contract_version": ROPE_CONTRACT_VERSION,
        "roots": result,
        "numerical_change": "Legacy unscaled decoder RoPE is intentionally replaced.",
    }


class LlamaStaticInputs(nn.Module):
    """Own bounded raw masks and RoPE tables outside the quantized wrapper."""

    def __init__(self, cos: torch.Tensor, sin: torch.Tensor, fill_value: float):
        super().__init__()
        if cos.ndim != 3 or cos.shape != sin.shape or cos.shape[0] != 1:
            raise ValueError("Static LLaMA RoPE tables must have shape [1, S, D].")
        if not math.isfinite(fill_value) or abs(fill_value) > torch.finfo(torch.float32).max:
            raise ValueError("Static attention mask fill value must be finite float32.")
        self.max_seq_len = int(cos.shape[1])
        self.register_buffer("cos", cos.detach().clone(), persistent=False)
        self.register_buffer("sin", sin.detach().clone(), persistent=False)
        mask = torch.full(
            (1, 1, self.max_seq_len, self.max_seq_len),
            fill_value, dtype=torch.float32, device=cos.device,
        ).triu_(1)
        self.register_buffer("attention_mask", mask, persistent=False)

    def rope(self, start: int, length: int, *, device, dtype):
        """Slice a fixed table without calling a rotary implementation."""
        end = start + length
        if start < 0 or length < 1 or end > self.max_seq_len:
            raise ValueError("Requested RoPE span exceeds the static export profile.")
        return (
            self.cos[:, start:end].to(device=device, dtype=dtype),
            self.sin[:, start:end].to(device=device, dtype=dtype),
        )

    def mask(self, query_length: int, key_length: int, start: int, *, device):
        """Slice the legacy [1, Q, K] additive mask."""
        if (
            start < 0 or query_length < 1 or key_length < 1
            or start + query_length > self.max_seq_len
            or key_length > self.max_seq_len
        ):
            raise ValueError("Requested mask exceeds the static export profile.")
        return self.attention_mask[:, 0, start : start + query_length, :key_length].to(device)


@torch.no_grad()
def prepare_llama_static_inputs(owner: nn.Module, max_seq_len: int) -> LlamaStaticInputs:
    """Prepare canonical static tables before tracing, never from live eager state.

    The snapshot includes the complete RoPE implementation, not only theta.
    A fresh copy handles length-dependent scaling using this export profile and
    leaves the saved source unchanged. Layer and full-model export use exactly
    this preparation path. Legacy checkpoints require explicit migration.
    """
    config = _owner_config(owner)
    validate_static_length(max_seq_len, int(config.max_position_embeddings))
    snapshot = _canonical_snapshot(owner)
    parameter = next(owner.parameters(), None)
    buffer = next(snapshot.buffers(), None)
    device = (
        parameter.device if parameter is not None else
        buffer.device if buffer is not None else torch.device("cpu")
    )
    rotary = copy.deepcopy(snapshot).to(device=device, dtype=torch.float32)
    positions = torch.arange(max_seq_len, device=device).unsqueeze(0)
    # Hugging Face LLaMA rotary uses hidden states only for device and dtype.
    cos, sin = rotary(torch.empty((), device=device, dtype=torch.float32), positions)
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    if cos.shape != (1, max_seq_len, head_dim) or sin.shape != cos.shape:
        raise ValueError("Canonical LLaMA rotary must return [1, S, head_dim] tables.")
    if not torch.isfinite(cos).all() or not torch.isfinite(sin).all():
        raise ValueError("Canonical LLaMA rotary produced non-finite values.")
    if owner.attn_options.rope == "pre_negated_sin":
        half = sin.shape[-1] // 2
        sin = torch.cat((-sin[..., :half], sin[..., half:]), dim=-1)
    return LlamaStaticInputs(cos, sin, float(owner.qcfg.attention_mask_fill_value))
