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

"""Fold Gemma4 PLE scales into floating-point tables before quantization.

This module intentionally depends only on PyTorch. It must run on a freshly
loaded model, before any quantization algorithm is prepared or calibrated.
"""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


# Bound validation temporaries, not the table itself. One unusually wide row is
# the minimum chunk; ordinary E2B rows are far smaller than this bound.
_MAX_CHUNK_ELEMENTS = 1 << 20


class FusedGemma4PLEEmbedding(nn.Embedding):
    """An ordinary lookup whose weight already contains the PLE scale.

    The class, rather than an ephemeral flag, carries the no-MUL contract through
    deepcopy and whole-module checkpoints. Reconstruct this architecture before
    loading a folded state_dict. ``embed_scale`` remains an identity buffer for
    the existing Gemma4 export/host interfaces, but is not used by forward.
    """

    def __init__(self, source: nn.Embedding):
        super().__init__(
            num_embeddings=source.num_embeddings,
            embedding_dim=source.embedding_dim,
            padding_idx=source.padding_idx,
            max_norm=source.max_norm,
            norm_type=source.norm_type,
            scale_grad_by_freq=source.scale_grad_by_freq,
            sparse=source.sparse,
            _weight=source.weight,
        )
        # nn.Embedding(_weight=...) shares storage but wraps a new Parameter.
        # Preserve the original Parameter identity, device and requires_grad too.
        self.weight = source.weight
        self.register_buffer(
            "embed_scale", torch.ones_like(source.embed_scale), persistent=False
        )
        self.train(source.training)


def gemma4_ple_scale_fusion_enabled(model_args: Mapping[str, Any]) -> bool:
    """Read the default-on setting without treating strings as booleans."""
    if not isinstance(model_args, Mapping):
        raise TypeError("model_args must be a mapping.")
    text = model_args.get("text", {})
    if not isinstance(text, Mapping):
        raise TypeError("model_args.text must be a mapping.")
    enabled = text.get("ple_embedding_scale_fusion", True)
    if not isinstance(enabled, bool):
        raise TypeError("model_args.text.ple_embedding_scale_fusion must be a bool.")
    return enabled


def _storage_key(tensor: torch.Tensor) -> tuple[torch.device, int]:
    """Identify shared storage, including aliases with different offsets."""
    return tensor.device, tensor.untyped_storage().data_ptr()


def _effective_scale(module: nn.Embedding, name: str) -> torch.Tensor:
    """Validate a standard, materialized Gemma4 PLE table and scalar."""
    if type(module).__name__ != "Gemma4TextScaledWordEmbedding":
        raise TypeError(f"{name}: expected Gemma4TextScaledWordEmbedding.")
    weight = module.weight
    if (
        weight.is_meta
        or not weight.is_floating_point()
        or weight.layout != torch.strided
        or weight.ndim != 2
        or weight.numel() == 0
        or not weight.is_contiguous()
    ):
        raise ValueError(
            f"{name}: expected a nonempty materialized, contiguous FP table."
        )
    if module.max_norm is not None:
        raise ValueError(f"{name}: scale folding does not support max_norm.")
    if any(
        getattr(module, attr, None)
        for attr in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
    ) or hasattr(module, "_hf_hook"):
        raise ValueError(
            f"{name}: remove embedding/offload hooks before scale folding."
        )
    scale = getattr(module, "embed_scale", None)
    if not isinstance(scale, torch.Tensor) or scale.is_meta or scale.numel() != 1:
        raise ValueError(f"{name}: embed_scale must be a materialized scalar tensor.")
    if scale.requires_grad:
        raise ValueError(f"{name}: embed_scale must be a constant, not trainable.")
    scale = scale.detach().to(device=weight.device, dtype=weight.dtype).reshape(())
    if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
        raise ValueError(
            f"{name}: embed_scale must be finite and positive in weight dtype."
        )
    return scale


@torch.no_grad()
def fuse_gemma4_ple_embedding_scale(model: nn.Module) -> tuple[str, ...]:
    """Fold only ``embed_tokens_per_layer`` in a fresh FP model, in place.

    Return the paths replaced during this call. Repeating this operation on an
    already fused FP model is a no-op. Registered aliases of a target table are
    rejected instead of mutating another consumer or duplicating a multi-GiB
    weight. All candidates, aliases and FP overflow are checked before mutation.

    Run before *any* prepare/calibrate/convert stage. Recognizable prepared
    WrapQ/HF quantized models are rejected; unmarked FP tensors cannot reveal
    whether an external quantizer has previously modified their values.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("Expected a fresh floating-point nn.Module.")
    modules = list(model.named_modules(remove_duplicate=False))
    for name, module in modules:
        # Every WrapQ QuantModuleBase owns qcfg and _mode. Do not import the
        # quantization stack just to inspect this raw-model lifecycle boundary.
        if (hasattr(module, "qcfg") and hasattr(module, "_mode")) or getattr(
            module, "is_quantized", False
        ):
            raise RuntimeError(
                f"{name or '<root>'}: PLE scale fusion requires a fresh FP model "
                "before prepare/calibration/quantization."
            )

    candidates: list[tuple[str, nn.Embedding, torch.Tensor]] = []
    found = False
    for name, module in modules:
        if not name or name.rsplit(".", 1)[-1] != "embed_tokens_per_layer":
            continue
        if module is None:
            continue
        found = True
        if isinstance(module, FusedGemma4PLEEmbedding):
            continue
        if not isinstance(module, nn.Embedding):
            raise TypeError(f"{name}: expected a raw Gemma4 PLE embedding.")
        candidates.append((name, module, _effective_scale(module, name)))
    if not found:
        raise ValueError("No embed_tokens_per_layer module found in the FP model.")
    if not candidates:
        return ()

    owners: dict[tuple[torch.device, int], list[str]] = {}
    tensors = list(model.named_parameters(remove_duplicate=False))
    tensors.extend(model.named_buffers(remove_duplicate=False))
    for name, tensor in tensors:
        if tensor.is_meta or tensor.layout != torch.strided or tensor.numel() == 0:
            continue
        owners.setdefault(_storage_key(tensor), []).append(name)
    for name, module, scale in candidates:
        aliases = owners[_storage_key(module.weight)]
        if aliases != [f"{name}.weight"]:
            raise ValueError(f"{name}: PLE weight has shared storage: {aliases}.")
        rows = max(1, _MAX_CHUNK_ELEMENTS // module.weight.shape[1])
        for chunk in module.weight.split(rows, dim=0):
            if not bool(torch.isfinite(chunk * scale).all()):
                raise ValueError(f"{name}: folded PLE weight would contain NaN/Inf.")

    # Construct every replacement before changing any table. _weight= reuses the
    # existing storage; validation above never materializes a full scaled table.
    replacements = [FusedGemma4PLEEmbedding(module) for _, module, _ in candidates]
    for (name, module, scale), replacement in zip(candidates, replacements):
        module.weight.mul_(scale)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, replacement)
    return tuple(name for name, _, _ in candidates)
