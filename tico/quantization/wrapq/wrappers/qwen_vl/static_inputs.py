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

"""Bounded full-text mRoPE inputs prepared before tracing Qwen3-VL."""

import copy

import torch

from tico.quantization.wrapq.utils.static_export import validate_static_length
from torch import nn


class Qwen3VLStaticTextInputs(nn.Module):
    """Own raw static buffers without owning quantized weights or observers."""

    position_ids: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    attention_mask: torch.Tensor

    def __init__(self, positions, cos, sin, fill_value):
        super().__init__()
        self.max_seq_len = int(positions.shape[-1])
        if positions.shape != (3, 1, self.max_seq_len):
            raise ValueError("Qwen3-VL static positions must have shape [3, 1, S].")
        if cos.shape[:2] != (1, self.max_seq_len) or cos.shape != sin.shape:
            raise ValueError("Qwen3-VL static RoPE tables must have shape [1, S, D].")
        self.register_buffer(
            "position_ids", positions.detach().clone(), persistent=False
        )
        self.register_buffer("cos", cos.detach().clone(), persistent=False)
        self.register_buffer("sin", sin.detach().clone(), persistent=False)
        mask = torch.full(
            (1, 1, self.max_seq_len, self.max_seq_len),
            fill_value,
            device=cos.device,
            dtype=torch.float32,
        ).triu_(1)
        self.register_buffer("attention_mask", mask, persistent=False)


@torch.no_grad()
def prepare_qwen3_vl_static_text_inputs(owner, max_seq_len):
    """Build fixed-grid tables using an immutable rotary snapshot or legacy tables.

    Never call the live eager rotary module. A fresh copy of the snapshot is used
    for each preparation because some rotary implementations update frequencies
    when called with a longer context.
    """
    validate_static_length(max_seq_len, int(owner.config.max_position_embeddings))
    parameter = next(owner.parameters())
    device = parameter.device
    positions = getattr(owner, "position_ids_template", None)
    cos = getattr(owner, "cos_template", None)
    sin = getattr(owner, "sin_template", None)
    if (
        isinstance(positions, torch.Tensor)
        and isinstance(cos, torch.Tensor)
        and isinstance(sin, torch.Tensor)
    ):
        if positions.shape[-1] < max_seq_len or cos.shape[1] < max_seq_len:
            raise ValueError("Legacy Qwen3-VL tables do not cover the export profile.")
        positions = positions[..., :max_seq_len]
        cos, sin = cos[:, :max_seq_len], sin[:, :max_seq_len]
    else:
        snapshot = getattr(owner, "_export_rotary_emb", None)
        if snapshot is None:
            raise ValueError(
                "This intermediate Qwen3-VL checkpoint has neither legacy tables "
                "nor a frozen export rotary snapshot. Use a pre-PR checkpoint or "
                "regenerate it with the static-input isolation code."
            )
        positions = owner._compute_3d_position_ids(
            device=device,
            visual_start_idx=owner._get_visual_start_idx(owner.qcfg),
            seq_len=max_seq_len,
            thw=owner._get_vision_grid_thw(owner.qcfg),
            spatial_merge_size=owner._get_spatial_merge_size(owner.qcfg),
        )
        rotary = copy.deepcopy(snapshot).to(device)
        # The historical full-text templates used the default float32 dummy.
        cos, sin = rotary(torch.empty(0, device=device, dtype=torch.float32), positions)
    return Qwen3VLStaticTextInputs(
        positions, cos, sin, float(owner.qcfg.attention_mask_fill_value)
    )
