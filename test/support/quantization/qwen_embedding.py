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

"""Tiny real Qwen3-VL lifecycle shared by CPU embedding stage tests."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class AffineGrid:
    scale: np.ndarray
    zero_point: np.ndarray
    minimum: int
    maximum: int
    dtype: np.dtype
    axis: int | None

    @classmethod
    def from_observer(cls, observer):
        state = observer.require_frozen()
        return cls(
            state.scale.detach().cpu().numpy().copy(),
            state.zero_point.detach().cpu().numpy().copy(),
            observer.dtype.qmin,
            observer.dtype.qmax,
            np.dtype(str(observer.dtype)),
            observer.channel_axis,
        )

    def _parameters(self, rank):
        shape = [1] * rank
        if self.axis is not None:
            shape[self.axis] = self.scale.size
        return self.scale.reshape(shape), self.zero_point.astype(np.float32).reshape(
            shape
        )

    def quantize(self, values, *, ties_even=False):
        scale, zero_point = self._parameters(values.ndim)
        divided = np.asarray(values, dtype=np.float32) / scale
        # Independent arithmetic, with NumPy rint reserved for WrapQ's contract.
        if ties_even:
            rounded = np.rint(divided)
        else:
            fraction, integral = np.modf(divided)
            rounded = integral + np.copysign(np.abs(fraction) >= 0.5, divided)
        return np.clip(rounded + zero_point, self.minimum, self.maximum).astype(
            self.dtype
        )

    def dequantize(self, values):
        scale, zero_point = self._parameters(values.ndim)
        return (values.astype(np.float32) - zero_point) * scale

    def fake_quantize(self, values, *, ties_even=False):
        return self.dequantize(self.quantize(values, ties_even=ties_even))


@dataclass
class QwenEmbeddingFixture:
    owner: torch.nn.Module
    text: Any
    embedding: Any
    rotation: Any
    decoder: Any
    embedding_grid: AffineGrid
    weight_grid: AffineGrid
    weight_codes: np.ndarray
    rotation_input_grid: AffineGrid | None
    rotation_output_grid: AffineGrid | None
    rotation_weight_grid: AffineGrid | None
    rotation_weight_codes: np.ndarray | None
    decoder_input_grid: AffineGrid
    r1: torch.Tensor | None

    def lookup(self, ids):
        return self.weight_grid.dequantize(self.weight_codes)[ids]

    def text_values(self, ids, *, ties_even=False):
        return self.embedding_grid.fake_quantize(self.lookup(ids), ties_even=ties_even)

    def rotate(self, values, *, ties_even=False):
        if self.rotation is None:
            return self.embedding_grid.quantize(values, ties_even=ties_even)
        assert self.rotation_input_grid is not None
        assert self.rotation_weight_grid is not None
        assert self.rotation_weight_codes is not None
        assert self.rotation_output_grid is not None
        inputs = self.rotation_input_grid.fake_quantize(values, ties_even=ties_even)
        weight = self.rotation_weight_grid.dequantize(self.rotation_weight_codes)
        output = np.matmul(inputs.reshape(-1, weight.shape[1]), weight.T).reshape(
            values.shape
        )
        return self.rotation_output_grid.quantize(output, ties_even=ties_even)

    @property
    def output_grid(self):
        return self.rotation_output_grid or self.embedding_grid


@torch.random.fork_rng(devices=[])
def build_quantized_qwen(*, rotation=False):
    """Use real SpinQuant and PTQ prepare/calibrate/convert, with frozen imported grids."""
    from tico.quantization import convert, prepare
    from tico.quantization.config.builders import build_qwen3_vl_ptq_config
    from tico.quantization.config.qwen3_vl_spinquant import Qwen3VLSpinQuantConfig
    from tico.quantization.config.specs import affine
    from tico.quantization.wrapq.dtypes import DType
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLForConditionalGeneration,
    )

    torch.manual_seed(204)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=16,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=16,
            attention_bias=False,
            attention_dropout=0.0,
            use_cache=False,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
        ),
        vision_config=dict(
            hidden_size=8,
            intermediate_size=16,
            depth=1,
            num_heads=2,
            temporal_patch_size=1,
            patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=8,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        ),
        image_token_id=14,
        video_token_id=15,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    getattr(config, "text_config")._attn_implementation = "eager"
    getattr(config, "vision_config")._attn_implementation = "eager"
    model = Qwen3VLForConditionalGeneration(config).eval()
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    scales = torch.arange(1, 17, dtype=torch.float32) / 256
    zeros = torch.tensor(
        [5, 17, 64, 128, 193, 250, 1, 90, 15, 30, 60, 120, 180, 240, 2, 100],
        dtype=torch.int32,
    )
    codes = torch.tensor([0, 1, 3, 7, 20, 90, 200, 255], dtype=torch.uint8).expand(
        16, -1
    )
    with torch.no_grad():
        model.model.language_model.embed_tokens.weight.copy_(
            (codes.float() - zeros[:, None]) * scales[:, None]
        )
    r1 = None
    if rotation:
        hadamard = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
        r1 = torch.kron(torch.kron(hadamard, hadamard), hadamard) / np.sqrt(8.0)
        spin = prepare(
            model,
            Qwen3VLSpinQuantConfig(
                init_method="external",
                r1=r1,
                enable_r2=False,
                show_progress=False,
                fuse_deepstack_visual_outputs=True,
            ),
        )
        model = convert(spin).eval()
    model_args = {
        "profile": "npu_export",
        "vision": {
            "grid_thw": (1, 4, 4),
            "visual_start_idx": 2,
            "spatial_merge_size": 2,
        },
    }
    qcfg = build_qwen3_vl_ptq_config(
        num_vision_blocks=1,
        num_text_layers=1,
        num_deepstack_mergers=1,
        model_args=model_args,
        activation=affine(DType.int(16)),
        linear_weight=affine(DType.uint(8)),
        vision_patch_embed_weight=affine(DType.uint(8)),
        embedding_weight=affine(DType.uint(8)),
        lm_head_weight=affine(DType.uint(8)),
        spin_rotation_weight=affine(DType.int(16)),
        norm=affine(DType.int(16)),
        norm_weight=affine(DType.int(16)),
    )
    prepared = prepare(model, qcfg)
    qtext = prepared.wrapped.model.wrapped.language_model.wrapped
    embedding = qtext.embed_tokens.wrapped
    rotate = (
        qtext.rotate_embedding.wrapped if qtext.rotate_embedding is not None else None
    )
    decoder = qtext.layers[0].wrapped
    embedding.weight_obs.load_qparams(scales, zeros, lock=True)
    embedding.act_out_obs.load_qparams(torch.tensor(1 / 32), torch.tensor(0), lock=True)
    if rotate is not None:
        rotate.obs_act_in.load_qparams(torch.tensor(1 / 16), torch.tensor(0), lock=True)
        rotate.obs_act_out.load_qparams(
            torch.tensor(1 / 64), torch.tensor(0), lock=True
        )
    ids = torch.tensor([[0, 1, 14, 14, 14, 14, 2, 3]])
    pixels = torch.linspace(-1, 1, 16 * 12).reshape(1, 16, 12)
    with torch.no_grad():
        prepared(
            input_ids=ids,
            pixel_values=pixels,
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            attention_mask=torch.ones_like(ids),
            use_cache=False,
        )
        qtext.embed_tokens(torch.arange(16).reshape(1, 16))
        if rotate is not None:
            qtext.rotate_embedding(qtext.embed_tokens(torch.arange(16).reshape(1, 16)))
    quantized = convert(prepared).eval()
    wg = AffineGrid.from_observer(embedding.weight_obs)
    rwg = AffineGrid.from_observer(rotate.obs_weight) if rotate is not None else None
    return QwenEmbeddingFixture(
        quantized.wrapped,
        qtext,
        embedding,
        rotate,
        decoder,
        AffineGrid.from_observer(embedding.act_out_obs),
        wg,
        wg.quantize(embedding.module.weight.detach().numpy(), ties_even=True),
        AffineGrid.from_observer(rotate.obs_act_in) if rotate is not None else None,
        AffineGrid.from_observer(rotate.obs_act_out) if rotate is not None else None,
        rwg,
        (
            rwg.quantize(rotate.module.weight.detach().numpy(), ties_even=True)
            if rwg is not None and rotate is not None
            else None
        ),
        AffineGrid.from_observer(decoder.obs_act_in),
        r1,
    )
