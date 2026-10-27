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

"""Binary Llama masks preserve causality and padding across integer dtypes."""

import unittest
from typing import cast

import numpy as np

import tico

import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime

from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.quant_attention import QuantLlamaAttention
from tico.quantization.wrapq.wrappers.llama.quant_decoder_layer import (
    QuantLlamaDecoderLayer,
)
from tico.quantization.wrapq.wrappers.llama.quant_model import QuantLlamaModel
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase

from test.quantization.quant_spec_helpers import make_affine_ptq_config


@unittest.skipUnless(has_transformers_for("llama"), "requires Transformers Llama")
class TestLlamaAttentionMasks(unittest.TestCase):
    @staticmethod
    def _wrapper(kind: str, profile: str) -> QuantModuleBase:
        from transformers import LlamaConfig, LlamaModel

        torch.manual_seed(0)
        config = LlamaConfig(
            hidden_size=8,
            intermediate_size=16,
            vocab_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            attention_dropout=0.0,
            max_position_embeddings=8,
        )
        config._attn_implementation = "eager"
        model = QuantLlamaModel(
            LlamaModel(config),
            qcfg=make_affine_ptq_config(model_args={"profile": profile}),
        ).eval()
        layer = cast(QuantLlamaDecoderLayer, model.layers[0].wrapped)
        if kind == "attention":
            return cast(QuantLlamaAttention, layer.self_attn.wrapped)
        if kind == "layer":
            return layer
        return model

    @staticmethod
    def _forward(wrapper: QuantModuleBase, mask: torch.Tensor):
        hidden = torch.sin(torch.arange(24).float() * 0.17).reshape(1, 3, 8)
        if isinstance(wrapper, QuantLlamaModel):
            result = wrapper(
                inputs_embeds=hidden,
                attention_mask=mask,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            return result.last_hidden_state, result.attentions[0]
        if isinstance(wrapper, QuantLlamaDecoderLayer):
            result = wrapper(
                hidden,
                attention_mask=mask,
                use_cache=True,
                output_attentions=True,
                return_type="tuple",
            )
            return result[:2]
        phase = torch.arange(12).float().reshape(1, 3, 4) * 0.1
        sin = phase.sin()
        if cast(QuantLlamaAttention, wrapper).attn_options.rope == "pre_negated_sin":
            sin[..., :2] = -sin[..., :2]
        return wrapper(hidden, (phase.cos(), sin), attention_mask=mask, use_cache=True)[
            :2
        ]

    def _check_binary_masks(self, kind: str) -> None:
        # Nonzero integer values mean "keep", including values other than one.
        token_mask = torch.tensor([[3, -1, 0]])
        masks = (
            token_mask,
            token_mask[:, None, :].expand(1, 3, 3),
            token_mask[:, None, None, :].expand(1, 1, 3, 3),
        )
        dtypes = (
            torch.bool,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        )
        for profile in ("npu_export", "reference_eval"):
            wrapper = self._wrapper(kind, profile)
            for quantized in (False, True):
                if quantized:
                    wrapper.enable_calibration()
                    with torch.no_grad():
                        self._forward(wrapper, token_mask.bool())
                    wrapper.freeze_qparams()
                with torch.no_grad():
                    expected_output, expected_weights = self._forward(
                        wrapper, token_mask.bool()
                    )
                    self.assertEqual(torch.count_nonzero(expected_weights[..., 2]), 0)
                    self.assertEqual(
                        torch.count_nonzero(expected_weights[..., 0, 1:]), 0
                    )
                    for mask in masks:
                        for dtype in dtypes:
                            with self.subTest(
                                kind=kind,
                                profile=profile,
                                quantized=quantized,
                                rank=mask.ndim,
                                dtype=dtype,
                            ):
                                output, weights = self._forward(wrapper, mask.to(dtype))
                                torch.testing.assert_close(
                                    output, expected_output, atol=0, rtol=0
                                )
                                torch.testing.assert_close(
                                    weights, expected_weights, atol=0, rtol=0
                                )

    def test_attention_binary_masks(self):
        self._check_binary_masks("attention")

    def test_layer_binary_masks(self):
        self._check_binary_masks("layer")

    def test_model_binary_masks(self):
        self._check_binary_masks("model")

    def test_integer_mask_circle_matches_boolean_mask(self):
        wrapper = cast(QuantLlamaAttention, self._wrapper("attention", "npu_export"))
        adapter = wrapper.as_export_module("prefill", return_kv=False).eval()
        hidden = torch.sin(torch.arange(24).float() * 0.17).reshape(1, 3, 8)
        phase = torch.arange(12).float().reshape(1, 3, 4) * 0.1
        sin = phase.sin()
        sin[..., :2] = -sin[..., :2]
        rope = (phase.cos(), sin)
        mask = torch.tensor([[3, -1, 0]], dtype=torch.int32)
        with torch.no_grad():
            expected = adapter(hidden, rope, mask.bool()).numpy()
        converted = tico.convert(adapter, (hidden, rope, mask), strict=False)
        document = CircleDocument.from_bytes(converted.circle_binary)
        document.verify()
        runtime = CircleReferenceRuntime(document)
        mask_tensor = runtime.program.tensor(runtime.program.inputs[-1])
        self.assertEqual(mask_tensor.dtype, np.dtype("int32"))
        self.assertEqual(tuple(mask_tensor.shape), (1, 3))
        self.assertIsNone(mask_tensor.contract.quantization)
        result = runtime.run(
            [hidden.numpy(), rope[0].numpy(), rope[1].numpy(), mask.numpy()]
        )
        # Float-only attention accumulates in float32 in both implementations.
        np.testing.assert_allclose(result.outputs[0], expected, atol=1e-6, rtol=1e-5)

    def test_float_masks_remain_additive(self):
        for kind in ("attention", "layer", "model"):
            for profile in ("npu_export", "reference_eval"):
                with self.subTest(kind=kind, profile=profile):
                    wrapper = self._wrapper(kind, profile)
                    with torch.no_grad():
                        _, weights = self._forward(wrapper, torch.zeros(1, 3, 3))
                    # An explicitly supplied additive mask owns causality. Zero
                    # bias permits future and padded positions to receive weight.
                    self.assertGreater(weights[..., 0, 1:].sum().item(), 0)
                    self.assertGreater(weights[..., 2].sum().item(), 0)
