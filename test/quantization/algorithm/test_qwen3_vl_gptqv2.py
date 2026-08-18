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

"""
Tests for the Qwen3-VL GPTQv2 quantizer helpers.

These tests verify that:
  - The GPTQ class from qwen3_vl_gptq.gptq is used for both v1 and v2.
  - _build_gptq_objects initializes native_inp when gptq_v2=True.
  - Conv3d native inputs correctly populate dXXT.
"""

import unittest

import torch

from tico.quantization.algorithm.qwen3_vl_gptq.gptq import GPTQ
from tico.quantization.algorithm.qwen3_vl_gptq.quantizer import Qwen3VLGPTQQuantizer
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig


class TestQwen3VLGPTQv2Core(unittest.TestCase):
    """Test GPTQv2 core mechanics on Conv3d layers."""

    @torch.no_grad()
    def test_conv3d_native_inputs_populate_dXXT(self):
        """dXXT should be computed and non-zero when FP and quantized inputs differ."""
        layer = torch.nn.Conv3d(
            in_channels=2,
            out_channels=3,
            kernel_size=(2, 2, 2),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        gptq = GPTQ(layer)

        current = torch.randn(1, 2, 3, 3, 3)
        native = current + 0.125
        out = layer(current)

        gptq.native_inp = [native]
        gptq.add_batch(current, out)

        self.assertIsNotNone(gptq.dXXT)
        assert gptq.dXXT is not None
        self.assertEqual(gptq.dXXT.shape, gptq.H.shape)
        self.assertGreater(gptq.dXXT.abs().sum().item(), 0.0)


class TestQwen3VLGPTQv2QuantizerHelpers(unittest.TestCase):
    """Test Qwen3VLGPTQQuantizer helper methods."""

    def test_build_gptq_objects_default_config(self):
        """_build_gptq_objects should create GPTQ objects with native_inp=None for v1."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        self.assertIsInstance(gptq_objs["linear"], GPTQ)
        # For v1 (gptq_v2=False), native_inp should not be initialized as a list
        self.assertIsNone(gptq_objs["linear"].native_inp)

    def test_build_gptq_objects_gptqv2_config(self):
        """_build_gptq_objects should create GPTQ objects with native_inp=[] for v2."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(gptq_v2=True))
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        self.assertIsInstance(gptq_objs["linear"], GPTQ)
        # For v2 (gptq_v2=True), native_inp should be initialized as an empty list
        self.assertEqual(gptq_objs["linear"].native_inp, [])

    def test_assign_native_inputs(self):
        """_assign_native_inputs should copy FP inputs to GPTQ objects."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(gptq_v2=True))
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        fp_inputs = [torch.randn(2, 4), torch.randn(3, 4)]
        native_inputs = {"linear": fp_inputs}

        quantizer._assign_native_inputs(gptq_objs, native_inputs)

        self.assertEqual(len(gptq_objs["linear"].native_inp), 2)
        self.assertTrue(torch.allclose(gptq_objs["linear"].native_inp[0], fp_inputs[0]))
        self.assertTrue(torch.allclose(gptq_objs["linear"].native_inp[1], fp_inputs[1]))

    def test_resolve_weight_bits_default(self):
        """_resolve_weight_bits should return the config default when no override."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(weight_bits=4))
        bits = quantizer._resolve_weight_bits(
            quantizer.config,
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 4)

    def test_resolve_weight_bits_override_full_name(self):
        """_resolve_weight_bits should use full-name override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"model.layers.0.self_attn.q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_resolve_weight_bits_override_local_name(self):
        """_resolve_weight_bits should use local-name override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"self_attn.q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_resolve_weight_bits_override_suffix(self):
        """_resolve_weight_bits should use suffix override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_module_device(self):
        """_module_device should return the device of the module's parameters."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        layer = torch.nn.Linear(4, 3)
        device = quantizer._module_device(layer)
        self.assertEqual(device, layer.weight.device)

    def test_copy_original_model(self):
        """_copy_original_model should create a deep copy on CPU."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        orig_model = quantizer._copy_original_model(model)

        # Should be a different object
        self.assertIsNot(orig_model, model)
        # Weights should match
        self.assertTrue(torch.allclose(orig_model[0].weight, model[0].weight))
        # Modifying one should not affect the other
        model[0].weight.data.fill_(0.0)
        self.assertFalse(torch.allclose(orig_model[0].weight, model[0].weight))


if __name__ == "__main__":
    unittest.main()
