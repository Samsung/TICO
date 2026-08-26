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

import os
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from tico.quantization.algorithm.qwen3_vl_gptq.gptq import GPTQ
from tico.quantization.algorithm.qwen3_vl_gptq.quantizer import (
    FPInputsCache,
    Qwen3VLGPTQQuantizer,
)
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
        dXXT = gptq.dXXT
        assert dXXT is not None
        self.assertEqual(dXXT.shape, gptq.H.shape)  # type: ignore[union-attr]
        self.assertGreater(dXXT.abs().sum().item(), 0.0)


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

        native_inp = gptq_objs["linear"].native_inp
        assert native_inp is not None
        self.assertEqual(len(native_inp), 2)
        self.assertTrue(torch.allclose(native_inp[0], fp_inputs[0]))
        self.assertTrue(torch.allclose(native_inp[1], fp_inputs[1]))

    def test_resolve_weight_bits_default(self):
        """_resolve_weight_bits should return the config default when no override."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(weight_bits=4))
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
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
            quantizer.config,  # type: ignore[arg-type]
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
            quantizer.config,  # type: ignore[arg-type]
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
            quantizer.config,  # type: ignore[arg-type]
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
        self.assertTrue(torch.allclose(orig_model[0].weight, model[0].weight))  # type: ignore[index]
        # Modifying one should not affect the other
        model[0].weight.data.fill_(0.0)
        self.assertFalse(torch.allclose(orig_model[0].weight, model[0].weight))  # type: ignore[index]


# ---------------------------------------------------------------------------
# Helpers for FP inputs cache tests
# ---------------------------------------------------------------------------


def _make_quantizer(fp_inputs_cache_path=None, gptq_v2=True):
    """Create a Qwen3VLGPTQQuantizer with minimal config for testing."""
    config = Qwen3VLGPTQConfig(
        weight_bits=8,
        gptq_v2=gptq_v2,
        fp_inputs_cache_path=fp_inputs_cache_path,
        show_progress=False,
        verbose=False,
    )
    return Qwen3VLGPTQQuantizer(config)


# ---------------------------------------------------------------------------
# Tests: FPInputsCache
# ---------------------------------------------------------------------------


class TestFPInputsCache(unittest.TestCase):
    """Core tests for the FPInputsCache hook-based collector and disk cache."""

    def test_caches_fp_input(self):
        """A forward hook stores the first positional arg in fp_cache."""
        cache = FPInputsCache(["linear"])
        linear = nn.Linear(4, 4)
        cache.add_hook({"linear": linear})

        inp = torch.randn(2, 4)
        linear(inp)
        cache.clear_hook()

        self.assertIn("linear", cache.fp_cache)
        self.assertEqual(len(cache.fp_cache["linear"]), 1)
        self.assertTrue(torch.equal(cache.fp_cache["linear"][0], inp))

    def test_save_and_load_roundtrip(self):
        """Save cache to disk, load it back, and verify tensor equality."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "fp_cache.pt")
            dummy_cache = {
                "vision.patch_embed": {
                    "proj": [torch.randn(2, 3), torch.randn(2, 3)],
                },
                "text.layers.0": {
                    "self_attn.q_proj": [torch.randn(4, 5)],
                },
            }
            torch.save(dummy_cache, cache_path)
            self.assertTrue(os.path.exists(cache_path))

            loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.assertEqual(set(loaded.keys()), set(dummy_cache.keys()))
            for stage in dummy_cache:
                for name in dummy_cache[stage]:
                    for i, t in enumerate(loaded[stage][name]):
                        self.assertTrue(torch.equal(t, dummy_cache[stage][name][i]))

    def test_raw_replay_returns_cached_on_hit(self):
        """_collect_native_inputs_from_raw_replay returns cached data without
        running any forward hooks when stage_desc is in the disk cache."""
        quantizer = _make_quantizer(fp_inputs_cache_path="/tmp/dummy.pt")
        cached_tensors = [torch.randn(2, 3)]
        quantizer._fp_inputs_disk_cache = {
            "vision.merger": {"merger.linear": cached_tensors},
        }

        dummy_model = MagicMock()
        result = quantizer._collect_native_inputs_from_raw_replay(
            model=dummy_model,
            subset={"merger.linear": MagicMock()},
            module_name={},
            cache_args=[[]],
            cache_kwargs={},
            num_batches=1,
            stage_desc="vision.merger",
        )

        self.assertIn("merger.linear", result)
        self.assertTrue(torch.equal(result["merger.linear"][0], cached_tensors[0]))
        dummy_model.assert_not_called()

    @torch.no_grad()
    def test_collect_then_cache_hit(self):
        """First call collects via hooks; second call returns from cache
        without re-running forward."""
        quantizer = _make_quantizer(fp_inputs_cache_path="/tmp/dummy.pt")

        linear = nn.Linear(4, 4)
        stage_module = nn.Sequential(linear)
        subset = {"0": linear}

        inp = torch.randn(2, 4)
        cached_args = [[inp]]
        cached_kwargs: dict = {}

        result1 = quantizer._collect_native_inputs_from_stage_cache(
            stage_module=stage_module,
            subset=subset,
            cached_args=cached_args,
            cached_kwargs=cached_kwargs,
            stage_desc="test_stage",
            num_batches=1,
        )
        self.assertIn("0", result1)
        self.assertTrue(torch.equal(result1["0"][0], inp))

        quantizer._fp_inputs_disk_cache["test_stage"] = result1

        broken_module = MagicMock(side_effect=RuntimeError("should not be called"))
        result2 = quantizer._collect_native_inputs_from_stage_cache(
            stage_module=broken_module,
            subset=subset,
            cached_args=cached_args,
            cached_kwargs=cached_kwargs,
            stage_desc="test_stage",
            num_batches=1,
        )
        self.assertTrue(torch.equal(result2["0"][0], result1["0"][0]))
        broken_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
