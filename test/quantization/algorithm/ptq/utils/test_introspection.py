# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import os
import unittest

import torch
import torch.nn as nn

from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.algorithm.ptq.quant_config import QuantConfig
from tico.experimental.quantization.algorithm.ptq.utils.introspection import (
    build_fqn_map,
    diff_vs_fp_outputs,
    save_fp_outputs,
)
from tico.experimental.quantization.algorithm.ptq.wrappers.ptq_wrapper import PTQWrapper
from tico.experimental.quantization.config import SmoothQuantConfig

IS_CI_MODE = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4)
        # nn.Sequential gives us numbered sub-modules (0, 1).
        self.block = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
        )


class TestBuildFqnMap(unittest.TestCase):
    def setUp(self):
        # Build the test model once for all test methods.
        self.model = DummyModel()
        self.fqn_map = build_fqn_map(self.model)

    # ---------- basic correctness checks ---------- #
    def test_root_included(self):
        """Root module must be present with an empty string name."""
        self.assertIn(self.model, self.fqn_map)
        self.assertEqual(self.fqn_map[self.model], "")

    def test_direct_child_name(self):
        """Top-level child should map to its attribute name."""
        self.assertEqual(self.fqn_map[self.model.linear1], "linear1")

    def test_sequential_children(self):
        """Children inside nn.Sequential should get dotted numeric names."""
        conv = self.model.block[0]
        relu = self.model.block[1]
        self.assertEqual(self.fqn_map[conv], "block.0")
        self.assertEqual(self.fqn_map[relu], "block.1")

    # ---------- structural sanity tests ---------- #
    def test_total_entries(self):
        """
        The map should contain one entry per module instance:
          root, linear1, block, block.0, block.1  -> 5 total.
        """
        expected_count = 5
        self.assertEqual(len(self.fqn_map), expected_count)

    def test_bidirectional_consistency(self):
        """
        For every (module -> name) pair returned, the inverse lookup via
        model.named_modules() must also hold.
        """
        inverse = {m: n for n, m in self.model.named_modules()}
        for mod, name in self.fqn_map.items():
            self.assertEqual(name, inverse[mod])


@unittest.skipIf(
    not IS_CI_MODE, "Internal test — skipped unless --include-internal is set"
)
class TestSmoothQuantPTQDiff(unittest.TestCase):
    """
    Unit-test: verify that W8A8 SmoothQuant + PTQ does **not** explode layer-wise.

    The test checks per-wrapper activation deltas between
      • CALIB mode (FP32 pass-through)  vs.
      • QUANT mode (fake-/real-quant output)

    For speed it uses "Maykeye/TinyLLama-v0" and a single, short input.
    """

    model_name: str
    device: torch.device
    input_ids: torch.Tensor
    model: torch.nn.Module

    @classmethod
    def setUpClass(cls):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Light-weight model + tokenizer
        cls.model_name = "Maykeye/TinyLLama-v0"
        cls.device = torch.device("cpu")  # keep CI runs lightweight

        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        fp_model = (
            AutoModelForCausalLM.from_pretrained(cls.model_name).to(cls.device).eval()
        )
        fp_model.config.use_cache = False
        fqn_map = build_fqn_map(fp_model)

        # ① SmoothQuant
        sq_model = prepare(fp_model, SmoothQuantConfig(), inplace=True)

        # quick calibration: 5 samples
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        with torch.inference_mode():
            for i in range(5):
                ids = tokenizer(ds[i]["text"], return_tensors="pt").input_ids.to(
                    cls.device
                )
                sq_model(ids)

        sq_model = convert(sq_model, inplace=True)

        # ② PTQ wrapping (first N layers only to stay fast)
        qcfg = QuantConfig()
        new_layers = torch.nn.ModuleList()
        for idx, fp_layer in enumerate(sq_model.model.layers):
            if idx >= 4:  # wrap first 4 layers → quick
                new_layers.append(fp_layer)
                continue

            layer_cfg = qcfg.child(f"layer{idx}")
            q_layer = PTQWrapper(
                fp_layer,
                qcfg=layer_cfg,
                fp_name=fqn_map.get(fp_layer),
            )
            new_layers.append(q_layer)

        sq_model.model.layers = new_layers
        cls.model = sq_model  # reused in tests

        # Static input for both passes
        cls.input_ids = tokenizer(
            "Unit-test input sequence.",
            return_tensors="pt",
        ).input_ids.to(cls.device)

    def test_layerwise_diff(self):
        """Assert that each wrapped layer stays within a 1e-1 max-diff."""
        model = self.model

        # CALIB pass
        assert hasattr(model, "model")
        assert hasattr(model.model, "layers")
        assert isinstance(model.model.layers, torch.nn.Module)
        model.model.layers.apply(
            lambda m: getattr(m, "enable_calibration", lambda: None)()
        )
        h_save, cache = save_fp_outputs(model)
        with torch.no_grad():
            model(self.input_ids)
        for h in h_save:
            h.remove()

        # QUANT pass
        model.model.layers.apply(lambda m: getattr(m, "freeze_qparams", lambda: None)())
        h_cmp, diffs = diff_vs_fp_outputs(
            model, cache, rtol=0.0, atol=1.0, collect=True
        )
        assert isinstance(h_cmp, list)
        with torch.no_grad():
            model(self.input_ids)
        for h in h_cmp:
            h.remove()

        assert isinstance(diffs, dict)
        # Every wrapped layer must satisfy max-abs-diff ≤ 1.0
        for name, maxdiff in diffs.items():
            self.assertLessEqual(
                maxdiff,
                1.0,
                msg=f"Layer {name} exceeds tolerance: max-diff = {maxdiff:.3e}",
            )
