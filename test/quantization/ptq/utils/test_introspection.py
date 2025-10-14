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
from tico.experimental.quantization.config.ptq import PTQConfig
from tico.experimental.quantization.config.smoothquant import SmoothQuantConfig
from tico.experimental.quantization.ptq.utils.introspection import (
    build_fqn_map,
    compare_layer_outputs,
    save_fp_outputs,
)
from tico.experimental.quantization.ptq.wrappers.ptq_wrapper import PTQWrapper

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


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
        self.assertIn(self.model, self.fqn_map)
        self.assertEqual(self.fqn_map[self.model], "")

    def test_direct_child_name(self):
        self.assertEqual(self.fqn_map[self.model.linear1], "linear1")

    def test_sequential_children(self):
        conv = self.model.block[0]
        relu = self.model.block[1]
        self.assertEqual(self.fqn_map[conv], "block.0")
        self.assertEqual(self.fqn_map[relu], "block.1")

    # ---------- structural sanity tests ---------- #
    def test_total_entries(self):
        expected_count = 5
        self.assertEqual(len(self.fqn_map), expected_count)

    def test_bidirectional_consistency(self):
        inverse = {m: n for n, m in self.model.named_modules()}
        for mod, name in self.fqn_map.items():
            self.assertEqual(name, inverse[mod])


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
class TestSmoothQuantPTQDiff(unittest.TestCase):
    """
    Unit-test: verify that W8A8 SmoothQuant + PTQ does NOT explode layer-wise.

    The test checks per-wrapper activation deltas between
      • CALIB mode (FP32 pass-through)  vs.
      • QUANT mode (fake-/real-quant output)

    For speed it uses "Maykeye/TinyLLama-v0" and a single, short input.
    """

    model_name: str
    device: torch.device
    input_ids: torch.Tensor
    model: torch.nn.Module
    fp_cache: dict[str, torch.Tensor]

    @classmethod
    def setUpClass(cls):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cls.model_name = "Maykeye/TinyLLama-v0"
        cls.device = torch.device("cpu")

        # tiny model + tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        fp_model = (
            AutoModelForCausalLM.from_pretrained(cls.model_name).to(cls.device).eval()
        )
        fp_model.config.use_cache = False
        fqn_map = build_fqn_map(fp_model)

        # SmoothQuant calibration
        sq_model = prepare(fp_model, SmoothQuantConfig(), inplace=True)
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        with torch.inference_mode():
            for i in range(5):
                ids = tokenizer(ds[i]["text"], return_tensors="pt").input_ids.to(
                    cls.device
                )
                sq_model(ids)
        sq_model = convert(sq_model, inplace=True)

        # PTQ-wrap first 4 layers
        qcfg = PTQConfig()
        new_layers = torch.nn.ModuleList()
        for idx, fp_layer in enumerate(sq_model.model.layers):
            if idx >= 4:
                new_layers.append(fp_layer)
                continue
            new_layers.append(
                PTQWrapper(
                    fp_layer,
                    qcfg=qcfg.child(f"layer{idx}"),
                    fp_name=fqn_map.get(fp_layer),
                )
            )
        sq_model.model.layers = new_layers
        cls.model = sq_model

        # prepare static input & capture FP refs
        cls.input_ids = tokenizer(
            "Unit-test input sequence.", return_tensors="pt"
        ).input_ids.to(cls.device)

        sq_model.model.layers.apply(
            lambda m: getattr(m, "enable_calibration", lambda: None)()
        )
        h_save, cls.fp_cache = save_fp_outputs(cls.model)
        with torch.no_grad():
            cls.model(cls.input_ids)
        for h in h_save:
            h.remove()

        # switch to QUANT mode
        sq_model.model.layers.apply(
            lambda m: getattr(m, "freeze_qparams", lambda: None)()
        )

    # ------------------------------------------------------------------ #
    # 1. Original diff-only assertion (updated for nested dict)          #
    # ------------------------------------------------------------------ #
    def test_layerwise_diff(self):
        h_cmp, stats = compare_layer_outputs(
            self.model,
            self.fp_cache,
            metrics=["diff"],
            rtol=0.0,
            atol=1.0,
            collect=True,
        )
        with torch.no_grad():
            self.model(self.input_ids)
        for h in h_cmp:
            h.remove()

        for name, metric_dict in stats.items():
            self.assertLessEqual(
                metric_dict["diff"],
                1.0,
                msg=f"{name}: diff={metric_dict['diff']:.3e} > 1.0",
            )

    # ------------------------------------------------------------------ #
    # 2. PEIR metric exists & is finite                                  #
    # ------------------------------------------------------------------ #
    def test_layerwise_peir(self):
        _, stats = compare_layer_outputs(
            self.model, self.fp_cache, metrics=["peir"], collect=True
        )
        for name, metric_dict in stats.items():
            val = metric_dict["peir"]
            self.assertTrue(
                torch.isfinite(torch.tensor(val)),
                msg=f"{name}: non-finite PEIR",
            )

    # ------------------------------------------------------------------ #
    # 3. Subset selection ('diff', 'peir')                               #
    # ------------------------------------------------------------------ #
    def test_metric_subset_selection(self):
        _, stats = compare_layer_outputs(
            self.model,
            self.fp_cache,
            metrics=["diff", "peir"],
            collect=True,
        )
        for metric_dict in stats.values():
            self.assertEqual(set(metric_dict), {"diff", "peir"})

    # ------------------------------------------------------------------ #
    # 4. Custom metric (mean-abs-error)                                  #
    # ------------------------------------------------------------------ #
    def test_custom_metric(self):
        def mae(a: torch.Tensor, b: torch.Tensor) -> float:
            return (a - b).abs().mean().item()

        _, stats = compare_layer_outputs(
            self.model,
            self.fp_cache,
            metrics=["mae"],
            custom_metrics={"mae": mae},
            collect=True,
        )
        for metric_dict in stats.values():
            self.assertIn("mae", metric_dict)
