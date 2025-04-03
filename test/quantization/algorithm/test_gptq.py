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

import tico
import torch

from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config import GPTQConfig, PT2EConfig
from tico.experimental.quantization.evaluation.evaluate import BACKEND, evaluate

IS_CI_MODE = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


class BigLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2048, 2048)
        self.linear2 = torch.nn.Linear(2048, 2048)
        self.linear3 = torch.nn.Linear(2048, 2048)
        self.linear4 = torch.nn.Linear(2048, 2048)

    def forward(self, x):
        z = self.linear(x)
        z = self.linear2(x)
        z = self.linear3(x)
        z = self.linear4(x)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 2048),)


class GPTQTest(unittest.TestCase):
    @unittest.skipIf(
        not IS_CI_MODE, "Internal test — skipped unless --include-internal is set"
    )
    @torch.inference_mode()
    def test_model(self):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
        model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

        # Load data
        dataset = load_dataset("wikiText", "wikitext-2-raw-v1", split="train")
        sample_input = tokenizer(dataset[0]["text"], return_tensors="pt").input_ids

        # base
        base_q_proj_w = model.model.layers[0].self_attn.q_proj.weight.clone()

        q_m = prepare(model, GPTQConfig(), args=(sample_input,))
        q_m = convert(q_m)

        # target
        target_q_proj_w = q_m.model.layers[0].self_attn.q_proj.weight

        # Check if weights are updated.
        self.assertFalse(torch.allclose(base_q_proj_w, target_q_proj_w))

        # TODO Check PEIR.
        # https://github.com/pytorch/pytorch/issues/148171

    @unittest.skipIf(
        not IS_CI_MODE, "Internal test — skipped unless --include-internal is set"
    )
    def test_net(self):
        q_m = BigLinear()
        ori_m = q_m
        example_inputs = ori_m.get_example_inputs()

        # Apply GPTQ
        q_m = prepare(q_m, GPTQConfig(), args=example_inputs)
        convert(q_m)

        # Apply PT2E
        q_m = prepare(q_m, PT2EConfig(), args=example_inputs)

        # Calibration
        for i in range(100):
            q_m(*ori_m.get_example_inputs())

        q_m = convert(q_m)

        # Export circle
        cm = tico.convert(q_m, example_inputs)
        cm.save("op.q.circle")

        # Evaluate
        results = evaluate(ori_m, cm, BACKEND.TRIV24, mode="return")
        # TODO Parametrize tolerance.
        tolerance = 1e-2
        assert results is not None
        assert "peir" in results
        assert (
            results["peir"][0] < tolerance
        ), f"PEIR exceeds tolerance. PEIR:{results['peir'][0]}%, tolerance: {tolerance}%"
