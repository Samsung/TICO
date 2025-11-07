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

from tico.quantization import convert, prepare
from tico.quantization.config.gptq import GPTQConfig
from tico.quantization.config.pt2e import PT2EConfig
from tico.quantization.evaluation.evaluate import BACKEND, evaluate

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


class BigLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2048, 2048)
        self.linear2 = torch.nn.Linear(2048, 2048)
        self.linear3 = torch.nn.Linear(2048, 2048)
        self.linear4 = torch.nn.Linear(2048, 2048)

    def forward(self, x):
        z = self.linear(x)
        z = self.linear2(z)
        z = self.linear3(z)
        z = self.linear4(z)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 2048),), {}


class NormConv2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 256, (3, 3), stride=1)
        self.conv2 = torch.nn.Conv2d(256, 512, (5, 5), stride=2)

    def forward(self, x):
        z = self.conv(x)
        z = self.conv2(z)
        return z

    def get_example_inputs(self):
        return (torch.randn(1, 128, 32, 32),), {}


class GPTQTest(unittest.TestCase):
    @unittest.skipIf(
        not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
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

        q_m = prepare(model, GPTQConfig(show_progress=False))
        q_m(sample_input)
        q_m = convert(q_m, inplace=True)

        # target
        target_q_proj_w = q_m.model.layers[0].self_attn.q_proj.weight

        # Check if weights are updated.
        self.assertFalse(torch.allclose(base_q_proj_w, target_q_proj_w))

        # TODO Check PEIR.
        # https://github.com/pytorch/pytorch/issues/148171

    @unittest.skipIf(
        not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
    )
    def test_net(self):
        q_m = BigLinear()
        q_m.eval()
        ori_m = q_m
        args, kwargs = ori_m.get_example_inputs()

        # Apply GPTQ
        q_m = prepare(q_m, GPTQConfig(show_progress=False))
        for _ in range(30):
            args, kwargs = ori_m.get_example_inputs()
            q_m(*args, **kwargs)
        convert(q_m, inplace=True)

        # Apply PT2E
        args, kwargs = ori_m.get_example_inputs()
        q_m = prepare(q_m, PT2EConfig(), args=args, kwargs=kwargs, inplace=False)

        # Calibration
        for i in range(100):
            args, kwargs = ori_m.get_example_inputs()
            q_m(*args, **kwargs)

        q_m = convert(q_m, inplace=False)

        # Export circle
        # pt2e exported model doesn't have `eval()` api.
        q_m.training = False
        cm = tico.convert(q_m, args, kwargs)

        # Evaluate
        results = evaluate(ori_m, cm, BACKEND.TRIV24, mode="return")
        # TODO Parametrize tolerance.
        tolerance = 0.02
        assert results is not None
        assert "peir" in results
        assert (
            results["peir"][0] < tolerance
        ), f"PEIR exceeds tolerance. PEIR:{results['peir'][0]}%, tolerance: {tolerance}%"

    @unittest.skipIf(
        not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
    )
    def test_normconv2d(self):
        q_m = NormConv2D()
        q_m.eval()
        ori_m = q_m
        args, kwargs = ori_m.get_example_inputs()

        # Apply GPTQ
        q_m = prepare(q_m, GPTQConfig(show_progress=False))
        for _ in range(30):
            args, kwargs = ori_m.get_example_inputs()
            q_m(*args, **kwargs)
        convert(q_m, inplace=True)
        # check that all convolution nodes are quantized
        assert hasattr(q_m, "quantizers"), "quantized model does not have quantizers"
        assert (
            "model.layers.0.conv" in q_m.quantizers
        ), "first conv node is not quantized"
        assert (
            "model.layers.0.conv2" in q_m.quantizers
        ), "second conv node is not quantized"

        # Apply PT2E
        args, kwargs = ori_m.get_example_inputs()
        q_m = prepare(q_m, PT2EConfig(), args=args, kwargs=kwargs, inplace=False)

        # Calibration
        for i in range(100):
            args, kwargs = ori_m.get_example_inputs()
            q_m(*args, **kwargs)

        q_m = convert(q_m, inplace=False)
        # Export circle
        # pt2e exported model doesn't have `eval()` api.
        q_m.training = False
        cm = tico.convert(q_m, args, kwargs)

        # Evaluate
        results = evaluate(ori_m, cm, BACKEND.TRIV24, mode="return")
        # TODO Parametrize tolerance.
        tolerance = 0.02
        assert results is not None
        assert "peir" in results
        assert (
            results["peir"][0] < tolerance
        ), f"PEIR exceeds tolerance. PEIR:{results['peir'][0]}%, tolerance: {tolerance}%"
