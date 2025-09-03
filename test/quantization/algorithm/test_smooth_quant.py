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

import numpy as np
import torch

from tico.experimental.quantization import convert, prepare
from tico.experimental.quantization.config import SmoothQuantConfig

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


class SmoothQuantTest(unittest.TestCase):
    @unittest.skipIf(
        not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
    )
    @torch.inference_mode()
    def test_value(self):
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Maykeye/TinyLLama-v0")
        model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")

        # Load data
        dataset = load_dataset("wikiText", "wikitext-2-raw-v1", split="train")
        sample_input = tokenizer(dataset[0]["text"], return_tensors="pt").input_ids

        # base
        base_output = model(sample_input).logits
        base_ln_weight = model.model.layers[0].input_layernorm.weight.clone()

        device = next(model.parameters()).device
        num_samples = 10

        # attach observers
        model = prepare(model, SmoothQuantConfig())

        # run calibration
        for i in range(num_samples):
            input_ids = tokenizer(dataset[i]["text"], return_tensors="pt").input_ids.to(
                device
            )
            model(input_ids)

        # apply smoothing
        q_m = convert(model)

        # target
        target_output = q_m(sample_input).logits
        target_ln_weight = q_m.model.layers[0].input_layernorm.weight

        # Check if weights are updated.
        self.assertFalse(torch.allclose(base_ln_weight, target_ln_weight))

        # Check if output values are same.
        np.testing.assert_allclose(
            actual=base_output,
            desired=target_output,
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"Value mismatches.\nbefore result: {base_output}\nafter result: {target_output}",
        )
