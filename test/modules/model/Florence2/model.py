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

import requests  # type: ignore[import-untyped]
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from test.modules.base import TestModuleBase


class Florence2(TestModuleBase):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-base",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        ).to("cpu")
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", trust_remote_code=True
        )
        # TODO: Revisit the threshold values
        self.rtol = 4e-3
        self.atol = 4e-3

    def forward(self, input_ids, pixel_values, attention_mask, decoder_input_ids):
        return self.model(input_ids, pixel_values, attention_mask, decoder_input_ids)

    def get_example_inputs(self):
        torch.manual_seed(0)
        seq_len = 590

        prompt = "<OD>"
        url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            "cpu", torch.float32
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = torch.ones(1, seq_len, dtype=torch.int64)
        max_id = int(torch.max(input_ids))
        decoder_input_ids = torch.randint(
            low=0,
            high=max_id - 1,
            size=(1, seq_len),
            dtype=torch.float32,
            device="cpu",
        ).type("torch.LongTensor")

        return (
            input_ids,
            pixel_values,
            attention_mask,
            decoder_input_ids,
        ), {}
