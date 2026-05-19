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

import importlib.util
import unittest

import torch

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(
    HAS_TRANSFORMERS, "transformers is required for static runtime helpers"
)
class TestStaticLlamaRuntimeHelpers(unittest.TestCase):
    def test_static_llama_padding_and_position_helpers(self):
        """Static LLaMA runtime helpers should handle right-padded prompt layouts."""
        from tico.quantization.recipes.debug.static_llama_runtime import (
            _build_position_ids_from_valid_token_mask,
            _normalize_valid_token_mask,
            _validate_padding_layout,
        )

        input_ids = torch.tensor([[4, 5, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 0, 0]])

        valid = _normalize_valid_token_mask(
            input_ids,
            attention_mask,
            pad_token_id=0,
            device=torch.device("cpu"),
        )
        _validate_padding_layout(valid, "right")
        position_ids = _build_position_ids_from_valid_token_mask(valid)

        self.assertEqual(valid.tolist(), [[True, True, False, False]])
        self.assertEqual(position_ids.tolist(), [[0, 1, 0, 0]])

    def test_static_llama_gather_last_token_logits(self):
        """Logit gathering should select the last real prompt token for each batch row."""
        from tico.quantization.recipes.debug.static_llama_runtime import (
            _gather_last_token_logits,
        )

        logits = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        valid = torch.tensor(
            [
                [True, True, False, False],
                [False, False, True, True],
            ]
        )

        gathered = _gather_last_token_logits(logits, valid)

        self.assertTrue(torch.equal(gathered[0], logits[0, 1]))
        self.assertTrue(torch.equal(gathered[1], logits[1, 3]))
