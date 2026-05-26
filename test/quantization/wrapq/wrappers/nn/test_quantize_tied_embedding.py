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

"""Smoke tests migrated from the tied-embedding quantization example."""

import os
import unittest

import torch
import torch.nn as nn

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_embedding import QuantEmbedding
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

IS_INTERNAL_TEST = os.environ.get("RUN_INTERNAL_TESTS", "0") == "1"


@unittest.skipIf(
    not IS_INTERNAL_TEST, "Internal test — run only if --include-internal is set"
)
class TiedEmbeddingLM(nn.Module):
    """Tiny language-model-like module with tied token embedding and LM-head weights."""

    def __init__(self, vocab_size: int = 16, hidden_size: int = 8) -> None:
        """Initialize a tied embedding and projection pair."""
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Run embedding lookup followed by the tied LM head."""
        return self.lm_head(self.embed(token_ids))


def _unwrap_weight(module: nn.Module) -> torch.Tensor:
    """Return the underlying floating-point weight through a WrapQ boundary."""
    if hasattr(module, "wrapped") and hasattr(module.wrapped, "module"):
        return module.wrapped.module.weight
    return module.weight  # type: ignore[attr-defined]


class TestNNTiedEmbeddingExample(unittest.TestCase):
    """Exercise tied-weight PTQ without running the heavy Circle sharing check."""

    def test_prepare_convert_tied_embedding_preserves_shared_parameter(self):
        """Quantize tied embedding modules and verify they still share one Parameter."""
        torch.manual_seed(123)
        model = TiedEmbeddingLM().eval()
        self.assertIs(model.embed.weight, model.lm_head.weight)

        qcfg = PTQConfig()
        model.embed = prepare(model.embed, qcfg)  # type: ignore[assignment]
        model.lm_head = prepare(model.lm_head, qcfg)  # type: ignore[assignment]
        self.assertIsInstance(model.embed.wrapped, QuantEmbedding)  # type: ignore[attr-defined]
        self.assertIsInstance(model.lm_head.wrapped, QuantLinear)  # type: ignore[attr-defined]
        self.assertIs(_unwrap_weight(model.embed), _unwrap_weight(model.lm_head))

        with torch.no_grad():
            for _ in range(3):
                token_ids = torch.randint(0, 16, (2, 5), dtype=torch.long)
                model(token_ids)

        model.embed = convert(model.embed)  # type: ignore[assignment]
        model.lm_head = convert(model.lm_head)  # type: ignore[assignment]
        self.assertIs(model.embed._mode, Mode.QUANT)  # type: ignore[attr-defined]
        self.assertIs(model.lm_head._mode, Mode.QUANT)  # type: ignore[attr-defined]
        self.assertIs(_unwrap_weight(model.embed), _unwrap_weight(model.lm_head))

        with torch.no_grad():
            output = model(torch.randint(0, 16, (1, 4), dtype=torch.long))
        self.assertEqual(tuple(output.shape), (1, 4, 16))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
