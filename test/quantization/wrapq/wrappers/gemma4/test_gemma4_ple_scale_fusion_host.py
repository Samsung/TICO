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

"""Host replay tests with explicit payloads, independent of calibration policy."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
    Gemma4PLEEmbeddingHostTable,
)


def observer(scale=0.125):
    """Return a simple per-tensor affine contract with a hand-known scale."""
    return {
        "scale": torch.tensor(scale),
        "zero_point": torch.tensor(0, dtype=torch.int),
        "quant_min": -32768,
        "quant_max": 32767,
        "channel_axis": None,
        "fake_quant_enabled": True,
    }


def payload(version=2, quantized=True):
    """Construct a tiny, exactly representable PLE table."""
    integers = torch.arange(48, dtype=torch.uint8).reshape(8, 6) % 16
    result = {
        "schema_version": version,
        "stage": "ple_embedding",
        "quantized": quantized,
        "num_hidden_layers": 2,
        "hidden_size_per_layer_input": 3,
        "vocab_size_per_layer_input": 8,
        "padding_idx": 0,
        "embed_scale": torch.tensor(1.0 if version == 2 else 16.0),
        "weight": integers.float() * 0.125,
        "weight_int": integers,
        "weight_scale": torch.full((8,), 0.125),
        "weight_zero_point": torch.zeros(8, dtype=torch.int),
        "weight_channel_axis": 0,
        "weight_float_dtype": "float32",
        "observers": {
            "embedding": observer(),
            "embed_scale": observer(),
            "act_out": observer(),
            "per_layer_token_inputs": observer(),
        },
    }
    return result


class TestGemma4PLEScaleFusionHost(unittest.TestCase):
    def test_fused_quantized_payload_bypasses_scale_observer(self):
        artifact = payload()
        # FQ(1; scale=.6) is 1.2, not 1. Merely multiplying by a quantized
        # identity would therefore fail this test; schema 2 must bypass it.
        artifact["observers"]["embed_scale"] = observer(0.6)
        host = Gemma4PLEEmbeddingHostTable(artifact)
        for count in (1, 4, 8):
            ids = torch.arange(count).reshape(1, count)
            expected = F.embedding(ids, artifact["weight"]).reshape(1, count, 2, 3)
            torch.testing.assert_close(host(ids), expected, atol=0, rtol=0)

    def test_legacy_quantized_payload_still_multiplies(self):
        artifact = payload(version=1)
        ids = torch.tensor([[0, 3, 7]])
        expected = (F.embedding(ids, artifact["weight"]) * 16).reshape(1, 3, 2, 3)
        torch.testing.assert_close(
            Gemma4PLEEmbeddingHostTable(artifact)(ids), expected, atol=0, rtol=0
        )

    def test_float_payloads_preserve_each_schema(self):
        ids = torch.tensor([[1, 4, 7]])
        for version in (1, 2):
            artifact = payload(version=version, quantized=False)
            expected = F.embedding(ids, artifact["weight"]) * artifact["embed_scale"]
            torch.testing.assert_close(
                Gemma4PLEEmbeddingHostTable(artifact)(ids),
                expected.reshape(1, 3, 2, 3),
                atol=0,
                rtol=0,
            )

    def test_fused_fp_host_export_contains_no_mul(self):
        host = Gemma4PLEEmbeddingHostTable(payload(quantized=False))
        ids = torch.tensor([[1, 2, 3]])
        ep = torch.export.export(host, (ids,))
        targets = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
        self.assertIn("aten.embedding.default", targets)
        self.assertFalse(any(t.startswith("aten.mul.") for t in targets))

    def test_disk_roundtrip(self):
        artifact = payload()
        ids = torch.tensor([[0, 1, 7]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ple.pt"
            torch.save(artifact, path)
            restored = Gemma4PLEEmbeddingHostTable.from_artifact(path)
            expected = Gemma4PLEEmbeddingHostTable(artifact)(ids)
            torch.testing.assert_close(restored(ids), expected, atol=0, rtol=0)

    def test_invalid_version_and_fused_scale_rejected(self):
        artifact = payload()
        artifact["schema_version"] = 999
        with self.assertRaisesRegex(ValueError, "schema_version"):
            Gemma4PLEEmbeddingHostTable(artifact)
        artifact = payload()
        artifact["embed_scale"] = torch.tensor(16.0)
        with self.assertRaisesRegex(ValueError, "embed_scale=1"):
            Gemma4PLEEmbeddingHostTable(artifact)

    def test_missing_observer_still_rejected(self):
        artifact = copy.deepcopy(payload())
        del artifact["observers"]["act_out"]
        with self.assertRaisesRegex(ValueError, "missing observers"):
            Gemma4PLEEmbeddingHostTable(artifact)


if __name__ == "__main__":
    unittest.main()
