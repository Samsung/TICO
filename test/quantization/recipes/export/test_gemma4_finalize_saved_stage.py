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

"""Saved finalize adapters preserve the direct Circle consumer contract."""

import tempfile
import unittest
from pathlib import Path

import torch
from tico.circle.runtime import ExecutionMode
from tico.quantization.recipes.export.gemma4_text_pipeline import (
    gemma4_text_stage_inputs,
    Gemma4CircleTextStage,
)
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    register_fake_quant_meta_kernels_for_dynamic_export,
)

from test.quantization.recipes.export.test_gemma4_finalize_circle import (
    _export_consumer,
    _HAS_GEMMA4,
    _make_consumer,
)


@unittest.skipUnless(_HAS_GEMMA4, "Transformers Gemma4 is required")
class TestGemma4FinalizeSavedStage(unittest.TestCase):
    """Keep saved-stage loading separate from direct optimization parity."""

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(823)
        register_fake_quant_meta_kernels_for_dynamic_export()

    def _exercise(self, quantized):
        for layer_type in ("full_attention", "sliding_attention"):
            with self.subTest(layer_type=layer_type, quantized=quantized):
                consumer, hidden, ple, rope, shared = _make_consumer(
                    layer_type, quantized
                )
                finalize = consumer.as_export_module("decode").eval()
                p = 2
                key = torch.arange(6).view(1, 6)
                query = torch.arange(6).view(6, 1)
                visible = (key <= query) & (key < p + 1)
                if layer_type == "sliding_attention":
                    visible &= key > query - 2
                mask = torch.where(visible, 0.0, -120.0).reshape(1, 1, 6, 6)
                one_hidden = hidden[:, p : p + 1, :]
                one_kwargs = dict(
                    attention_mask=mask[..., p : p + 1, :].squeeze(1),
                    position_embeddings=tuple(value[:, p : p + 1, :] for value in rope),
                    per_layer_input=ple[:, p : p + 1, :],
                    shared_key_value=shared,
                )
                _program, runtime, binary = _export_consumer(
                    finalize, one_hidden, one_kwargs
                )
                mode = (
                    ExecutionMode.FAKE_QUANTIZE if quantized else ExecutionMode.NATIVE
                )
                inputs, _spec = torch.utils._pytree.tree_flatten(
                    ((one_hidden,), one_kwargs)
                )
                expected = runtime.run(inputs, mode=mode).outputs
                self.assertEqual(len(expected), 1)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "consumer.circle"
                    path.write_bytes(binary)
                    loaded = Gemma4CircleTextStage(
                        path,
                        input_shapes=gemma4_text_stage_inputs(
                            {"is_shared": True, "head_dim": 4, "num_kv_heads": 1},
                            query_length=1,
                            capacity=6,
                            hidden_size=8,
                            ple_dim=2,
                            decode=True,
                        ),
                        output_shapes=[[1, 1, 8]],
                        execution_mode=mode,
                    )
                    torch.testing.assert_close(
                        loaded(one_hidden, **one_kwargs),
                        torch.from_numpy(expected[0]),
                        rtol=0,
                        atol=0,
                    )

    def test_float_saved_finalize_matches_direct_circle(self):
        self._exercise(False)

    def test_frozen_quantized_saved_finalize_matches_direct_circle(self):
        self._exercise(True)


if __name__ == "__main__":
    unittest.main()
