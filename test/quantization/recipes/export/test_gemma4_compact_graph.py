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

"""Actual Gemma4 producer/consumer graphs keep compact internal key axes."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tico.circle.runtime import ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.recipes.export.gemma4_text_pipeline import (
    gemma4_text_stage_inputs,
    Gemma4CircleTextStage,
)
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    register_fake_quant_meta_kernels_for_dynamic_export,
)

from test.quantization.recipes.debug.test_static_gemma4_compact import (
    make_compact_runtime,
)
from test.quantization.recipes.export.test_gemma4_finalize_circle import (
    _export_consumer,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import _has_gemma4


@unittest.skipUnless(_has_gemma4(), "transformers Gemma4 modules are required")
class TestGemma4CompactGraph(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        register_fake_quant_meta_kernels_for_dynamic_export()

    def _exercise(self, quantized):
        runtime = make_compact_runtime(quantized=quantized, multiple_consumers=False)
        for index, module in enumerate(runtime.decode_layers):
            with self.subTest(index=index, quantized=quantized):
                shared = index in runtime.shared_kv_producers
                attention = runtime.text_model.layers[index].wrapped.self_attn.wrapped
                capacity = 3 if attention.is_sliding else 10
                head_dim = attention.head_dim
                heads = runtime.text_config.num_key_value_heads
                hidden = torch.randn(1, 1, 16) * 0.3
                key_length = capacity if shared else capacity - 1
                shape = (1, heads, key_length, head_dim)
                kwargs = dict(
                    attention_mask=torch.zeros(1, 1, capacity),
                    position_embeddings=(
                        torch.ones(1, 1, head_dim),
                        torch.zeros(1, 1, head_dim),
                    ),
                    per_layer_input=torch.randn(1, 1, 8) * 0.3,
                )
                kwargs["shared_key_value" if shared else "past_key_value"] = (
                    torch.randn(shape) * 0.3,
                    torch.randn(shape) * 0.3,
                )
                program, circle, binary = _export_consumer(module, hidden, kwargs)
                matmuls = [
                    node
                    for node in program.graph.nodes
                    if node.op == "call_function"
                    and node.target == torch.ops.aten.matmul.default
                ]
                softmaxes = [
                    node
                    for node in program.graph.nodes
                    if node.op == "call_function"
                    and node.target == torch.ops.aten.softmax.int
                ]
                self.assertGreater(len(softmaxes), 0)
                self.assertEqual(len(matmuls), len(softmaxes) * 2)
                for qk, pv, softmax in zip(matmuls[::2], matmuls[1::2], softmaxes):
                    self.assertEqual(qk.meta["val"].shape[-1], capacity)
                    self.assertEqual(softmax.meta["val"].shape[-1], capacity)
                    self.assertEqual(pv.args[0].meta["val"].shape[-1], capacity)
                    self.assertEqual(pv.args[1].meta["val"].shape[-2], capacity)
                circle_softmaxes = [
                    op for op in circle.program.operators if op.name == "SOFTMAX"
                ]
                circle_matmuls = [
                    op for op in circle.program.operators if op.name == "BATCH_MATMUL"
                ]
                self.assertEqual(len(circle_softmaxes), len(softmaxes))
                self.assertEqual(len(circle_matmuls), 2 * len(circle_softmaxes))
                for qk, pv, softmax in zip(
                    circle_matmuls[::2], circle_matmuls[1::2], circle_softmaxes
                ):
                    self.assertEqual(
                        circle.program.tensor(qk.outputs[0]).shape[-1], capacity
                    )
                    self.assertEqual(
                        circle.program.tensor(softmax.inputs[0]).shape[-1], capacity
                    )
                    self.assertEqual(
                        circle.program.tensor(softmax.outputs[0]).shape[-1], capacity
                    )
                    self.assertEqual(
                        circle.program.tensor(pv.inputs[0]).shape[-1], capacity
                    )
                    self.assertEqual(
                        circle.program.tensor(pv.inputs[1]).shape[-2], capacity
                    )
                cache_tensors = [
                    tensor
                    for tensor in circle.input_tensors
                    if "key_value" in tensor.name
                ]
                self.assertEqual(len(cache_tensors), 2)
                self.assertTrue(all(tensor.shape == shape for tensor in cache_tensors))
                if not shared:
                    self.assertEqual(
                        circle.output_tensors[1].shape, (1, heads, 1, head_dim)
                    )
                    self.assertEqual(
                        circle.output_tensors[2].shape, (1, heads, 1, head_dim)
                    )

                leaves, spec = torch.utils._pytree.tree_flatten(((hidden,), kwargs))
                encoded = [
                    (
                        fake_quantize_array(value.numpy(), tensor)
                        if tensor.is_quantized_integer
                        else value.numpy()
                    )
                    for value, tensor in zip(leaves, circle.input_tensors)
                ]
                args, eager_kwargs = torch.utils._pytree.tree_unflatten(
                    [torch.from_numpy(np.array(value, copy=True)) for value in encoded],
                    spec,
                )
                with torch.no_grad():
                    eager = module(*args, **eager_kwargs)
                eager_values = eager if isinstance(eager, tuple) else (eager,)
                mode = (
                    ExecutionMode.FAKE_QUANTIZE if quantized else ExecutionMode.NATIVE
                )
                result = circle.run(encoded, mode=mode).outputs
                for actual, wanted, tensor in zip(
                    result, eager_values, circle.output_tensors
                ):
                    # Same frozen affine grids. One output step covers cross-
                    # runtime rounding; 2e-6 covers float32 reduction roundoff.
                    step = (
                        max(tensor.quantization.scale)
                        if tensor.is_quantized_integer
                        else 0.0
                    )
                    np.testing.assert_allclose(
                        actual, wanted.numpy(), atol=step + 2e-6, rtol=1e-4
                    )
                layer = {
                    "is_shared": shared,
                    "head_dim": head_dim,
                    "num_kv_heads": heads,
                }
                outputs = [[1, 1, 16]] + (
                    [] if shared else [[1, heads, 1, head_dim]] * 2
                )
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "decode.circle"
                    path.write_bytes(binary)
                    loaded = Gemma4CircleTextStage(
                        path,
                        input_shapes=gemma4_text_stage_inputs(
                            layer,
                            query_length=1,
                            capacity=capacity,
                            hidden_size=16,
                            ple_dim=8,
                            decode=True,
                        ),
                        output_shapes=outputs,
                        execution_mode=mode,
                    )
                    reloaded = loaded(hidden, **kwargs)
                    direct = circle.run(leaves, mode=mode).outputs
                    loaded_values = (
                        reloaded if isinstance(reloaded, tuple) else (reloaded,)
                    )
                    for actual, wanted in zip(loaded_values, direct):
                        torch.testing.assert_close(
                            actual, torch.from_numpy(wanted), atol=0, rtol=0
                        )

    def test_float_actual_attention_key_axes_and_saved_circle(self):
        self._exercise(False)

    def test_frozen_affine_attention_key_axes_and_saved_circle(self):
        self._exercise(True)


if __name__ == "__main__":
    unittest.main()
