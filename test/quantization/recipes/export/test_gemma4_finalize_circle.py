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

"""Real frozen Gemma4 consumer graphs: full prefill versus one-query finalize."""

import unittest
from typing import cast

import numpy as np
import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.wrappers.gemma4.quant_text_attention import (
    QuantGemma4TextAttention,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_decoder_layer import (
    QuantGemma4TextDecoderLayer,
)
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    register_fake_quant_meta_kernels_for_dynamic_export,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.support.quantization.gemma_kv_cache import tensor_qparams, tiny_config

try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer

    _HAS_GEMMA4 = True
except ImportError:
    _HAS_GEMMA4 = False


def _make_consumer(layer_type, quantized):
    """Calibrate full prompt graphs once; both adapters borrow those frozen grids."""
    config = tiny_config(layer_type)
    config.hidden_size_per_layer_input = 2
    config.max_position_embeddings = 6
    config.sliding_window = 2
    consumer_idx = 2 if layer_type == "sliding_attention" else 1
    qcfg = make_affine_ptq_config(
        dtype=DType.int(16), model_args={"text": {"max_seq": 6}}
    )
    producer = QuantGemma4TextDecoderLayer(
        Gemma4TextDecoderLayer(config, layer_idx=0).eval(), qcfg=qcfg
    ).eval()
    consumer = QuantGemma4TextDecoderLayer(
        Gemma4TextDecoderLayer(config, layer_idx=consumer_idx).eval(), qcfg=qcfg
    ).eval()
    cast(QuantGemma4TextAttention, consumer.self_attn.wrapped).bind_cache_source(
        cast(QuantGemma4TextAttention, producer.self_attn.wrapped)
    )
    hidden = torch.randn(1, 6, 8) * 0.3
    ple = torch.randn(1, 6, 2) * 0.3
    # Include nonzero token positions; selected RoPE must retain original position.
    angles = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) * 0.07
    rope = (angles.cos(), -angles.sin())
    mask = torch.where(
        torch.arange(6).view(1, 6) <= torch.arange(6).view(6, 1), 0.0, -120.0
    ).reshape(1, 1, 6, 6)
    kwargs = dict(per_layer_input=ple, position_embeddings=rope, attention_mask=mask)
    if quantized:
        producer.enable_calibration()
        with torch.no_grad():
            producer(hidden, **kwargs, use_cache=True)
        producer.freeze_qparams()
    with torch.no_grad():
        prefix_hidden, shared = producer(hidden, **kwargs, use_cache=True)
    if quantized:
        consumer.enable_calibration()
        with torch.no_grad():
            consumer(prefix_hidden, **kwargs, shared_key_value=shared)
        consumer.freeze_qparams()
    return consumer, prefix_hidden, ple, rope, shared


def _export_consumer(module, hidden, kwargs):
    """Export a real consumer for Circle numerical and shape checks."""
    with torch.no_grad():
        program = torch.export.export(module, (hidden,), kwargs=kwargs, strict=False)
        converted = tico.convert_from_exported_program(program)
    document = CircleDocument.from_bytes(converted.circle_binary)
    document.verify()
    return program, CircleReferenceRuntime(document), converted.circle_binary


@unittest.skipUnless(_HAS_GEMMA4, "Transformers Gemma4 is required")
class TestGemma4FinalizeCircle(unittest.TestCase):
    """Verify serialized qlen=1 compute, boundary domains and CPU reference parity."""

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(823)
        register_fake_quant_meta_kernels_for_dynamic_export()

    def _run(self, module, runtime, hidden, kwargs, quantized):
        leaves, spec = torch.utils._pytree.tree_flatten(((hidden,), kwargs))
        self.assertEqual(len(leaves), len(runtime.program.inputs))
        encoded = []
        for value, index in zip(leaves, runtime.program.inputs):
            tensor = runtime.program.tensor(index)
            array = value.detach().numpy()
            if tensor.is_quantized_integer:
                array = fake_quantize_array(array, tensor)
            encoded.append(np.array(array, copy=True))
        args, eager_kwargs = torch.utils._pytree.tree_unflatten(
            [torch.from_numpy(array.copy()) for array in encoded], spec
        )
        with torch.no_grad():
            expected = module(*args, **eager_kwargs)
        mode = ExecutionMode.FAKE_QUANTIZE if quantized else ExecutionMode.NATIVE
        result = runtime.run(encoded, mode=mode)
        self.assertEqual(len(result.outputs), 1)
        # Float32 RMSNorm/softmax/reduction roundoff. Frozen INT16 outputs have
        # a much smaller step than UINT8; use one output step plus FP roundoff
        # for cross-runtime arithmetic (selection parity itself is tighter).
        output_tensor = runtime.program.tensor(runtime.program.outputs[0])
        step = max(output_tensor.quantization.scale) if quantized else 0.0
        np.testing.assert_allclose(
            result.outputs[0],
            expected.detach().numpy(),
            rtol=1e-4,
            atol=step + 2e-6,
        )
        return result.outputs[0]

    def _exercise(self, quantized):
        for layer_type in ("full_attention", "sliding_attention"):
            with self.subTest(layer_type=layer_type, quantized=quantized):
                consumer, hidden, ple, rope, shared = _make_consumer(
                    layer_type, quantized
                )
                full = consumer.as_export_module("prefill").eval()
                finalize = consumer.as_export_module("decode").eval()
                # Right-padded prompt: the final physical tensor slot is invalid.
                p = 2
                key = torch.arange(6).view(1, 6)
                query = torch.arange(6).view(6, 1)
                visible = (key <= query) & (key < p + 1)
                if layer_type == "sliding_attention":
                    visible &= key > query - 2
                mask = torch.where(visible, 0.0, -120.0).reshape(1, 1, 6, 6)
                full_kwargs = dict(
                    attention_mask=mask,
                    position_embeddings=rope,
                    per_layer_input=ple,
                    shared_key_value=shared,
                )
                one_kwargs = dict(
                    attention_mask=mask[..., p : p + 1, :].squeeze(1),
                    position_embeddings=tuple(value[:, p : p + 1, :] for value in rope),
                    per_layer_input=ple[:, p : p + 1, :],
                    shared_key_value=shared,
                )
                with torch.no_grad():
                    wanted = full(hidden, **full_kwargs)[:, p : p + 1, :]
                    actual = finalize(hidden[:, p : p + 1, :], **one_kwargs)
                # Same frozen grids: selecting a query changes only float32 GEMM
                # accumulation order, not quantization policy or calibration.
                torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=2e-6)
                full_ep, full_rt, full_binary = _export_consumer(
                    full, hidden, full_kwargs
                )
                one_ep, one_rt, one_binary = _export_consumer(
                    finalize, hidden[:, p : p + 1, :], one_kwargs
                )
                self.assertEqual(one_rt.input_tensors[0].shape, (1, 1, 8))
                self.assertEqual(one_rt.output_tensors[0].shape, (1, 1, 8))
                # All projection/MLP linear inputs carry only one token. The
                # external KV tensor keeps all S slots, including token p.
                linears = [
                    node
                    for node in one_ep.graph.nodes
                    if node.op == "call_function"
                    and node.target == torch.ops.aten.linear.default
                ]
                self.assertGreaterEqual(len(linears), 6)
                for node in linears:
                    self.assertEqual(tuple(node.args[0].meta["val"].shape[:2]), (1, 1))
                self.assertEqual(one_rt.input_tensors[-2].shape, (1, 1, 6, 4))
                attention_shapes = []
                for program, expected_rows in ((full_ep, 12), (one_ep, 2)):
                    matmuls = [
                        node
                        for node in program.graph.nodes
                        if node.op == "call_function"
                        and node.target == torch.ops.aten.matmul.default
                    ]
                    shapes = [tuple(node.meta["val"].shape) for node in matmuls]
                    self.assertEqual(
                        shapes, [(1, expected_rows, 6), (1, expected_rows, 4)]
                    )
                    attention_shapes.append(shapes)
                print(
                    f"Gemma4 {layer_type} quantized={quantized}: "
                    f"Circle bytes {len(full_binary)} -> {len(one_binary)}, "
                    f"operators {len(full_rt.program.operators)} -> {len(one_rt.program.operators)}, "
                    f"attention {attention_shapes[0]} -> {attention_shapes[1]}"
                )
                if quantized:
                    # Host selection introduces no graph operator and preserves
                    # every input/output domain. Reusing decode adds no Q/DQ.
                    self.assertEqual(
                        [tensor_qparams(t) for t in full_rt.input_tensors],
                        [tensor_qparams(t) for t in one_rt.input_tensors],
                    )
                    full_codes = [op.name for op in full_rt.program.operators]
                    one_codes = [op.name for op in one_rt.program.operators]
                    for code in ("QUANTIZE", "DEQUANTIZE"):
                        self.assertEqual(full_codes.count(code), one_codes.count(code))
                full_value = self._run(full, full_rt, hidden, full_kwargs, quantized)
                one_value = self._run(
                    finalize, one_rt, hidden[:, p : p + 1, :], one_kwargs, quantized
                )
                np.testing.assert_allclose(
                    one_value, full_value[:, p : p + 1, :], rtol=1e-5, atol=2e-6
                )

    def test_float_serialized_finalize_matches_full_prefill(self):
        self._exercise(False)

    def test_frozen_quantized_serialized_finalize_matches_full_prefill(self):
        self._exercise(True)


if __name__ == "__main__":
    unittest.main()
