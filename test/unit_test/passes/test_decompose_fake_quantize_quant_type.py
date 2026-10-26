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

"""Characterize fake-quant range selection through both helpers and real passes."""

import unittest

import torch

from tico.passes.decompose_fake_quantize import (
    DecomposeFakeQuantize,
    get_quant_type as scalar_quant_type,
)
from tico.passes.decompose_fake_quantize_tensor_qparams import (
    DecomposeFakeQuantizeTensorQParams,
    get_quant_type as tensor_quant_type,
)

_SUPPORTED_RANGES = (
    (0, 15, torch.uint8),
    (0, 255, torch.uint8),
    (-32768, 32767, torch.int16),
    (-32767, 32767, torch.int16),
)
_SOURCE_TARGETS = {
    "scalar": torch.ops.aten.fake_quantize_per_tensor_affine.default,
    "channel": torch.ops.aten.fake_quantize_per_channel_affine.default,
    "tensor": torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams,
    "cachemask": torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default,
}


class _FakeQuantize(torch.nn.Module):
    def __init__(self, kind, quant_min, quant_max):
        super().__init__()
        self.kind = kind
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.register_buffer(
            "scale",
            torch.tensor([0.5, 0.25, 1.0] if kind == "channel" else 0.25),
        )
        self.register_buffer(
            "zero_point",
            torch.tensor([0, 1, 2] if kind == "channel" else 2, dtype=torch.int32),
        )
        self.register_buffer("enabled", torch.tensor(True))

    def forward(self, value):
        if self.kind == "scalar":
            return torch.ops.aten.fake_quantize_per_tensor_affine.default(
                value, 0.25, 2, self.quant_min, self.quant_max
            )
        if self.kind == "channel":
            return torch.ops.aten.fake_quantize_per_channel_affine.default(
                value, self.scale, self.zero_point, 1, self.quant_min, self.quant_max
            )
        if self.kind == "tensor":
            return torch.ops.aten.fake_quantize_per_tensor_affine.tensor_qparams(
                value, self.scale, self.zero_point, self.quant_min, self.quant_max
            )
        return torch.ops.aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default(
            value,
            self.scale,
            self.zero_point,
            self.enabled,
            self.quant_min,
            self.quant_max,
        )[
            0
        ]


class FakeQuantizeQuantTypeTest(unittest.TestCase):
    def test_supported_ranges_keep_both_import_paths_and_keyword_arguments(self):
        for helper in (scalar_quant_type, tensor_quant_type):
            for quant_min, quant_max, dtype in _SUPPORTED_RANGES:
                with self.subTest(
                    helper=helper.__module__, quant_min=quant_min, quant_max=quant_max
                ):
                    self.assertEqual(helper(quant_min, quant_max), dtype)
                    self.assertEqual(helper(min=quant_min, max=quant_max), dtype)

    def test_unsupported_ranges_keep_exact_error(self):
        ranges = ((0, 7), (-128, 127), (0, 256), (-32768, 32768), (15, 0))
        for helper in (scalar_quant_type, tensor_quant_type):
            for quant_min, quant_max in ranges:
                with self.subTest(
                    helper=helper.__module__, quant_min=quant_min, quant_max=quant_max
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        helper(min=quant_min, max=quant_max)
                    self.assertEqual(
                        str(caught.exception),
                        f"Not supported min/max values: {quant_min}/{quant_max}",
                    )

    def test_all_overloads_preserve_quantization_and_values(self):
        # Binary-exact scales let parity use exact equality, including rounding
        # ties and inputs outside the unsigned and signed quantization ranges.
        value = torch.tensor([[-10000.0, -1.125, -0.25], [0.0, 0.375, 10000.0]])
        original_value = value.clone()
        for kind, source_target in _SOURCE_TARGETS.items():
            for quant_min, quant_max, dtype in _SUPPORTED_RANGES:
                with self.subTest(kind=kind, quant_min=quant_min, quant_max=quant_max):
                    model = _FakeQuantize(kind, quant_min, quant_max).eval()
                    buffers = {
                        name: tensor.clone() for name, tensor in model.named_buffers()
                    }
                    ep = torch.export.export(model, (value,))
                    sources = [n for n in ep.graph.nodes if n.target == source_target]
                    self.assertEqual(len(sources), 1)
                    source_args = sources[0].args
                    before = ep.module()(value)

                    test_pass = (
                        DecomposeFakeQuantize()
                        if kind in ("scalar", "channel")
                        else DecomposeFakeQuantizeTensorQParams()
                    )
                    self.assertTrue(test_pass.call(ep).modified)
                    qd = torch.ops.quantized_decomposed
                    quant_target, dequant_target = (
                        (
                            qd.quantize_per_channel.default,
                            qd.dequantize_per_channel.default,
                        )
                        if kind == "channel"
                        else (
                            qd.quantize_per_tensor.default,
                            qd.dequantize_per_tensor.default,
                        )
                    )
                    quant_nodes = [
                        n for n in ep.graph.nodes if n.target == quant_target
                    ]
                    dequant_nodes = [
                        n for n in ep.graph.nodes if n.target == dequant_target
                    ]
                    self.assertEqual(len(quant_nodes), 1)
                    self.assertEqual(len(dequant_nodes), 1)
                    self.assertFalse(
                        any(n.target == source_target for n in ep.graph.nodes)
                    )
                    quant, dequant = quant_nodes[0], dequant_nodes[0]
                    self.assertEqual(quant.kwargs["dtype"], dtype)
                    self.assertEqual(dequant.kwargs["dtype"], dtype)
                    self.assertEqual(quant.args[-2:], (quant_min, quant_max))
                    self.assertIs(quant.args[0], source_args[0])
                    self.assertIs(dequant.args[0], quant)
                    self.assertEqual(dequant.args[1:], quant.args[1:])
                    if kind == "channel":
                        self.assertIs(quant.args[1], source_args[1])
                        self.assertIs(quant.args[2], source_args[2])
                        self.assertEqual(quant.args[3], 1)
                    else:
                        self.assertEqual(quant.args[1:3], (0.25, 2))

                    after = ep.module()(value)
                    self.assertEqual(after.shape, before.shape)
                    self.assertEqual(after.dtype, before.dtype)
                    self.assertTrue(torch.equal(after, before))
                    self.assertTrue(torch.equal(value, original_value))
                    for name, tensor in model.named_buffers():
                        self.assertTrue(torch.equal(tensor, buffers[name]))
                    graph_before_second_call = str(ep.graph)
                    self.assertFalse(test_pass.call(ep).modified)
                    self.assertEqual(str(ep.graph), graph_before_second_call)

    def test_all_overloads_reject_unsupported_range(self):
        value = torch.tensor([[0.0, 0.25, 0.5]])
        for kind, source_target in _SOURCE_TARGETS.items():
            with self.subTest(kind=kind):
                ep = torch.export.export(
                    _FakeQuantize(kind, -128, 127).eval(), (value,)
                )
                self.assertEqual(
                    sum(n.target == source_target for n in ep.graph.nodes), 1
                )
                test_pass = (
                    DecomposeFakeQuantize()
                    if kind in ("scalar", "channel")
                    else DecomposeFakeQuantizeTensorQParams()
                )
                graph_before = str(ep.graph)
                with self.assertRaises(RuntimeError) as caught:
                    test_pass.call(ep)
                self.assertEqual(
                    str(caught.exception), "Not supported min/max values: -128/127"
                )
                self.assertEqual(str(ep.graph), graph_before)
