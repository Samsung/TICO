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

"""Serialize explicit Quantize boundaries without losing sequential domains."""

import unittest

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime
from tico.serialize.circle_serializer import build_circle_bytes
from tico.serialize.quant_param import QPARAM_KEY, QuantParam


class _ConcatRequantize(torch.nn.Module):
    def forward(self, left, right):
        left = torch.fake_quantize_per_tensor_affine(left, 0.5, 127, 0, 255)
        left = torch.fake_quantize_per_tensor_affine(left, 0.125, 17, 0, 255)
        right = torch.fake_quantize_per_tensor_affine(right, 0.5, 127, 0, 255)
        right = torch.fake_quantize_per_tensor_affine(right, 0.25, 23, 0, 255)
        combined = torch.cat((left, right), dim=1)
        return torch.fake_quantize_per_tensor_affine(combined, 0.375, 29, 0, 255)


class _ExplicitQuantize(torch.nn.Module):
    def forward(self, value):
        return torch.ops.quantized_decomposed.quantize_per_tensor.default(
            value, 0.125, 17, 0, 255, torch.uint8
        )


class QuantizePerTensorRetargetSerializationTest(unittest.TestCase):
    def test_public_conversion_preserves_branch_and_concat_domains(self):
        inputs = (torch.tensor([[-3.0, 1.0]]), torch.tensor([[1.5, 2.0]]))
        for optimize in (False, True):
            with self.subTest(circle_optimize=optimize):
                converted = tico.convert(
                    _ConcatRequantize().eval(),
                    inputs,
                    config=tico.CompileConfigV1(circle_optimize=optimize),
                )
                document = CircleDocument.from_bytes(converted.circle_binary)
                document.verify()
                runtime = CircleReferenceRuntime(document)
                quantizes = [
                    op for op in runtime.program.operators if op.name == "QUANTIZE"
                ]
                # Source -> branch and branch -> CAT are distinct transitions.
                # In particular the left branch saturates -3 to -2.125 before
                # the CAT grid rounds it to -2.25; direct retargeting gives -3.
                self.assertEqual(len(quantizes), 4)
                concat = next(
                    op for op in runtime.program.operators if op.name == "CONCATENATION"
                )
                producers = {index: op for op in quantizes for index in op.outputs}
                for index, branch_scale, branch_zp in zip(
                    concat.inputs, (0.125, 0.25), (17, 23)
                ):
                    final = producers[index]
                    branch = producers[final.inputs[0]]
                    source = runtime.program.tensor(branch.inputs[0])
                    intermediate = runtime.program.tensor(branch.outputs[0])
                    destination = runtime.program.tensor(final.outputs[0])
                    for tensor, scale, zp in (
                        (source, 0.5, 127),
                        (intermediate, branch_scale, branch_zp),
                        (destination, 0.375, 29),
                    ):
                        self.assertEqual(tensor.dtype, np.uint8)
                        assert tensor.quantization is not None
                        self.assertEqual(tensor.quantization.scale, (scale,))
                        self.assertEqual(tensor.quantization.zero_point, (zp,))
                result = runtime.run(
                    [
                        np.array([[121, 129]], dtype=np.uint8),
                        np.array([[130, 131]], dtype=np.uint8),
                    ]
                )
                np.testing.assert_array_equal(
                    result.outputs[0], np.array([[23, 32, 33, 34]], dtype=np.uint8)
                )
                output = runtime.program.tensor(concat.outputs[0])
                assert output.quantization is not None
                self.assertEqual(output.quantization.scale, (0.375,))
                self.assertEqual(output.quantization.zero_point, (29,))

    def test_serializer_rejects_metadata_only_domain_change(self):
        program = torch.export.export(
            _ExplicitQuantize().eval(), (torch.tensor([[0.5, 1.0]]),)
        )
        quantize = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        quantize.meta[QPARAM_KEY] = QuantParam(
            scale=[0.375], zero_point=[29], dtype="uint8"
        )

        with self.assertRaisesRegex(
            RuntimeError, "Quantize qparam mismatch after propagation"
        ):
            build_circle_bytes(program)


if __name__ == "__main__":
    unittest.main()
