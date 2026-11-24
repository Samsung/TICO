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

"""A shared affine Quantize owns one domain for all matching DQ consumers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import tico
import torch
import torch.ao.quantization.fx._decomposed  # register Q/DQ
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.serialize.quant_param import QPARAM_KEY, QuantParam


class _SharedRequantizer(torch.nn.Module):
    def __init__(
        self,
        *,
        same_domain=False,
        return_codes=False,
        first_scale=None,
        second_scale=None,
    ):
        super().__init__()
        self.scale, self.zero_point = (0.125, 128) if same_domain else (0.5, 2)
        self.return_codes = return_codes
        self.first_scale = self.scale if first_scale is None else first_scale
        self.second_scale = self.scale if second_scale is None else second_scale

    def forward(self, value):
        source_codes = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            value, 0.125, 128, 0, 255, torch.uint8
        )
        source = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            source_codes, 0.125, 128, 0, 255, torch.uint8
        )
        codes = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            source, self.scale, self.zero_point, 0, 255, torch.uint8
        )
        first = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            codes, self.first_scale, self.zero_point, 0, 255, torch.uint8
        )
        second = torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            codes, self.second_scale, self.zero_point, 0, 255, torch.uint8
        )
        outputs = (first, second.reshape(2, 2), source)
        return (*outputs, codes) if self.return_codes else outputs


class SharedRequantizerExportTest(unittest.TestCase):
    # Both grids are binary-exact and destination rounding avoids halfway cases.
    sample = torch.tensor([[-16.0, -0.625, 0.625, 1.625]])
    input_codes = np.array([[0, 123, 133, 141]], dtype=np.uint8)
    output_codes = np.array([[0, 1, 3, 5]], dtype=np.uint8)
    output_real = np.array([[-1.0, -0.5, 0.5, 1.5]], dtype=np.float32)

    def test_fold_preserves_one_shared_requantizer_domain(self):
        for return_codes in (False, True):
            with self.subTest(return_codes=return_codes):
                exported = torch.export.export(
                    _SharedRequantizer(return_codes=return_codes).eval(), (self.sample,)
                )
                FoldQuantOps().call(exported)
                quantizes = [
                    node
                    for node in exported.graph.nodes
                    if node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ]
                self.assertEqual(len(quantizes), 1)
                quantize = quantizes[0]
                self.assertEqual(
                    quantize.meta[QPARAM_KEY],
                    QuantParam(dtype="uint8", scale=[0.5], zero_point=[2]),
                )
                self.assertEqual(
                    quantize.args[0].meta[QPARAM_KEY],
                    QuantParam(dtype="uint8", scale=[0.125], zero_point=[128]),
                )
                self.assertFalse(
                    any(
                        node.target
                        == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                        for node in exported.graph.nodes
                    )
                )
                output = next(
                    node for node in exported.graph.nodes if node.op == "output"
                )
                self.assertIs(output.args[0][0], quantize)
                self.assertIs(output.args[0][1].args[0], quantize)
                if return_codes:
                    self.assertIs(output.args[0][3], quantize)
                before = str(exported.graph)
                FoldQuantOps().call(exported)
                self.assertEqual(str(exported.graph), before)

    def test_mismatching_dq_consumer_is_not_folded(self):
        for first_scale, second_scale in ((0.5, 0.75), (0.75, 0.5)):
            with self.subTest(first_scale=first_scale, second_scale=second_scale):
                exported = torch.export.export(
                    _SharedRequantizer(
                        first_scale=first_scale, second_scale=second_scale
                    ).eval(),
                    (self.sample,),
                )
                FoldQuantOps().call(exported)
                dequantizes = [
                    node
                    for node in exported.graph.nodes
                    if node.target
                    == torch.ops.quantized_decomposed.dequantize_per_tensor.default
                ]
                self.assertEqual(len(dequantizes), 1)
                mismatched = dequantizes[0]
                self.assertEqual(mismatched.args[1:3], (0.75, 2))
                quantize = mismatched.args[0]
                self.assertEqual(quantize.args[1:3], (0.5, 2))
                self.assertEqual(
                    quantize.meta[QPARAM_KEY],
                    QuantParam(dtype="uint8", scale=[0.5], zero_point=[2]),
                )
                self.assertNotIn(QPARAM_KEY, mismatched.meta)
                output = next(
                    node for node in exported.graph.nodes if node.op == "output"
                )
                self.assertIs(
                    output.args[0][0], quantize if first_scale == 0.5 else mismatched
                )
                self.assertIs(
                    output.args[0][1].args[0],
                    quantize if second_scale == 0.5 else mismatched,
                )

    def test_mismatching_dq_consumer_is_rejected_by_export(self):
        for first_scale, second_scale in ((0.5, 0.75), (0.75, 0.5)):
            with self.subTest(first_scale=first_scale, second_scale=second_scale):
                with self.assertRaisesRegex(
                    RuntimeError, "DEQUANTIZE arguments disagree with input qparams"
                ):
                    tico.convert(
                        _SharedRequantizer(
                            first_scale=first_scale, second_scale=second_scale
                        ).eval(),
                        (self.sample,),
                    )

    def test_saved_circle_preserves_shared_domains_codes_and_values(self):
        for optimize in (False, True):
            for same_domain in (False, True):
                for return_codes in (False, True):
                    with self.subTest(
                        optimize=optimize,
                        same_domain=same_domain,
                        return_codes=return_codes,
                    ):
                        module = _SharedRequantizer(
                            same_domain=same_domain, return_codes=return_codes
                        ).eval()
                        converted = tico.convert(
                            module,
                            (self.sample,),
                            config=tico.CompileConfigV1(circle_optimize=optimize),
                        )
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "shared.circle"
                            converted.save(path)
                            document = CircleDocument.load(path)
                            document.verify(raise_on_error=True)
                            runtime = CircleReferenceRuntime(document)
                        program = runtime.program
                        quantizes = [
                            op for op in program.operators if op.name == "QUANTIZE"
                        ]
                        # A direct integer output keeps its explicit Q alive,
                        # even when the DQ consumers use the source grid.
                        self.assertEqual(
                            len(quantizes), int(not same_domain or return_codes)
                        )
                        self.assertFalse(
                            any(op.name == "DEQUANTIZE" for op in program.operators)
                        )
                        target = (module.scale, module.zero_point)
                        for index, grid in (
                            (program.inputs[0], (0.125, 128)),
                            (program.outputs[0], target),
                            (program.outputs[1], target),
                            (program.outputs[2], (0.125, 128)),
                        ):
                            tensor = program.tensor(index)
                            self.assertEqual(tensor.dtype, np.uint8)
                            qparam = tensor.quantization
                            assert qparam is not None
                            self.assertEqual(qparam.scale, (grid[0],))
                            self.assertEqual(qparam.zero_point, (grid[1],))
                        if return_codes:
                            self.assertEqual(
                                program.outputs[3],
                                quantizes[0].outputs[0],
                            )
                        native = runtime.run(
                            [self.input_codes], mode=ExecutionMode.NATIVE
                        )
                        fake = runtime.run(
                            [self.sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                        )
                        codes = self.input_codes if same_domain else self.output_codes
                        real = self.sample.numpy() if same_domain else self.output_real
                        expected_codes = [codes, codes.reshape(2, 2), self.input_codes]
                        expected_real = [real, real.reshape(2, 2), self.sample.numpy()]
                        if return_codes:
                            expected_codes.append(codes)
                            expected_real.append(real)
                        self.assertEqual(len(native.outputs), len(expected_codes))
                        self.assertEqual(len(fake.outputs), len(expected_real))
                        for native_value, native_expected in zip(
                            native.outputs, expected_codes
                        ):
                            self.assertEqual(native_value.dtype, np.uint8)
                            np.testing.assert_array_equal(native_value, native_expected)
                        for fake_value, fake_expected in zip(
                            fake.outputs, expected_real
                        ):
                            self.assertEqual(fake_value.dtype, np.float32)
                            np.testing.assert_array_equal(fake_value, fake_expected)
                        eager = module(self.sample)
                        for eager_value, eager_expected in zip(
                            eager[:3], expected_real[:3]
                        ):
                            np.testing.assert_array_equal(
                                eager_value.numpy(), eager_expected
                            )
                        if return_codes:
                            np.testing.assert_array_equal(eager[3].numpy(), codes)


if __name__ == "__main__":
    unittest.main()
