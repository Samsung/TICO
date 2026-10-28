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

"""Mixed affine concat legalization through the public Circle export pipeline."""

import unittest

import numpy as np

import tico

import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode


class _MixedConcat(torch.nn.Module):
    def __init__(self, input_dtypes, output_dtype):
        super().__init__()
        self.input_dtypes = input_dtypes
        self.output_dtype = output_dtype
        self.destination = (0.25, 101) if output_dtype == "uint8" else (0.5, 0)
        # Keep the matching input on the destination grid. This avoids testing
        # the independent backward-propagation policy for same-dtype domains.
        self.sources = tuple(
            (
                self.destination
                if dtype == output_dtype
                else (0.125, 73 if dtype == "uint8" else 0)
            )
            for dtype in input_dtypes
        )

    @staticmethod
    def _quantize(
        value: torch.Tensor, dtype: str, qparam: tuple[float, int]
    ) -> torch.Tensor:
        minimum, maximum = (0, 255) if dtype == "uint8" else (-32768, 32767)
        scale, zero_point = qparam
        return torch.fake_quantize_per_tensor_affine(
            value, scale, zero_point, minimum, maximum
        )

    def forward(self, first, second):
        first = self._quantize(first, self.input_dtypes[0], self.sources[0])
        second = self._quantize(second, self.input_dtypes[1], self.sources[1])
        joined = self._quantize(
            torch.cat((first, second), dim=1),
            self.output_dtype,
            self.destination,
        )
        return joined, first, second, second.reshape(2, 1)


class MixedDtypeConcatExportTest(unittest.TestCase):
    def _assert_contract(self, tensor, dtype, qparam, shape):
        self.assertEqual(tensor.dtype, np.dtype(dtype))
        self.assertEqual(tensor.shape, shape)
        self.assertIsNotNone(tensor.quantization)
        self.assertEqual(tuple(tensor.quantization.scale), (qparam[0],))
        self.assertEqual(tuple(tensor.quantization.zero_point), (qparam[1],))

    def test_concat_requantizes_connections_and_preserves_external_contracts(self):
        for input_dtypes in (("uint8", "int16"), ("int16", "uint8")):
            for output_dtype in ("uint8", "int16"):
                for optimize in (False, True):
                    with self.subTest(
                        inputs=input_dtypes, output=output_dtype, optimize=optimize
                    ):
                        module = _MixedConcat(input_dtypes, output_dtype).eval()
                        sample = torch.tensor([[-0.5, 1.5]])
                        converted = tico.convert(
                            module,
                            (sample, sample.clone()),
                            config=tico.CompileConfigV1(circle_optimize=optimize),
                        )
                        document = CircleDocument.from_bytes(converted.circle_binary)
                        document.verify()
                        runtime = CircleReferenceRuntime(document)
                        program = runtime.program
                        concat = next(
                            operator
                            for operator in program.operators
                            if operator.name == "CONCATENATION"
                        )
                        self.assertEqual(concat.options.axis, 1)
                        self.assertEqual(concat.outputs, (program.outputs[0],))
                        for index in concat.inputs:
                            self._assert_contract(
                                program.tensor(index),
                                output_dtype,
                                module.destination,
                                (1, 2),
                            )
                        self._assert_contract(
                            program.tensor(program.outputs[0]),
                            output_dtype,
                            module.destination,
                            (1, 4),
                        )

                        producers = {
                            index: operator
                            for operator in program.operators
                            for index in operator.outputs
                        }
                        for position, dtype in enumerate(input_dtypes):
                            source_index = program.inputs[position]
                            self._assert_contract(
                                program.tensor(source_index),
                                dtype,
                                module.sources[position],
                                (1, 2),
                            )
                            self.assertEqual(
                                program.outputs[position + 1], source_index
                            )
                            concat_input = concat.inputs[position]
                            if dtype == output_dtype:
                                self.assertEqual(concat_input, source_index)
                            else:
                                requantize = producers[concat_input]
                                self.assertEqual(requantize.name, "QUANTIZE")
                                self.assertEqual(requantize.inputs, (source_index,))
                                self.assertLess(requantize.index, concat.index)

                        branch = producers[program.outputs[3]]
                        self.assertEqual(branch.name, "RESHAPE")
                        self.assertEqual(branch.inputs[0], program.inputs[1])
                        self._assert_contract(
                            program.tensor(program.outputs[3]),
                            input_dtypes[1],
                            module.sources[1],
                            (2, 1),
                        )

                        # Both inputs encode real [-0.5, 1.5] exactly. Execute
                        # serialized integer tensors, never the folded FX graph.
                        inputs = [
                            np.array(
                                [[int(-0.5 / scale) + zp, int(1.5 / scale) + zp]],
                                dtype=dtype,
                            )
                            for dtype, (scale, zp) in zip(input_dtypes, module.sources)
                        ]
                        scale, zp = module.destination
                        expected_half = np.array(
                            [[int(-0.5 / scale) + zp, int(1.5 / scale) + zp]],
                            dtype=output_dtype,
                        )
                        result = runtime.run(inputs, mode=ExecutionMode.NATIVE)
                        np.testing.assert_array_equal(
                            result.outputs[0],
                            np.concatenate((expected_half, expected_half), axis=1),
                        )
                        np.testing.assert_array_equal(result.outputs[1], inputs[0])
                        np.testing.assert_array_equal(result.outputs[2], inputs[1])
                        np.testing.assert_array_equal(
                            result.outputs[3], inputs[1].reshape(2, 1)
                        )


if __name__ == "__main__":
    unittest.main()
