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

"""Preserve MX contracts for every serialized split output, including unused ones."""

import unittest

import tico
import torch
from circle_schema import circle
from tico.circle.io import model_from_bytes
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx


class _MXSplit(torch.nn.Module):
    def __init__(self, sizes, selected=0, dim=-1, elem_format="int8"):
        super().__init__()
        self.sizes = sizes
        self.selected = selected
        self.dim = dim
        self.elem_format = elem_format

    def forward(self, value):
        value = quantize_mx(value, self.elem_format, -1)
        return torch.split(value, self.sizes, dim=self.dim)[self.selected]


class MXSplitExportTest(unittest.TestCase):
    def test_all_split_outputs_keep_mx_axis_when_only_one_is_used(self):
        for elem_format, dtype in (
            ("int8", circle.TensorType.TensorType.MXINT8),
            ("fp4", circle.TensorType.TensorType.MXFP4),
        ):
            for selected in (0, -1):
                for optimize in (False, True):
                    with self.subTest(
                        elem_format=elem_format, selected=selected, optimize=optimize
                    ):
                        converted = tico.convert(
                            _MXSplit(
                                [32, 32, 31], selected, elem_format=elem_format
                            ).eval(),
                            (torch.ones(1, 2, 95),),
                            config=tico.CompileConfigV1(circle_optimize=optimize),
                        )
                        model = model_from_bytes(converted.circle_binary)
                        graph = model.subgraphs[0]
                        split = next(
                            op
                            for op in graph.operators
                            if model.operatorCodes[op.opcodeIndex].builtinCode
                            == circle.BuiltinOperator.BuiltinOperator.SPLIT_V
                        )
                        self.assertEqual(len(split.outputs), 3)
                        for output, size in zip(split.outputs, (32, 32, 31)):
                            tensor = graph.tensors[output]
                            self.assertEqual(tensor.type, dtype)
                            self.assertEqual(list(tensor.shape), [1, 2, size])
                            self.assertEqual(tensor.quantization.quantizedDimension, 2)
                        self.assertEqual(graph.outputs[0], split.outputs[selected])

    def test_nonblock_split_keeps_unused_output_axis(self):
        converted = tico.convert(
            _MXSplit([1, 2], dim=1).eval(),
            (torch.ones(1, 3, 64),),
        )
        model = model_from_bytes(converted.circle_binary)
        graph = model.subgraphs[0]
        split = next(
            op
            for op in graph.operators
            if model.operatorCodes[op.opcodeIndex].builtinCode
            == circle.BuiltinOperator.BuiltinOperator.SPLIT_V
        )
        for output, size in zip(split.outputs, (1, 2)):
            tensor = graph.tensors[output]
            self.assertEqual(tensor.type, circle.TensorType.TensorType.MXINT8)
            self.assertEqual(list(tensor.shape), [1, size, 64])
            self.assertEqual(tensor.quantization.quantizedDimension, 2)

    def test_unaligned_unused_split_is_rejected(self):
        # The observed first output is aligned; the unused second output cuts a
        # block and cannot be given a valid MX SplitV output contract.
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                with self.assertRaisesRegex(NotYetSupportedError, "cuts or regroups"):
                    tico.convert(
                        _MXSplit([32, 31, 33]).eval(),
                        (torch.ones(1, 2, 96),),
                        config=tico.CompileConfigV1(circle_optimize=optimize),
                    )


if __name__ == "__main__":
    unittest.main()
