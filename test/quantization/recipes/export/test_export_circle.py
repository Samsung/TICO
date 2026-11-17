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

"""Recipe compile policies and their effect on real Circle conversion."""

import tempfile
import unittest
from pathlib import Path

import torch
from circle_schema import circle

from tico.circle import CircleDocument
from tico.config import CompileConfigV1
from tico.quantization.recipes.export.circle import (
    convert_and_save_circle,
    resolve_compile_config,
)
from tico.utils.errors import NotYetSupportedError
from tico.utils.mx.mx_ops import quantize_mx


class _MixedMul(torch.nn.Module):
    def forward(self, x, y):
        left = quantize_mx(x, "int8", -1)
        right = torch.fake_quantize_per_tensor_affine(y, 0.07, -3, -32768, 32767)
        return torch.fake_quantize_per_tensor_affine(
            left * right, 0.003, 7, -32768, 32767
        )


class _MXLayout(torch.nn.Module):
    def forward(self, x):
        value = quantize_mx(x, "int8", -1).reshape(4, 32)
        return torch.fake_quantize_per_tensor_affine(value, 0.003, 7, -32768, 32767)


class TestRecipeCompileConfig(unittest.TestCase):
    def test_omitted_options_keep_public_defaults(self):
        self.assertEqual(resolve_compile_config({}), CompileConfigV1())
        self.assertTrue(resolve_compile_config({}).move_affine_quantize_before_layout)
        self.assertFalse(resolve_compile_config({}).legalize_mx_affine_domains)

    def test_boolean_values_preserve_each_option_independently(self):
        for enabled in (True, False):
            values = {
                "preserve_embedding_quantization": enabled,
                "move_affine_quantize_before_layout": enabled,
                "legalize_mx_affine_domains": enabled,
            }
            with self.subTest(enabled=enabled):
                result = resolve_compile_config(values)
                for name in values:
                    self.assertIs(getattr(result, name), enabled)
        result = resolve_compile_config(
            {
                "preserve_embedding_quantization": True,
                "move_affine_quantize_before_layout": False,
                "legalize_mx_affine_domains": True,
            }
        )
        self.assertTrue(result.preserve_embedding_quantization)
        self.assertFalse(result.move_affine_quantize_before_layout)
        self.assertTrue(result.legalize_mx_affine_domains)

    def test_invalid_types_are_diagnosed_without_boolean_coercion(self):
        invalid_values: tuple[object, ...] = ("true", "false", 1, 0, None, [])
        for option in (
            "preserve_embedding_quantization",
            "move_affine_quantize_before_layout",
            "legalize_mx_affine_domains",
        ):
            for value in invalid_values:
                with self.subTest(option=option, value=value):
                    with self.assertRaisesRegex(
                        TypeError, rf"export\.{option} must be a boolean"
                    ):
                        resolve_compile_config({option: value})

    def _convert(self, module, inputs, options):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stage.circle"
            convert_and_save_circle(
                module, inputs, path, config=resolve_compile_config(options)
            )
            document = CircleDocument.from_bytes(path.read_bytes())
        document.verify(raise_on_error=True)
        return document

    def test_legalization_option_reaches_real_stage_conversion(self):
        value = torch.linspace(-2, 2, 64).reshape(2, 32)
        with self.assertRaisesRegex(NotYetSupportedError, "Unsupported dtype"):
            self._convert(
                _MixedMul(),
                (value, value.flip(-1)),
                {"legalize_mx_affine_domains": False},
            )
        document = self._convert(
            _MixedMul(), (value, value.flip(-1)), {"legalize_mx_affine_domains": True}
        )
        graph = document.model.subgraphs[0]
        mul = [
            op
            for op in graph.operators
            if document.model.operatorCodes[op.opcodeIndex].builtinCode
            == circle.BuiltinOperator.BuiltinOperator.MUL
        ]
        self.assertEqual(len(mul), 1)
        types = circle.TensorType.TensorType
        self.assertEqual(
            [graph.tensors[i].type for i in mul[0].inputs], [types.MXINT8, types.MXINT8]
        )
        self.assertEqual(graph.tensors[mul[0].outputs[0]].type, types.MXINT8)

    def test_layout_option_changes_real_stage_layout_domain(self):
        value = torch.linspace(-2, 2, 128).reshape(2, 64)
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                document = self._convert(
                    _MXLayout(),
                    (value,),
                    {
                        "move_affine_quantize_before_layout": enabled,
                    },
                )
                graph = document.model.subgraphs[0]
                layouts = [
                    op
                    for op in graph.operators
                    if document.model.operatorCodes[op.opcodeIndex].builtinCode
                    == circle.BuiltinOperator.BuiltinOperator.RESHAPE
                ]
                self.assertEqual(len(layouts), 1)
                types = circle.TensorType.TensorType
                expected = types.INT16 if enabled else types.MXINT8
                self.assertEqual(graph.tensors[layouts[0].inputs[0]].type, expected)
                self.assertEqual(graph.tensors[layouts[0].outputs[0]].type, expected)
