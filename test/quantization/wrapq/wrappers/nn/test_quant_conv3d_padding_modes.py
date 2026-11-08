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

"""Conv3d decomposition must preserve the module's boundary extension."""

import unittest

import numpy as np
import tico
import tico.quantization
import torch

from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.nn.quant_conv3d_decomposed import (
    QuantConv3dDecomposed,
)
from torch import nn

from test.quantization.quant_spec_helpers import make_affine_ptq_config


class QuantConv3dPaddingModesTest(unittest.TestCase):
    modes = ("zeros", "reflect", "replicate", "circular")

    @staticmethod
    def _module(mode, **kwargs):
        module = nn.Conv3d(2, 4, groups=2, padding_mode=mode, **kwargs)
        with torch.no_grad():
            module.weight.fill_(1)
            assert module.bias is not None
            module.bias.copy_(torch.tensor([-1.0, 0.0, 1.0, 2.0]))
        return module

    @staticmethod
    def _install_unit_grids(layer):
        # Exact grids isolate boundary semantics with fake quantization enabled.
        for _, observer in layer.named_observers():
            assert isinstance(observer, AffineObserverBase)
            channels = 4 if observer.channel_axis is not None else 1
            observer.load_qparams(
                torch.ones(channels), torch.zeros(channels, dtype=torch.int)
            )

    def test_padding_modes_preserve_no_quant_and_calibration_values(self):
        x = torch.arange(54, dtype=torch.float32).reshape(1, 2, 3, 3, 3)
        for mode in self.modes:
            with self.subTest(mode=mode):
                module = self._module(mode, kernel_size=3, padding=1)
                layer = QuantConv3dDecomposed(module)
                expected = module(x)
                torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
                layer.enable_calibration()
                torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)

    def test_asymmetric_same_padding_with_dilation(self):
        x = torch.arange(128, dtype=torch.float32).reshape(1, 2, 4, 4, 4)
        for mode in self.modes:
            with self.subTest(mode=mode):
                module = self._module(
                    mode, kernel_size=2, padding="same", dilation=(1, 2, 1)
                )
                layer = QuantConv3dDecomposed(module)
                torch.testing.assert_close(layer(x), module(x), rtol=0, atol=0)

    def test_quantized_padding_preserves_exactly_representable_values(self):
        x = torch.arange(54, dtype=torch.float32).reshape(1, 2, 3, 3, 3)
        config = make_affine_ptq_config(
            dtype=DType.int(16), qscheme=QScheme.PER_TENSOR_SYMM
        )
        for mode in self.modes:
            with self.subTest(mode=mode):
                module = self._module(mode, kernel_size=3, padding=1)
                layer = QuantConv3dDecomposed(module, qcfg=config)
                layer.enable_calibration()
                layer(x)
                self._install_unit_grids(layer)
                layer.freeze_qparams()
                torch.testing.assert_close(layer(x), module(x), rtol=0, atol=0)

    def test_public_prepare_convert_uses_registered_padding_policy(self):
        x = torch.arange(54, dtype=torch.float32).reshape(1, 2, 3, 3, 3)
        config = make_affine_ptq_config(
            dtype=DType.int(16), qscheme=QScheme.PER_TENSOR_SYMM
        )
        for mode in self.modes:
            with self.subTest(mode=mode):
                module = self._module(mode, kernel_size=3, padding=1).eval()
                prepared = tico.quantization.prepare(module, config, inplace=False)
                self.assertIsInstance(prepared.wrapped, QuantConv3dDecomposed)
                torch.testing.assert_close(prepared(x), module(x), rtol=0, atol=0)
                self._install_unit_grids(prepared)
                quantized = tico.quantization.convert(prepared)
                torch.testing.assert_close(quantized(x), module(x), rtol=0, atol=0)

    def test_default_circle_export_preserves_padding_and_quantization(self):
        x = torch.arange(54, dtype=torch.float32).reshape(1, 2, 3, 3, 3)
        for mode in self.modes:
            for quantized in (False, True):
                with self.subTest(mode=mode, quantized=quantized):
                    module = self._module(mode, kernel_size=3, padding=1)
                    layer = QuantConv3dDecomposed(
                        module,
                        qcfg=make_affine_ptq_config(
                            dtype=DType.int(16), qscheme=QScheme.PER_TENSOR_SYMM
                        ),
                    ).eval()
                    layer(x)
                    if quantized:
                        layer.enable_calibration()
                        layer(x)
                        self._install_unit_grids(layer)
                        layer.freeze_qparams()
                    converted = tico.convert(layer, (x,))
                    document = CircleDocument.from_bytes(converted.circle_binary)
                    self.assertTrue(document.verify().ok)
                    runtime = CircleReferenceRuntime(document)
                    dtype = np.dtype("int16" if quantized else "float32")
                    self.assertEqual(runtime.input_tensors[0].dtype, dtype)
                    self.assertEqual(runtime.output_tensors[0].dtype, dtype)
                    operators = [op.name for op in runtime.program.operators]
                    self.assertIn("DEPTHWISE_CONV_2D", operators)
                    self.assertNotIn("DEQUANTIZE", operators)
                    if quantized:
                        for tensor in (*runtime.input_tensors, *runtime.output_tensors):
                            self.assertIsNotNone(tensor.quantization)
                            assert tensor.quantization is not None
                            self.assertEqual(tensor.quantization.scale, (1.0,))
                            self.assertEqual(tensor.quantization.zero_point, (0,))
                    if quantized and mode != "zeros":
                        active = {
                            index
                            for op in runtime.program.operators
                            for index in (*op.inputs, *op.outputs)
                            if index >= 0
                        }
                        for index in active:
                            tensor = runtime.program.tensors[index]
                            if not tensor.is_constant:
                                self.assertEqual(tensor.dtype, dtype)
                                self.assertIsNotNone(tensor.quantization)
                    execution = (
                        ExecutionMode.FAKE_QUANTIZE
                        if quantized
                        else ExecutionMode.NATIVE
                    )
                    actual = runtime.run([x.numpy()], mode=execution).outputs[0]
                    np.testing.assert_array_equal(actual, module(x).detach().numpy())

    def test_invalid_boundary_extent_preserves_native_error(self):
        x = torch.ones(1, 2, 3, 3, 3)
        for mode, padding, message in (
            ("reflect", (3, 1, 1), "Padding size"),
            ("reflect", (1, 3, 1), "Padding size"),
            ("circular", (4, 1, 1), "wrapping around more than once"),
        ):
            module = self._module(mode, kernel_size=3, padding=padding)
            for candidate in (module, QuantConv3dDecomposed(module)):
                with self.subTest(
                    mode=mode, padding=padding, candidate=type(candidate).__name__
                ):
                    with self.assertRaisesRegex(RuntimeError, message):
                        candidate(x)

    def test_replication_can_extend_beyond_input_extent(self):
        x = torch.arange(54, dtype=torch.float32).reshape(1, 2, 3, 3, 3)
        module = self._module("replicate", kernel_size=3, padding=(4, 1, 4))
        layer = QuantConv3dDecomposed(module)
        torch.testing.assert_close(layer(x), module(x), rtol=0, atol=0)
