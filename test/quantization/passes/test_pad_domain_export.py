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

"""Zero PAD must preserve source codes before a genuine affine grid change."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.passes.fold_quant_ops import _inherit_qparam_from_view_chain
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.errors import InvalidArgumentError


class _Pad(torch.nn.Module):
    def __init__(
        self,
        source_grid: tuple[float, int],
        output_grid: tuple[float, int],
        *,
        shared=False,
        fill=0.0,
    ):
        super().__init__()
        self.source_grid = source_grid
        self.output_grid = output_grid
        self.shared = shared
        self.fill = fill

    def forward(self, value):
        source = torch.fake_quantize_per_tensor_affine(value, *self.source_grid, 0, 255)
        padded = torch.nn.functional.pad(source, (1, 1), value=self.fill)
        result = torch.fake_quantize_per_tensor_affine(
            padded, *self.output_grid, 0, 255
        )
        return (result, source) if self.shared else result


class PadDomainExportTest(unittest.TestCase):
    @torch.no_grad()
    def _check_saved_export(
        self,
        source_grid,
        output_grid,
        sample,
        input_codes,
        expected_codes,
        expected_real,
        *,
        shared=False,
    ):
        module = _Pad(source_grid, output_grid, shared=shared).eval()
        sample = torch.tensor(sample)
        input_codes = np.asarray(input_codes, dtype=np.uint8)
        expected_codes = np.asarray(expected_codes, dtype=np.uint8)
        expected_real = np.asarray(expected_real, dtype=np.float32)
        for optimize in (False, True):
            with self.subTest(optimize=optimize, shared=shared):
                circle = tico.convert(
                    module,
                    (sample,),
                    config=tico.CompileConfigV1(circle_optimize=optimize),
                )
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "pad.circle"
                    circle.save(path)
                    self.assertGreater(path.stat().st_size, 0)
                    document = CircleDocument.load(path)
                    document.verify(raise_on_error=True)
                    runtime = CircleReferenceRuntime(document)
                    native = runtime.run([input_codes], mode=ExecutionMode.NATIVE)
                    fake = runtime.run(
                        [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                    )
                program = runtime.program
                for index, grid in (
                    (program.inputs[0], source_grid),
                    (program.outputs[0], output_grid),
                ):
                    tensor = program.tensor(index)
                    self.assertEqual(tensor.dtype, np.uint8)
                    qparam = tensor.quantization
                    assert qparam is not None
                    self.assertEqual(qparam.scale, (grid[0],))
                    self.assertEqual(qparam.zero_point, (grid[1],))
                self.assertEqual(native.outputs[0].dtype, np.uint8)
                self.assertEqual(native.outputs[0].shape, expected_codes.shape)
                np.testing.assert_array_equal(native.outputs[0], expected_codes)
                dequantized = (
                    native.outputs[0].astype(np.float32) - output_grid[1]
                ) * output_grid[0]
                np.testing.assert_array_equal(dequantized, expected_real)
                self.assertEqual(fake.outputs[0].dtype, np.float32)
                self.assertEqual(fake.outputs[0].shape, expected_real.shape)
                np.testing.assert_array_equal(fake.outputs[0], expected_real)
                eager = module(sample)
                np.testing.assert_array_equal(
                    (eager[0] if shared else eager).numpy(), expected_real
                )
                quantizers = [op for op in program.operators if op.name == "QUANTIZE"]
                self.assertEqual(len(quantizers), int(source_grid != output_grid))
                pad = next(op for op in program.operators if op.name == "PAD")
                for index in (pad.inputs[0], pad.outputs[0]):
                    qparam = program.tensor(index).quantization
                    assert qparam is not None
                    self.assertEqual(qparam.scale, (source_grid[0],))
                    self.assertEqual(qparam.zero_point, (source_grid[1],))
                if shared:
                    self.assertEqual(len(native.outputs), 2)
                    source = program.tensor(program.outputs[1])
                    qparam = source.quantization
                    assert qparam is not None
                    self.assertEqual(qparam.scale, (source_grid[0],))
                    self.assertEqual(qparam.zero_point, (source_grid[1],))
                    np.testing.assert_array_equal(native.outputs[1], input_codes)
                    np.testing.assert_array_equal(fake.outputs[1], sample.numpy())

    def test_distinct_domain_native_codes(self):
        self._check_saved_export(
            (0.5, 127),
            (1.0, 0),
            [[1.0, 2.0]],
            [[129, 131]],
            [[0, 1, 2, 0]],
            [[0.0, 1.0, 2.0, 0.0]],
        )

    def test_distinct_domain_rounding_and_saturation(self):
        self._check_saved_export(
            (0.125, 128),
            (0.5, 3),
            [[-2.125, 1.125, 2.875]],
            [[111, 137, 151]],
            [[3, 0, 5, 9, 3]],
            [[0.0, -1.5, 1.0, 3.0, 0.0]],
        )

    def test_same_domain_has_no_redundant_quantize(self):
        self._check_saved_export(
            (0.125, 128),
            (0.125, 128),
            [[-2.125, 1.125, 2.875]],
            [[111, 137, 151]],
            [[128, 111, 137, 151, 128]],
            [[0.0, -2.125, 1.125, 2.875, 0.0]],
        )

    def test_shared_source_keeps_original_domain_and_codes(self):
        self._check_saved_export(
            (0.125, 128),
            (0.5, 3),
            [[-2.125, 1.125, 2.875]],
            [[111, 137, 151]],
            [[3, 0, 5, 9, 3]],
            [[0.0, -1.5, 1.0, 3.0, 0.0]],
            shared=True,
        )

    def test_nonzero_padding_remains_unsupported(self):
        with self.assertRaisesRegex(
            InvalidArgumentError, "Only support 0 value padding"
        ):
            tico.convert(
                _Pad((0.125, 128), (0.5, 3), fill=1.0).eval(), (torch.ones(1, 3),)
            )

    def test_affine_per_channel_source_is_not_inherited(self):
        class PadOnly(torch.nn.Module):
            def forward(self, value):
                return torch.ops.aten.constant_pad_nd.default(value, [1, 1], 0.0)

        exported = torch.export.export(PadOnly().eval(), (torch.ones(2, 3),))
        source = next(node for node in exported.graph.nodes if node.op == "placeholder")
        pad = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.constant_pad_nd.default
        )
        source.meta[QPARAM_KEY] = QuantParam(
            scale=[0.125, 0.25],
            zero_point=[128, 128],
            dtype="uint8",
            quantized_dimension=0,
        )
        self.assertFalse(_inherit_qparam_from_view_chain(pad))
        self.assertNotIn(QPARAM_KEY, pad.meta)


if __name__ == "__main__":
    unittest.main()
