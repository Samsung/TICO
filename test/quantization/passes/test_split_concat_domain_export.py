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

"""Preserve sequential affine rounding across split/getitem and concat."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.propagate_qparam_backward import PropagateQParamBackward
from tico.quantization.passes.propagate_qparam_forward import PropagateQParamForward
from tico.serialize.quant_param import QPARAM_KEY


class _SplitConcat(torch.nn.Module):
    def __init__(self, same_domain=False, negate=False, shared=True):
        super().__init__()
        self.same_domain = same_domain
        self.negate = negate
        self.shared = shared

    def forward(self, value):
        source = torch.fake_quantize_per_tensor_affine(value, 0.125, 128, 0, 255)
        first, second = torch.split(source, [2, 2], dim=-1)
        first = torch.fake_quantize_per_tensor_affine(
            first,
            0.125 if self.same_domain else 0.5,
            128 if self.same_domain else 2,
            0,
            255,
        )
        second = torch.fake_quantize_per_tensor_affine(
            second,
            0.125 if self.same_domain else 0.75,
            128 if self.same_domain else 73,
            0,
            255,
        )
        joined = torch.cat((-second if self.negate else second, first), dim=-1)
        joined = torch.fake_quantize_per_tensor_affine(
            joined,
            0.125 if self.same_domain else 0.25,
            128 if self.same_domain else 113,
            0,
            255,
        )
        return (joined, source, first, second) if self.shared else joined


class SplitConcatDomainExportTest(unittest.TestCase):
    # All source and destination rounding decisions avoid ties. -16 saturates
    # on the first branch's asymmetric grid, independently of runtime rounding.
    sample = torch.tensor([[-16.0, -0.625, 0.625, 1.625]])

    def test_fold_discovers_split_source_before_removing_qdq(self):
        exported = torch.export.export(_SplitConcat().eval(), (self.sample,))
        DecomposeFakeQuantize().call(exported)
        FoldQuantOps().call(exported)
        quantizers = [
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        ]
        self.assertEqual(len(quantizers), 2)
        for quantizer, scale, zp in zip(quantizers, (0.5, 0.75), (2, 73)):
            self.assertEqual(quantizer.meta[QPARAM_KEY].scale, [scale])
            self.assertEqual(quantizer.meta[QPARAM_KEY].zero_point, [zp])
            split_output = quantizer.args[0]
            assert isinstance(split_output, torch.fx.Node)
            self.assertEqual(split_output.meta[QPARAM_KEY].scale, [0.125])
            self.assertEqual(split_output.meta[QPARAM_KEY].zero_point, [128])
        PropagateQParamForward().call(exported)
        PropagateQParamBackward().call(exported)
        # A downstream CAT must not retarget an established branch boundary.
        for quantizer, scale, zp in zip(quantizers, (0.5, 0.75), (2, 73)):
            self.assertEqual(quantizer.args[1:3], (scale, zp))
            self.assertEqual(quantizer.meta[QPARAM_KEY].scale, [scale])
            self.assertEqual(quantizer.meta[QPARAM_KEY].zero_point, [zp])

    def test_serialized_domains_and_saved_reference_execution(self):
        for optimize in (False, True):
            for negate in (False, True):
                for shared in (False, True):
                    with self.subTest(optimize=optimize, negate=negate, shared=shared):
                        module = _SplitConcat(negate=negate, shared=shared).eval()
                        converted = tico.convert(
                            module,
                            (self.sample,),
                            config=tico.CompileConfigV1(circle_optimize=optimize),
                        )
                        with tempfile.TemporaryDirectory() as directory:
                            path = Path(directory) / "split.circle"
                            path.write_bytes(converted.circle_binary)
                            self.assertGreater(path.stat().st_size, 0)
                            document = CircleDocument.load(path)
                            document.verify()
                            runtime = CircleReferenceRuntime(document)
                        program = runtime.program
                        split = next(
                            op
                            for op in program.operators
                            if op.name in ("SPLIT_V", "SPLIT")
                        )
                        for index in split.outputs:
                            qparam = program.tensor(index).quantization
                            assert qparam is not None
                            self.assertEqual(qparam.scale, (0.125,))
                            self.assertEqual(qparam.zero_point, (128,))
                        concat = next(
                            op for op in program.operators if op.name == "CONCATENATION"
                        )
                        for index in concat.inputs + concat.outputs:
                            qparam = program.tensor(index).quantization
                            assert qparam is not None
                            self.assertEqual(qparam.scale, (0.25,))
                            self.assertEqual(qparam.zero_point, (113,))
                        # Independent exact expectations: source->branch rounding,
                        # then target rounding. All numbers are binary-exact.
                        first = np.array([[-1.0, -0.5]], dtype=np.float32)
                        second = np.array([[0.75, 1.5]], dtype=np.float32)
                        source = self.sample.numpy()
                        joined = np.concatenate(
                            (-second if negate else second, first), axis=-1
                        )
                        expected = (
                            (joined, source, first, second) if shared else (joined,)
                        )
                        result = runtime.run([source], mode=ExecutionMode.FAKE_QUANTIZE)
                        self.assertEqual(len(result.outputs), len(expected))
                        for actual, wanted in zip(result.outputs, expected):
                            np.testing.assert_array_equal(actual, wanted)
                        if not negate:
                            encoded_source = (source / 0.125 + 128).astype(np.uint8)
                            native = runtime.run(
                                [encoded_source], mode=ExecutionMode.NATIVE
                            )
                            grids = ((0.25, 113), (0.125, 128), (0.5, 2), (0.75, 73))
                            for actual, wanted, (scale, zp) in zip(
                                native.outputs, expected, grids
                            ):
                                np.testing.assert_array_equal(
                                    actual, (wanted / scale + zp).astype(np.uint8)
                                )

    def test_same_domain_has_no_redundant_quantize(self):
        converted = tico.convert(_SplitConcat(same_domain=True).eval(), (self.sample,))
        runtime = CircleReferenceRuntime(
            CircleDocument.from_bytes(converted.circle_binary)
        )
        self.assertFalse(any(op.name == "QUANTIZE" for op in runtime.program.operators))
        result = runtime.run([self.sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE)
        np.testing.assert_array_equal(
            result.outputs[0], self.sample.numpy()[:, [2, 3, 0, 1]]
        )


if __name__ == "__main__":
    unittest.main()
