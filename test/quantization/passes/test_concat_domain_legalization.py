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

"""Concat quantization-domain legalization for every TICO Concat direction.

A Circle CONCATENATION copies integer codes, so its inputs and output must
share one dtype, scale and zero point. These tests pin the unified rule for
width-direction and other ``tico.ops.Concat`` modules: explicit Quantize on
mismatching input connections or at the output boundary, nothing for matching
domains, preserved producer domains, one reused conversion for a repeated
operand, preserved sequential requantization, idempotence, saved-Circle round
trips under O0/O1, and NATIVE/FAKE_QUANTIZE reference execution against
hand-computed integer codes.
"""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.ops import Concat
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine
from tico.quantization.passes.insert_quantize_on_dtype_mismatch import (
    InsertQuantizeOnDtypeMismatch,
)
from tico.quantization.passes.propagate_qparam_backward import PropagateQParamBackward
from tico.quantization.passes.propagate_qparam_forward import PropagateQParamForward
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.ops.quant_concat import QuantConcat
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import CatArgs

_QUANTIZE = torch.ops.quantized_decomposed.quantize_per_tensor.default
_LIMITS = {"uint8": (0, 255), "int16": (-32768, 32767)}


def _qp(dtype: str, scale: float, zero_point: int) -> QuantParam:
    return QuantParam(scale=[scale], zero_point=[zero_point], dtype=dtype)


class WidthConcatModule(torch.nn.Module):
    """Rank-three channels-last width concatenation (axis = rank - 2)."""

    def __init__(self) -> None:
        super().__init__()
        self.concat = Concat(dim=1, allow_distinct_input_qparams=True)

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return self.concat((lhs, rhs))

    def get_example_inputs(self):
        return (torch.randn(1, 5, 4), torch.randn(1, 7, 4)), {}


class ChannelConcatModule(torch.nn.Module):
    """Rank-three concatenation along the last (channel) axis."""

    def __init__(self) -> None:
        super().__init__()
        self.concat = Concat(dim=-1)

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        return self.concat((lhs, rhs))

    def get_example_inputs(self):
        return (torch.randn(1, 5, 4), torch.randn(1, 5, 6)), {}


class SharedOperandConcatModule(torch.nn.Module):
    """Repeat one operand and expose it to a graph output and another consumer."""

    def __init__(self) -> None:
        super().__init__()
        self.concat = Concat(dim=1, allow_distinct_input_qparams=True)

    def forward(self, lhs: torch.Tensor, rhs: torch.Tensor):
        return self.concat((lhs, rhs, lhs)), lhs, lhs.permute(0, 2, 1)

    def get_example_inputs(self):
        return (torch.randn(1, 5, 4), torch.randn(1, 7, 4)), {}


def _export_cat(module: torch.nn.Module):
    args, kwargs = module.get_example_inputs()  # type: ignore[operator]
    with torch.no_grad():
        exported = torch.export.export(module.eval(), args, kwargs)
    cat = next(
        node
        for node in exported.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.cat.default
    )
    return exported, cat, CatArgs(*cat.args, **cat.kwargs)


class ConcatDomainLegalizationPassTest(unittest.TestCase):
    """Pass-level contracts on real TICO Concat graphs with assigned qparams."""

    def _prepare(self, module, inputs, output):
        exported, cat, args = _export_cat(module)
        placeholders = list(dict.fromkeys(args.tensors))
        self.assertEqual(len(placeholders), len(inputs))
        for node, qparam in zip(placeholders, inputs):
            node.meta[QPARAM_KEY] = copy.deepcopy(qparam)
        cat.meta[QPARAM_KEY] = copy.deepcopy(output)
        return exported, cat, placeholders

    def _assert_quantize(self, node, source, destination: QuantParam) -> None:
        self.assertEqual(node.target, _QUANTIZE)
        minimum, maximum = _LIMITS[destination.dtype]
        assert destination.scale is not None and destination.zero_point is not None
        self.assertEqual(
            node.args,
            (
                source,
                destination.scale[0],
                destination.zero_point[0],
                minimum,
                maximum,
                getattr(torch, destination.dtype),
            ),
        )
        self.assertEqual(node.meta[QPARAM_KEY], destination)
        self.assertIsNot(node.meta[QPARAM_KEY], destination)
        self.assertEqual(node.meta["val"].shape, source.meta["val"].shape)
        self.assertEqual(node.meta["val"].dtype, getattr(torch, destination.dtype))

    def _assert_stable(self, exported) -> None:
        nodes = list(exported.graph.nodes)
        text = str(exported.graph)
        qparams = {
            node: copy.deepcopy(node.meta[QPARAM_KEY])
            for node in nodes
            if QPARAM_KEY in node.meta
        }
        InsertQuantizeOnDtypeMismatch().call(exported)
        self.assertEqual(list(exported.graph.nodes), nodes)
        self.assertEqual(str(exported.graph), text)
        for node, qparam in qparams.items():
            self.assertEqual(node.meta[QPARAM_KEY], qparam)

    def test_width_same_dtype_mismatch_converts_only_differing_input(self) -> None:
        cases = {
            "uint8 scale": (
                _qp("uint8", 0.125, 128),
                _qp("uint8", 0.0625, 128),
                _qp("uint8", 0.125, 128),
            ),
            "uint8 zero point": (
                _qp("uint8", 0.125, 128),
                _qp("uint8", 0.125, 96),
                _qp("uint8", 0.125, 128),
            ),
            "int16 scale": (
                _qp("int16", 2**-9, 0),
                _qp("int16", 2**-10, 0),
                _qp("int16", 2**-9, 0),
            ),
            "mixed dtype": (
                _qp("int16", 2**-9, 0),
                _qp("uint8", 0.0625, 128),
                _qp("int16", 2**-9, 0),
            ),
        }
        for name, (lhs, rhs, output) in cases.items():
            for module in (WidthConcatModule(), ChannelConcatModule()):
                with self.subTest(case=name, module=type(module).__name__):
                    exported, cat, (lhs_node, rhs_node) = self._prepare(
                        module, (lhs, rhs), output
                    )
                    InsertQuantizeOnDtypeMismatch().call(exported)
                    operands = tuple(cat.args[0])
                    # The matching connection is untouched, the other one is
                    # converted; producers keep their own domains.
                    self.assertIs(operands[0], lhs_node)
                    self._assert_quantize(operands[1], rhs_node, output)
                    self.assertEqual(lhs_node.meta[QPARAM_KEY], lhs)
                    self.assertEqual(rhs_node.meta[QPARAM_KEY], rhs)
                    self.assertEqual(cat.meta[QPARAM_KEY], output)
                    self.assertEqual(
                        sum(node.target == _QUANTIZE for node in exported.graph.nodes),
                        1,
                    )
                    self._assert_stable(exported)

    def test_every_differing_input_connection_is_converted(self) -> None:
        lhs, rhs = _qp("uint8", 0.125, 128), _qp("uint8", 0.0625, 96)
        output = _qp("uint8", 0.25, 64)
        exported, cat, nodes = self._prepare(WidthConcatModule(), (lhs, rhs), output)
        InsertQuantizeOnDtypeMismatch().call(exported)
        for operand, source in zip(cat.args[0], nodes):
            self._assert_quantize(operand, source, output)
        self.assertEqual(nodes[0].meta[QPARAM_KEY], lhs)
        self.assertEqual(nodes[1].meta[QPARAM_KEY], rhs)
        self._assert_stable(exported)

    def test_identical_domains_insert_nothing(self) -> None:
        domain = _qp("uint8", 0.125, 128)
        for module in (WidthConcatModule(), ChannelConcatModule()):
            with self.subTest(module=type(module).__name__):
                exported, cat, nodes = self._prepare(module, (domain, domain), domain)
                # Observer statistics are metadata, not part of the grid.
                nodes[1].meta[QPARAM_KEY].min = [-3.0]
                nodes[1].meta[QPARAM_KEY].max = [5.0]
                before = str(exported.graph)
                InsertQuantizeOnDtypeMismatch().call(exported)
                self.assertEqual(str(exported.graph), before)
                self.assertEqual(tuple(cat.args[0]), tuple(nodes))
                self.assertFalse(
                    any(node.target == _QUANTIZE for node in exported.graph.nodes)
                )

    def test_uniform_inputs_requantize_at_output_boundary(self) -> None:
        cases = {
            "uint8 grid": (_qp("uint8", 0.125, 128), _qp("uint8", 0.25, 64)),
            "int16 to uint8": (_qp("int16", 2**-9, 0), _qp("uint8", 0.25, 64)),
            "uint8 to int16": (_qp("uint8", 0.125, 128), _qp("int16", 2**-9, 0)),
        }
        for name, (shared, output) in cases.items():
            with self.subTest(case=name):
                exported, cat, nodes = self._prepare(
                    WidthConcatModule(), (shared, shared), output
                )
                graph_output = exported.graph.output_node()
                InsertQuantizeOnDtypeMismatch().call(exported)
                # Concat stays in the shared input grid; no derived grid is
                # synthesized. One Quantize moves the whole result to B.
                self.assertEqual(tuple(cat.args[0]), tuple(nodes))
                self.assertEqual(cat.meta[QPARAM_KEY], shared)
                self.assertIsNot(cat.meta[QPARAM_KEY], nodes[0].meta[QPARAM_KEY])
                (quantize,) = tuple(cat.users)
                self._assert_quantize(quantize, cat, output)
                self.assertIn(quantize, graph_output.all_input_nodes)
                self.assertNotIn(cat, graph_output.all_input_nodes)
                self.assertEqual(
                    sum(node.target == _QUANTIZE for node in exported.graph.nodes),
                    1,
                )
                self._assert_stable(exported)

    def test_repeated_operand_reuses_one_conversion_and_keeps_other_uses(self) -> None:
        lhs, rhs = _qp("uint8", 0.125, 128), _qp("uint8", 0.0625, 96)
        exported, cat, (lhs_node, rhs_node) = self._prepare(
            SharedOperandConcatModule(), (lhs, rhs), rhs
        )
        permute = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.permute.default
        )
        graph_output = exported.graph.output_node()
        InsertQuantizeOnDtypeMismatch().call(exported)
        first, second, third = tuple(cat.args[0])
        self.assertIs(first, third)
        self._assert_quantize(first, lhs_node, rhs)
        self.assertIs(second, rhs_node)
        self.assertEqual(
            sum(node.target == _QUANTIZE for node in exported.graph.nodes), 1
        )
        # The producer keeps its domain and its other consumers.
        self.assertEqual(lhs_node.meta[QPARAM_KEY], lhs)
        self.assertIs(permute.args[0], lhs_node)
        self.assertIn(lhs_node, graph_output.all_input_nodes)
        self._assert_stable(exported)

    def test_unsupported_granularity_is_rejected_explicitly(self) -> None:
        per_channel = QuantParam(
            scale=[0.125, 0.25], zero_point=[0, 0], quantized_dimension=0, dtype="uint8"
        )
        exported, _, _ = self._prepare(
            WidthConcatModule(),
            (per_channel, _qp("uint8", 0.125, 128)),
            _qp("uint8", 0.125, 128),
        )
        with self.assertRaisesRegex(NotYetSupportedError, "per-tensor affine"):
            InsertQuantizeOnDtypeMismatch().call(exported)

    def test_propagation_preserves_established_domains_for_every_direction(
        self,
    ) -> None:
        lhs, rhs, output = (
            _qp("uint8", 0.125, 17),
            _qp("uint8", 0.1875, 23),
            _qp("uint8", 0.25, 29),
        )
        for module in (WidthConcatModule(), ChannelConcatModule()):
            with self.subTest(module=type(module).__name__, direction="forward"):
                exported, cat, nodes = self._prepare(module, (lhs, lhs), output)
                PropagateQParamForward().call(exported)
                self.assertEqual(cat.meta[QPARAM_KEY], output)
                for node in nodes:
                    self.assertEqual(node.meta[QPARAM_KEY], lhs)
            with self.subTest(module=type(module).__name__, direction="backward"):
                exported, cat, nodes = self._prepare(module, (lhs, rhs), output)
                PropagateQParamBackward().call(exported)
                self.assertEqual(nodes[0].meta[QPARAM_KEY], lhs)
                self.assertEqual(nodes[1].meta[QPARAM_KEY], rhs)
                self.assertEqual(cat.meta[QPARAM_KEY], output)

    def test_propagation_fills_missing_domains_for_every_direction(self) -> None:
        domain = _qp("uint8", 0.125, 17)
        for module in (WidthConcatModule(), ChannelConcatModule()):
            with self.subTest(module=type(module).__name__, direction="forward"):
                exported, cat, args = _export_cat(module)
                for node in args.tensors:
                    node.meta[QPARAM_KEY] = copy.deepcopy(domain)
                PropagateQParamForward().call(exported)
                self.assertEqual(cat.meta[QPARAM_KEY], domain)
                self.assertIsNot(cat.meta[QPARAM_KEY], args.tensors[0].meta[QPARAM_KEY])
            with self.subTest(module=type(module).__name__, direction="backward"):
                exported, cat, args = _export_cat(module)
                cat.meta[QPARAM_KEY] = copy.deepcopy(domain)
                PropagateQParamBackward().call(exported)
                for node in args.tensors:
                    self.assertEqual(node.meta[QPARAM_KEY], domain)
                    self.assertIsNot(node.meta[QPARAM_KEY], cat.meta[QPARAM_KEY])


def _quant_config(dtype: str) -> PTQConfig:
    if dtype == "uint8":
        return PTQConfig(
            activation=affine(DType.uint(8), qscheme=QScheme.PER_TENSOR_ASYMM),
            weight=affine(DType.uint(8), qscheme=QScheme.PER_CHANNEL_ASYMM),
            strict_wrap=False,
        )
    return PTQConfig(
        activation=affine(DType.int(16), qscheme=QScheme.PER_TENSOR_SYMM),
        weight=affine(DType.int(16), qscheme=QScheme.PER_CHANNEL_SYMM),
        strict_wrap=False,
    )


class _FrozenWidthConcat(torch.nn.Module):
    """Real QuantConcat with imported, locked output qparams on fixed input grids.

    ``lhs_chain`` adds a second fake-quantization to the left branch so the
    export contains an established requantization before the concat boundary.
    """

    def __init__(self, dtype, lhs, rhs, out, *, lhs_chain=None) -> None:
        super().__init__()
        self.dtype = dtype
        self.lhs, self.rhs, self.out, self.lhs_chain = lhs, rhs, out, lhs_chain
        self.concat = QuantConcat(
            Concat(dim=1, allow_distinct_input_qparams=True), qcfg=_quant_config(dtype)
        )
        observer = self.concat.obs_act_out
        assert isinstance(observer, AffineObserverBase)
        observer.load_qparams(
            torch.tensor(out[0], dtype=torch.float32),
            torch.tensor(out[1], dtype=torch.int32),
        )
        self.concat.freeze_qparams()
        self.eval()

    def _fq(self, value, grid):
        minimum, maximum = _LIMITS[self.dtype]
        return torch.fake_quantize_per_tensor_affine(
            value, grid[0], grid[1], minimum, maximum
        )

    def forward(self, lhs, rhs):
        lhs = self._fq(lhs, self.lhs)
        if self.lhs_chain is not None:
            lhs = self._fq(lhs, self.lhs_chain)
        return self.concat((lhs, self._fq(rhs, self.rhs)))


class ConcatDomainLegalizationExportTest(unittest.TestCase):
    """Saved-Circle contracts and reference execution for width concats.

    Expected integer codes are computed by hand from the fixed grids. Every
    case except ``test_reference_tie_rounding`` avoids rounding ties, so the
    expectations do not depend on a backend's tie-breaking rule.
    """

    # (dtype, lhs grid, rhs grid, output grid, lhs codes, rhs codes,
    #  expected output codes, expected QUANTIZE (source, destination) pairs)
    CASES = {
        "uint8 scale": (
            "uint8",
            (0.125, 128),
            (0.09375, 128),
            (0.125, 128),
            [96, 131, 150, 255],
            [96, 135, 149, 255],
            [96, 131, 150, 255, 104, 133, 144, 223],
            [((0.09375, 128), (0.125, 128))],
        ),
        "uint8 zero point with saturation": (
            "uint8",
            (0.125, 128),
            (0.125, 96),
            (0.125, 128),
            [0, 128, 200, 250],
            [0, 96, 200, 250],
            [0, 128, 200, 250, 32, 128, 232, 255],
            [((0.125, 96), (0.125, 128))],
        ),
        "int16 scale": (
            "int16",
            (2**-9, 0),
            (3 * 2**-11, 0),
            (2**-9, 0),
            [-32768, 7, 21, 32767],
            [-32768, 7, 21, 32767],
            [-32768, 7, 21, 32767, -24576, 5, 16, 24575],
            [((3 * 2**-11, 0), (2**-9, 0))],
        ),
        "both inputs differ": (
            "uint8",
            (0.125, 128),
            (0.09375, 128),
            (0.25, 64),
            [0, 134, 178, 254],
            [96, 136, 152, 248],
            [0, 67, 89, 127, 52, 67, 73, 109],
            [((0.125, 128), (0.25, 64)), ((0.09375, 128), (0.25, 64))],
        ),
        "identical grids": (
            "uint8",
            (0.125, 128),
            (0.125, 128),
            (0.125, 128),
            [0, 134, 178, 254],
            [96, 136, 152, 248],
            [0, 134, 178, 254, 96, 136, 152, 248],
            [],
        ),
        # All inputs share A and only the output is B: concat runs in A and one
        # Quantize moves the joined tensor to B.
        "uniform inputs with different output": (
            "uint8",
            (0.125, 128),
            (0.125, 128),
            (0.25, 64),
            [0, 134, 178, 254],
            [96, 136, 152, 248],
            [0, 67, 89, 127, 48, 68, 76, 124],
            [((0.125, 128), (0.25, 64))],
        ),
    }

    @staticmethod
    def _dequantize(codes, grid):
        return ((np.asarray(codes, dtype=np.float32) - grid[1]) * grid[0]).astype(
            np.float32
        )

    @staticmethod
    def _assert_grid(tensor, dtype, grid):
        assert tensor.quantization is not None
        return (
            tensor.dtype == np.dtype(dtype)
            and tuple(tensor.quantization.scale) == (grid[0],)
            and tuple(tensor.quantization.zero_point) == (grid[1],)
        )

    def _convert(self, module, sample, optimize):
        with torch.inference_mode():
            converted = tico.convert(
                module,
                sample,
                config=tico.CompileConfigV1(circle_optimize=optimize),
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "concat.circle"
            path.write_bytes(converted.circle_binary)
            document = CircleDocument.load(path)
            document.verify()
            return CircleReferenceRuntime(document)

    def test_saved_circle_domains_and_reference_execution(self) -> None:
        for name, case in self.CASES.items():
            dtype, lhs, rhs, out, lhs_codes, rhs_codes, expected, quantizes = case
            lhs_real = self._dequantize(lhs_codes, lhs).reshape(1, 4, 1)
            rhs_real = self._dequantize(rhs_codes, rhs).reshape(1, 4, 1)
            for optimize in (False, True):
                with self.subTest(case=name, optimize=optimize):
                    module = _FrozenWidthConcat(dtype, lhs, rhs, out)
                    runtime = self._convert(
                        module,
                        (torch.from_numpy(lhs_real), torch.from_numpy(rhs_real)),
                        optimize,
                    )
                    program = runtime.program
                    concat = next(
                        op for op in program.operators if op.name == "CONCATENATION"
                    )
                    self.assertEqual(concat.options.axis, 1)
                    concat_grid = (
                        lhs if name == "uniform inputs with different output" else out
                    )
                    for index in concat.inputs + concat.outputs:
                        self.assertTrue(
                            self._assert_grid(
                                program.tensor(index), dtype, concat_grid
                            ),
                            program.tensor(index),
                        )
                    found = [
                        (
                            tuple(program.tensor(op.inputs[0]).quantization.scale)
                            + tuple(
                                program.tensor(op.inputs[0]).quantization.zero_point
                            ),
                            tuple(program.tensor(op.outputs[0]).quantization.scale)
                            + tuple(
                                program.tensor(op.outputs[0]).quantization.zero_point
                            ),
                        )
                        for op in program.operators
                        if op.name == "QUANTIZE"
                    ]
                    self.assertEqual(
                        sorted(found),
                        sorted((src, dst) for src, dst in quantizes),
                    )
                    # Graph inputs keep their own grids.
                    self.assertTrue(
                        self._assert_grid(program.tensor(program.inputs[0]), dtype, lhs)
                    )
                    self.assertTrue(
                        self._assert_grid(program.tensor(program.inputs[1]), dtype, rhs)
                    )
                    self.assertTrue(
                        self._assert_grid(
                            program.tensor(program.outputs[0]), dtype, out
                        )
                    )

                    native = runtime.run(
                        [
                            np.asarray(lhs_codes, dtype=dtype).reshape(1, 4, 1),
                            np.asarray(rhs_codes, dtype=dtype).reshape(1, 4, 1),
                        ],
                        mode=ExecutionMode.NATIVE,
                    )
                    np.testing.assert_array_equal(
                        native.outputs[0],
                        np.asarray(expected, dtype=dtype).reshape(1, 8, 1),
                    )
                    fake = runtime.run(
                        [lhs_real, rhs_real], mode=ExecutionMode.FAKE_QUANTIZE
                    )
                    np.testing.assert_array_equal(
                        fake.outputs[0],
                        self._dequantize(expected, out).reshape(1, 8, 1),
                    )
                    # The frozen fake-quantized model is the semantic reference.
                    with torch.inference_mode():
                        eager = module(
                            torch.from_numpy(lhs_real), torch.from_numpy(rhs_real)
                        )
                    np.testing.assert_array_equal(eager.numpy(), fake.outputs[0])

    def test_existing_requantization_is_chained_before_the_concat_grid(self) -> None:
        # Left: -3 saturates to -2.125 on (0.125, 17), then rounds to -2.25 on
        # the concat grid; retargeting the branch Quantize would give -3.
        module = _FrozenWidthConcat(
            "uint8", (0.5, 127), (0.5, 127), (0.375, 29), lhs_chain=(0.125, 17)
        )
        lhs = torch.tensor([[[-3.0], [1.0]]])
        rhs = torch.tensor([[[1.5], [2.0]]])
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                runtime = self._convert(module, (lhs, rhs), optimize)
                program = runtime.program
                concat = next(
                    op for op in program.operators if op.name == "CONCATENATION"
                )
                for index in concat.inputs + concat.outputs:
                    self.assertTrue(
                        self._assert_grid(program.tensor(index), "uint8", (0.375, 29))
                    )
                producers = {
                    index: op for op in program.operators for index in op.outputs
                }
                final = producers[concat.inputs[0]]
                branch = producers[final.inputs[0]]
                self.assertEqual((final.name, branch.name), ("QUANTIZE", "QUANTIZE"))
                self.assertTrue(
                    self._assert_grid(
                        program.tensor(branch.outputs[0]), "uint8", (0.125, 17)
                    )
                )
                self.assertEqual(branch.inputs[0], program.inputs[0])
                self.assertTrue(
                    self._assert_grid(
                        program.tensor(program.inputs[0]), "uint8", (0.5, 127)
                    )
                )
                right = producers[concat.inputs[1]]
                self.assertEqual(right.name, "QUANTIZE")
                self.assertEqual(right.inputs[0], program.inputs[1])
                self.assertEqual(
                    sum(op.name == "QUANTIZE" for op in program.operators), 3
                )
                native = runtime.run(
                    [
                        np.array([[[121], [129]]], dtype=np.uint8),
                        np.array([[[130], [131]]], dtype=np.uint8),
                    ],
                    mode=ExecutionMode.NATIVE,
                )
                np.testing.assert_array_equal(
                    native.outputs[0],
                    np.array([[[23], [32], [33], [34]]], dtype=np.uint8),
                )
                fake = runtime.run(
                    [lhs.numpy(), rhs.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                )
                np.testing.assert_array_equal(
                    fake.outputs[0],
                    np.array([[[-2.25], [1.125], [1.5], [1.875]]], dtype=np.float32),
                )

    def test_reference_tie_rounding(self) -> None:
        # A 2:1 scale ratio puts every odd source code on a tie. The reference
        # runtime documents round-half-away-from-zero; this pins that semantic
        # reference only and makes no claim about an integer backend's ties.
        module = _FrozenWidthConcat("uint8", (0.125, 128), (0.0625, 128), (0.125, 128))
        lhs = torch.zeros(1, 1, 1)
        rhs = torch.tensor([[[0.0625], [-0.0625], [0.1875]]])
        runtime = self._convert(module, (lhs, rhs), True)
        native = runtime.run(
            [
                np.array([[[128]]], dtype=np.uint8),
                np.array([[[129], [127], [131]]], dtype=np.uint8),
            ],
            mode=ExecutionMode.NATIVE,
        )
        np.testing.assert_array_equal(
            native.outputs[0], np.array([[[128], [129], [127], [130]]], dtype=np.uint8)
        )


if __name__ == "__main__":
    unittest.main()
