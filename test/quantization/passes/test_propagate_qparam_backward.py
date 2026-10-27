# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import copy
import unittest
from typing import List

import torch
import torch.ao.quantization.fx._decomposed  # register `quantize_per_tensor`
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape
from tico.quantization.passes.propagate_qparam_backward import PropagateQParamBackward
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.validate_args_kwargs import (
    CatArgs,
    PermuteArgs,
    QuantizePerTensorArgs,
    ReshapeArgs,
)


class SingleOpPropagateQParamBackwardTest(unittest.TestCase):
    """
    How to use?

    Example)

    class DummyTest(SingleOpPropagateQParamBackwardTest):
        def test_pass(self):
            self.setup(DummyNet(), target_op=torch.ops.aten.reshape.default, dtype="uint8")

            inputs = [self.target.args[0]]

            self.run_test(inputs)
    """

    initialized: bool = False

    def setup(self, mod: torch.nn.Module, target_op, scale=1.0, zp=0, dtype="uint8"):
        assert hasattr(mod, "get_example_inputs")
        self.args, self.kwargs = mod.get_example_inputs()  # type: ignore[operator]
        self.scale = scale
        self.zp = zp
        self.dtype = dtype

        with torch.no_grad():
            self.ep = torch.export.export(
                mod.eval(),
                self.args,
                self.kwargs,
            )

        # This is necessary for testing Reshape on torch 2.5
        ConvertLayoutOpToReshape().call(self.ep)

        # Find target node
        target_node = None
        for node in self.ep.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target == target_op:
                target_node = node
                break

        assert target_node is not None
        self.target = target_node

        # Set qparam to all input nodes
        qparam = QuantParam()
        qparam.scale = [self.scale]
        qparam.zero_point = [self.zp]
        qparam.dtype = self.dtype

        self.target.meta[QPARAM_KEY] = qparam

        self.initialized = True

    def run_test(self, inputs: List[torch.fx.Node]):
        assert self.initialized, "setup() must be called first"

        # Before pass
        for input_ in inputs:
            self.assertFalse(QPARAM_KEY in input_.meta)

        target_pass = PropagateQParamBackward()
        target_pass.call(self.ep)

        # After pass
        for input_ in inputs:
            self.assertTrue(QPARAM_KEY in input_.meta)
            self.assertTrue(input_.meta[QPARAM_KEY].scale == [self.scale])
            self.assertTrue(input_.meta[QPARAM_KEY].zero_point == [self.zp])
            self.assertTrue(input_.meta[QPARAM_KEY].dtype == self.dtype)


class ReshapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten.reshape.default(x, (3, 2))

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class ReshapeTest(SingleOpPropagateQParamBackwardTest):
    def test_u8(self):
        self.setup(ReshapeModule(), torch.ops.aten.reshape.default, dtype="uint8")
        args = ReshapeArgs(*self.target.args)
        inputs = [args.input]
        self.run_test(inputs)

    def test_s16(self):
        self.setup(ReshapeModule(), torch.ops.aten.reshape.default, dtype="int16")
        args = ReshapeArgs(*self.target.args)
        inputs = [args.input]
        self.run_test(inputs)


class CatModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.cat([x, y])

    def get_example_inputs(self):
        return (torch.randn(2, 3), torch.randn(2, 3)), {}


class CatTest(SingleOpPropagateQParamBackwardTest):
    def test_u8(self):
        self.setup(CatModule(), torch.ops.aten.cat.default, dtype="uint8")
        args = CatArgs(*self.target.args)
        inputs = args.tensors
        self.run_test(inputs)

    def test_s16(self):
        self.setup(CatModule(), torch.ops.aten.cat.default, dtype="int16")
        args = CatArgs(*self.target.args)
        inputs = args.tensors
        self.run_test(inputs)


class _SharedConcatSource(torch.nn.Module):
    def __init__(self, *, producer=False, view=False):
        super().__init__()
        self.producer = producer
        self.view = view

    def forward(self, left, right):
        source = torch.relu(left) if self.producer else left
        operand = source.reshape(2, 3) if self.view else source
        return torch.cat((operand, right)), source, right


class ConcatEstablishedDomainTest(unittest.TestCase):
    def test_inputs_and_shared_producers_keep_their_established_domains(self):
        for producer in (False, True):
            for view in (False, True):
                with self.subTest(producer=producer, view=view):
                    program = torch.export.export(
                        _SharedConcatSource(producer=producer, view=view),
                        (torch.ones(2, 3), torch.ones(2, 3)),
                    )
                    ConvertLayoutOpToReshape().call(program)
                    concat = next(
                        node
                        for node in program.graph.nodes
                        if node.target == torch.ops.aten.cat.default
                    )
                    output = next(
                        node for node in program.graph.nodes if node.op == "output"
                    )
                    source = output.args[0][1]
                    first, second = concat.args[0]
                    original = QuantParam(scale=[0.125], zero_point=[3], dtype="int16")
                    source.meta[QPARAM_KEY] = copy.deepcopy(original)
                    first.meta[QPARAM_KEY] = copy.deepcopy(original)
                    concat.meta[QPARAM_KEY] = QuantParam(
                        scale=[0.5], zero_point=[-7], dtype="int16"
                    )
                    second.meta[QPARAM_KEY] = copy.deepcopy(concat.meta[QPARAM_KEY])

                    PropagateQParamBackward().call(program)

                    self.assertEqual(source.meta[QPARAM_KEY], original)
                    self.assertEqual(first.meta[QPARAM_KEY], original)
                    self.assertIs(output.args[0][1], source)
                    self.assertEqual(second.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])

    def test_only_unannotated_concat_inputs_inherit_the_output_domain(self):
        program = torch.export.export(CatModule(), (torch.ones(2, 3), torch.ones(2, 3)))
        concat = next(
            node
            for node in program.graph.nodes
            if node.target == torch.ops.aten.cat.default
        )
        first, second = concat.args[0]
        original = QuantParam(scale=[0.125], zero_point=[3], dtype="int16")
        first.meta[QPARAM_KEY] = original
        concat.meta[QPARAM_KEY] = QuantParam(
            scale=[0.5], zero_point=[-7], dtype="int16"
        )

        PropagateQParamBackward().call(program)

        self.assertIs(first.meta[QPARAM_KEY], original)
        self.assertEqual(second.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])
        self.assertIsNot(second.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])


class PermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ops.aten.permute.default(x, (0, 1))

    def get_example_inputs(self):
        return (torch.randn(2, 3),), {}


class PermuteTest(SingleOpPropagateQParamBackwardTest):
    def test_u8(self):
        self.setup(PermuteModule(), torch.ops.aten.permute.default, dtype="uint8")
        args = PermuteArgs(*self.target.args)
        inputs = [args.input]
        self.run_test(inputs)

    def test_s16(self):
        self.setup(PermuteModule(), torch.ops.aten.permute.default, dtype="int16")
        args = PermuteArgs(*self.target.args)
        inputs = [args.input]
        self.run_test(inputs)


class ExplicitQuantizeConcatModule(torch.nn.Module):
    def __init__(self, dtype=torch.uint8, view_chain=False, multiple_users=False):
        super().__init__()
        self.dtype = dtype
        self.view_chain = view_chain
        self.multiple_users = multiple_users

    def forward(self, x):
        quant_min, quant_max = (
            (0, 255) if self.dtype == torch.uint8 else (-32767, 32767)
        )
        quantized = torch.ops.quantized_decomposed.quantize_per_tensor.default(
            x, 0.125, 17, quant_min, quant_max, self.dtype
        )
        concat_input = quantized
        if self.view_chain:
            concat_input = quantized.reshape(3, 2).permute(1, 0)
        result = torch.cat((concat_input, concat_input), dim=0)
        if self.multiple_users:
            return result, quantized
        return result


class UnsupportedQuantizeConcatModule(torch.nn.Module):
    def __init__(self, per_channel=False):
        super().__init__()
        self.per_channel = per_channel

    def forward(self, x, scale, zero_point):
        if self.per_channel:
            quantized = torch.ops.quantized_decomposed.quantize_per_channel.default(
                x, scale, zero_point, 0, 0, 255, torch.uint8
            )
        else:
            quantized = torch.ops.quantized_decomposed.quantize_per_tensor.tensor(
                x, scale, zero_point, 0, 255, torch.uint8
            )
        return torch.cat((quantized, quantized), dim=0)


class ExplicitQuantizePropagationTest(unittest.TestCase):
    def _export(self, dtype=torch.uint8, view_chain=False, multiple_users=False):
        module = ExplicitQuantizeConcatModule(dtype, view_chain, multiple_users)
        exported = torch.export.export(
            module.eval(), (torch.arange(6, dtype=torch.float32).reshape(2, 3),)
        )
        ConvertLayoutOpToReshape().call(exported)
        quantize = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
        )
        concat = next(
            node
            for node in exported.graph.nodes
            if node.target == torch.ops.aten.cat.default
        )
        qparam_dtype = "uint8" if dtype == torch.uint8 else "int16"
        quantize.meta[QPARAM_KEY] = QuantParam(
            scale=[0.125], zero_point=[17], dtype=qparam_dtype
        )
        concat.meta[QPARAM_KEY] = QuantParam(
            scale=[0.25], zero_point=[29], dtype=qparam_dtype
        )
        return exported, quantize, concat

    def test_retargets_arguments_and_metadata(self):
        for dtype in (torch.uint8, torch.int16):
            with self.subTest(dtype=dtype):
                exported, quantize, concat = self._export(dtype)
                original_args = QuantizePerTensorArgs(*quantize.args)
                quantize.meta["retained_metadata"] = {"value": [42]}
                retained_metadata = quantize.meta["retained_metadata"]
                # Keep range and dtype kwargs while changing a keyword scale.
                quantize.args = (original_args.tensor,)
                quantize.kwargs = {
                    "scale": original_args.scale,
                    "zero_point": original_args.zero_p,
                    "quant_min": original_args.quant_min,
                    "quant_max": original_args.quant_max,
                    "dtype": original_args.dtype,
                }

                PropagateQParamBackward().call(exported)

                self.assertEqual(quantize.kwargs["scale"], 0.25)
                self.assertEqual(quantize.kwargs["zero_point"], 29)
                self.assertEqual(quantize.args, (original_args.tensor,))
                self.assertEqual(quantize.kwargs["quant_min"], original_args.quant_min)
                self.assertEqual(quantize.kwargs["quant_max"], original_args.quant_max)
                self.assertEqual(quantize.kwargs["dtype"], original_args.dtype)
                self.assertEqual(quantize.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])
                self.assertIsNot(quantize.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])
                self.assertIsNot(
                    quantize.meta[QPARAM_KEY].scale, concat.meta[QPARAM_KEY].scale
                )
                self.assertIsNot(
                    quantize.meta[QPARAM_KEY].zero_point,
                    concat.meta[QPARAM_KEY].zero_point,
                )
                self.assertIs(quantize.meta["retained_metadata"], retained_metadata)
                quantize_nodes = [
                    node
                    for node in exported.graph.nodes
                    if node.target
                    == torch.ops.quantized_decomposed.quantize_per_tensor.default
                ]
                self.assertEqual(quantize_nodes, [quantize])
                inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
                expected_quantized = (torch.round(inputs / 0.25) + 29).to(dtype)
                self.assertTrue(
                    torch.equal(
                        exported.module()(inputs),
                        torch.cat((expected_quantized, expected_quantized)),
                    )
                )

    def test_retargets_through_view_chain_and_stops_at_quantize(self):
        exported, quantize, concat = self._export(view_chain=True)
        source = quantize.args[0]
        self.assertNotIn(QPARAM_KEY, source.meta)

        PropagateQParamBackward().call(exported)

        for node in exported.graph.nodes:
            if node.target in (
                torch.ops.aten.reshape.default,
                torch.ops.aten.permute.default,
            ):
                self.assertEqual(node.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])
        args = QuantizePerTensorArgs(*quantize.args, **quantize.kwargs)
        self.assertEqual(args.scale, 0.25)
        self.assertEqual(args.zero_p, 29)
        self.assertEqual(quantize.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])
        self.assertNotIn(QPARAM_KEY, source.meta)

    def test_integer_scale_remains_a_valid_quantize_argument(self):
        exported, quantize, concat = self._export()
        concat.meta[QPARAM_KEY].scale = [1]

        PropagateQParamBackward().call(exported)

        args = QuantizePerTensorArgs(*quantize.args, **quantize.kwargs)
        self.assertIsInstance(args.scale, float)
        self.assertEqual(args.scale, 1.0)
        self.assertEqual(quantize.meta[QPARAM_KEY], concat.meta[QPARAM_KEY])

    def test_multiple_users_keep_original_domain(self):
        for view_chain in (False, True):
            with self.subTest(view_chain=view_chain):
                exported, quantize, _ = self._export(
                    view_chain=view_chain, multiple_users=True
                )
                self.assertEqual(len(quantize.users), 2)
                original_args = quantize.args
                original_qparam = quantize.meta[QPARAM_KEY]

                PropagateQParamBackward().call(exported)

                self.assertEqual(quantize.args, original_args)
                self.assertIs(quantize.meta[QPARAM_KEY], original_qparam)

    def test_dtype_mismatch_keeps_original_domain(self):
        exported, quantize, concat = self._export()
        concat.meta[QPARAM_KEY].dtype = "int16"
        original_args = quantize.args
        original_qparam = quantize.meta[QPARAM_KEY]

        PropagateQParamBackward().call(exported)

        self.assertEqual(quantize.args, original_args)
        self.assertIs(quantize.meta[QPARAM_KEY], original_qparam)

    def test_inconsistent_storage_dtype_keeps_original_domain(self):
        exported, quantize, _ = self._export()
        quantize.update_arg(5, torch.int16)
        original_args = quantize.args
        original_qparam = quantize.meta[QPARAM_KEY]

        PropagateQParamBackward().call(exported)

        self.assertEqual(quantize.args, original_args)
        self.assertIs(quantize.meta[QPARAM_KEY], original_qparam)

    def test_invalid_affine_qparams_keep_original_domain(self):
        invalid_qparams = (
            QuantParam(scale=None, zero_point=[29], dtype="uint8"),
            QuantParam(scale=[], zero_point=[29], dtype="uint8"),
            QuantParam(scale=[0.25, 0.5], zero_point=[29], dtype="uint8"),
            QuantParam(scale=[0.25], zero_point=None, dtype="uint8"),
            QuantParam(scale=[0.25], zero_point=[], dtype="uint8"),
            QuantParam(scale=[0.25], zero_point=[29, 31], dtype="uint8"),
            QuantParam(
                scale=[0.25], zero_point=[29], quantized_dimension=0, dtype="uint8"
            ),
            QuantParam(scale=[0.25], zero_point=[29], dtype="mxint8"),
        )
        for qparam in invalid_qparams:
            for destination in (False, True):
                with self.subTest(qparam=qparam, destination=destination):
                    exported, quantize, concat = self._export()
                    node = quantize if destination else concat
                    node.meta[QPARAM_KEY] = copy.deepcopy(qparam)
                    original_args = quantize.args
                    original_qparam = quantize.meta[QPARAM_KEY]

                    PropagateQParamBackward().call(exported)

                    self.assertEqual(quantize.args, original_args)
                    self.assertIs(quantize.meta[QPARAM_KEY], original_qparam)

    def test_missing_source_qparam_keeps_original_domain(self):
        exported, quantize, concat = self._export()
        del concat.meta[QPARAM_KEY]
        original_args = quantize.args
        original_qparam = quantize.meta[QPARAM_KEY]

        PropagateQParamBackward().call(exported)

        self.assertEqual(quantize.args, original_args)
        self.assertIs(quantize.meta[QPARAM_KEY], original_qparam)

    def test_unsupported_quantize_overloads_keep_original_domain(self):
        for per_channel in (False, True):
            with self.subTest(per_channel=per_channel):
                shape = (2,) if per_channel else ()
                inputs = (
                    torch.arange(6, dtype=torch.float32).reshape(2, 3),
                    torch.full(shape, 0.125),
                    torch.full(shape, 17, dtype=torch.int64),
                )
                exported = torch.export.export(
                    UnsupportedQuantizeConcatModule(per_channel).eval(), inputs
                )
                target = (
                    torch.ops.quantized_decomposed.quantize_per_channel.default
                    if per_channel
                    else torch.ops.quantized_decomposed.quantize_per_tensor.tensor
                )
                quantize = next(
                    node for node in exported.graph.nodes if node.target == target
                )
                concat = next(
                    node
                    for node in exported.graph.nodes
                    if node.target == torch.ops.aten.cat.default
                )
                original_qparam = QuantParam(
                    scale=[0.125, 0.125] if per_channel else [0.125],
                    zero_point=[17, 17] if per_channel else [17],
                    quantized_dimension=0 if per_channel else None,
                    dtype="uint8",
                )
                quantize.meta[QPARAM_KEY] = original_qparam
                concat.meta[QPARAM_KEY] = QuantParam(
                    scale=[0.25], zero_point=[29], dtype="uint8"
                )
                original_args = quantize.args

                PropagateQParamBackward().call(exported)

                self.assertEqual(quantize.args, original_args)
                self.assertIs(quantize.meta[QPARAM_KEY], original_qparam)
