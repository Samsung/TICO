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

"""Regression tests for independent width-Concat input qparams."""

from __future__ import annotations

import copy
import unittest

import torch

from tico.ops import Concat
from tico.quantization.passes.concat_qparam import is_width_direction_concat
from tico.quantization.passes.propagate_qparam_backward import PropagateQParamBackward
from tico.quantization.passes.propagate_qparam_forward import PropagateQParamForward
from tico.serialize.quant_param import QPARAM_KEY, QuantParam
from tico.utils.validate_args_kwargs import CatArgs


class WidthConcatModule(torch.nn.Module):
    """Concatenate rank-three channels-last tensors along width."""

    def __init__(self) -> None:
        super().__init__()
        self.concat = Concat(dim=1, allow_distinct_input_qparams=True)

    def forward(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        return self.concat((lhs, rhs))

    def get_example_inputs(self):
        return (
            torch.randn(1, 5, 4),
            torch.randn(1, 7, 4),
        ), {}


class RawSequenceConcatModule(torch.nn.Module):
    """Use raw torch.cat along a rank-three sequence-like dimension."""

    def forward(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat((lhs, rhs), dim=1)

    def get_example_inputs(self):
        return (
            torch.randn(1, 5, 4),
            torch.randn(1, 7, 4),
        ), {}


class ChannelConcatModule(torch.nn.Module):
    """Concatenate rank-three tensors along the last channel dimension."""

    def __init__(self) -> None:
        super().__init__()
        self.concat = Concat(dim=-1)

    def forward(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        return self.concat((lhs, rhs))

    def get_example_inputs(self):
        return (
            torch.randn(1, 5, 4),
            torch.randn(1, 5, 6),
        ), {}


def _qparam(scale: float, zero_point: int = 0) -> QuantParam:
    return QuantParam(
        scale=[scale],
        zero_point=[zero_point],
        dtype="uint8",
    )


def _export_cat(module: torch.nn.Module):
    args, kwargs = module.get_example_inputs()  # type: ignore[operator]
    with torch.no_grad():
        exported = torch.export.export(module.eval(), args, kwargs)
    cat = next(
        node
        for node in exported.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.cat.default
    )
    cat_args = CatArgs(*cat.args, **cat.kwargs)
    return exported, cat, cat_args


class WidthConcatQParamPropagationTest(unittest.TestCase):
    def test_width_detection(self) -> None:
        _, width_cat, width_args = _export_cat(WidthConcatModule())
        _, channel_cat, channel_args = _export_cat(ChannelConcatModule())
        _, raw_cat, raw_args = _export_cat(RawSequenceConcatModule())

        self.assertTrue(is_width_direction_concat(width_cat, width_args))
        self.assertFalse(is_width_direction_concat(channel_cat, channel_args))
        self.assertFalse(is_width_direction_concat(raw_cat, raw_args))

    def test_forward_does_not_replace_width_concat_output_qparam(self) -> None:
        exported, cat, args = _export_cat(WidthConcatModule())
        input_qparam = _qparam(0.125, 17)
        output_qparam = _qparam(0.25, 29)
        for tensor in args.tensors:
            tensor.meta[QPARAM_KEY] = copy.deepcopy(input_qparam)
        cat.meta[QPARAM_KEY] = copy.deepcopy(output_qparam)

        PropagateQParamForward().call(exported)

        self.assertEqual(cat.meta[QPARAM_KEY], output_qparam)
        for tensor in args.tensors:
            self.assertEqual(tensor.meta[QPARAM_KEY], input_qparam)

    def test_backward_preserves_distinct_width_concat_inputs(self) -> None:
        exported, cat, args = _export_cat(WidthConcatModule())
        lhs_qparam = _qparam(0.125, 17)
        rhs_qparam = _qparam(0.1875, 23)
        output_qparam = _qparam(0.25, 29)
        args.tensors[0].meta[QPARAM_KEY] = copy.deepcopy(lhs_qparam)
        args.tensors[1].meta[QPARAM_KEY] = copy.deepcopy(rhs_qparam)
        cat.meta[QPARAM_KEY] = copy.deepcopy(output_qparam)

        PropagateQParamBackward().call(exported)

        self.assertEqual(args.tensors[0].meta[QPARAM_KEY], lhs_qparam)
        self.assertEqual(args.tensors[1].meta[QPARAM_KEY], rhs_qparam)
        self.assertEqual(cat.meta[QPARAM_KEY], output_qparam)

    def test_non_width_concat_keeps_existing_backward_behavior(self) -> None:
        exported, cat, args = _export_cat(ChannelConcatModule())
        output_qparam = _qparam(0.25, 29)
        cat.meta[QPARAM_KEY] = copy.deepcopy(output_qparam)

        PropagateQParamBackward().call(exported)

        for tensor in args.tensors:
            self.assertEqual(tensor.meta[QPARAM_KEY], output_qparam)
            self.assertIsNot(tensor.meta[QPARAM_KEY], cat.meta[QPARAM_KEY])


if __name__ == "__main__":
    unittest.main()
