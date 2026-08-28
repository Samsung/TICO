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

"""Tests for quantized Clamp operand legalization."""

import unittest

import torch

from tico.passes import ops
from tico.passes.decompose_fake_quantize import DecomposeFakeQuantize
from tico.passes.remove_unused_placeholder import RemoveUnusedPlaceholder
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.legalize_quantized_clamp import LegalizeQuantizedClamp
from tico.quantization.wrapq.utils.check_missing_qparam import check_missing_qparam
from tico.serialize.quant_param import QPARAM_KEY
from tico.utils.errors import NotYetSupportedError
from tico.utils.validate_args_kwargs import ClampArgs
from torch._export.utils import get_lifted_tensor_constant


class _TensorBoundClamp(torch.nn.Module):
    """Clamp between different INT16 fake-quantization domains."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("lower", torch.tensor(-1.25))
        self.register_buffer("upper", torch.tensor(1.75))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.fake_quantize_per_tensor_affine(
            inputs,
            0.125,
            0,
            -32768,
            32767,
        )
        outputs = torch.clamp(inputs, self.lower, self.upper)
        return torch.fake_quantize_per_tensor_affine(
            outputs,
            0.25,
            0,
            -32768,
            32767,
        )


class _IdentityTensorBoundClamp(torch.nn.Module):
    """Clamp UINT8 activations by the complete real range."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("lower", torch.tensor(-float("inf")))
        self.register_buffer("upper", torch.tensor(float("inf")))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.fake_quantize_per_tensor_affine(inputs, 0.1, 128, 0, 255)
        outputs = torch.clamp(inputs, self.lower, self.upper)
        return torch.fake_quantize_per_tensor_affine(outputs, 0.1, 128, 0, 255)


class _ScalarBoundClamp(torch.nn.Module):
    """Use the scalar Clamp overload around UINT8 activations."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.fake_quantize_per_tensor_affine(inputs, 0.1, 128, 0, 255)
        outputs = torch.clamp(inputs, min=-1.0, max=1.0)
        return torch.fake_quantize_per_tensor_affine(outputs, 0.1, 128, 0, 255)


class _DynamicBoundClamp(torch.nn.Module):
    """Expose unsupported floating-point runtime Clamp bounds."""

    def forward(
        self,
        inputs: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
    ) -> torch.Tensor:
        outputs = torch.clamp(inputs, lower, upper)
        return torch.fake_quantize_per_tensor_affine(
            outputs,
            0.25,
            0,
            -32768,
            32767,
        )


def _fold_fake_quant(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
) -> torch.export.ExportedProgram:
    """Export a module and fold its Q-DQ pairs to qparam metadata."""
    exported_program = torch.export.export(module.eval(), args)
    DecomposeFakeQuantize().call(exported_program)
    FoldQuantOps().call(exported_program)
    return exported_program


def _clamp_nodes(
    exported_program: torch.export.ExportedProgram,
) -> list[torch.fx.Node]:
    """Return all remaining ATen Clamp nodes."""
    return [
        node
        for node in exported_program.graph.nodes
        if node.op == "call_function" and node.target in ops.aten.clamp
    ]


class TestLegalizeQuantizedClamp(unittest.TestCase):
    """Validate qparam alignment and offline bound quantization."""

    def test_int16_bounds_share_output_domain_and_input_is_requantized(self) -> None:
        """Finite tensor bounds should become INT16 constants in the Clamp domain."""
        exported_program = _fold_fake_quant(
            _TensorBoundClamp(),
            (torch.randn(2, 3),),
        )

        target_pass = LegalizeQuantizedClamp()
        self.assertTrue(target_pass.call(exported_program).modified)
        self.assertFalse(target_pass.call(exported_program).modified)
        RemoveUnusedPlaceholder().call(exported_program)

        clamps = _clamp_nodes(exported_program)
        self.assertEqual(len(clamps), 1)
        clamp = clamps[0]
        clamp_args = ClampArgs(*clamp.args, **clamp.kwargs)  # type: ignore[arg-type]

        self.assertEqual(
            clamp_args.input.target,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
        )
        self.assertEqual(clamp_args.input.meta[QPARAM_KEY], clamp.meta[QPARAM_KEY])

        self.assertIsInstance(clamp_args.min, torch.fx.Node)
        self.assertIsInstance(clamp_args.max, torch.fx.Node)
        lower = get_lifted_tensor_constant(exported_program, clamp_args.min)
        upper = get_lifted_tensor_constant(exported_program, clamp_args.max)
        self.assertIsNotNone(lower)
        self.assertIsNotNone(upper)
        assert lower is not None and upper is not None
        self.assertEqual(lower.dtype, torch.int16)
        self.assertEqual(upper.dtype, torch.int16)
        self.assertEqual(lower.item(), -5)
        self.assertEqual(upper.item(), 7)
        self.assertEqual(clamp_args.min.meta[QPARAM_KEY], clamp.meta[QPARAM_KEY])
        self.assertEqual(clamp_args.max.meta[QPARAM_KEY], clamp.meta[QPARAM_KEY])

        self.assertEqual(exported_program.graph_signature.inputs_to_buffers, {})
        check_missing_qparam(exported_program, strict=True)

    def test_full_uint8_range_removes_clamp_and_original_buffers(self) -> None:
        """Infinite bounds should collapse to the UINT8 dtype range and disappear."""
        exported_program = _fold_fake_quant(
            _IdentityTensorBoundClamp(),
            (torch.randn(2, 3),),
        )

        self.assertTrue(LegalizeQuantizedClamp().call(exported_program).modified)
        RemoveUnusedPlaceholder().call(exported_program)

        self.assertEqual(_clamp_nodes(exported_program), [])
        self.assertEqual(exported_program.graph_signature.inputs_to_buffers, {})
        check_missing_qparam(exported_program, strict=True)

    def test_scalar_overload_is_normalized_to_quantized_tensor_bounds(self) -> None:
        """Scalar literals should become UINT8 tensor constants with shared qparams."""
        exported_program = _fold_fake_quant(
            _ScalarBoundClamp(),
            (torch.randn(2, 3),),
        )

        LegalizeQuantizedClamp().call(exported_program)
        RemoveUnusedPlaceholder().call(exported_program)

        clamp = _clamp_nodes(exported_program)[0]
        self.assertEqual(clamp.target, torch.ops.aten.clamp.Tensor)
        args = ClampArgs(*clamp.args, **clamp.kwargs)  # type: ignore[arg-type]
        assert isinstance(args.min, torch.fx.Node)
        assert isinstance(args.max, torch.fx.Node)
        lower = get_lifted_tensor_constant(exported_program, args.min)
        upper = get_lifted_tensor_constant(exported_program, args.max)
        assert lower is not None and upper is not None
        self.assertEqual(lower.dtype, torch.uint8)
        self.assertEqual(upper.dtype, torch.uint8)
        self.assertEqual(lower.item(), 118)
        self.assertEqual(upper.item(), 138)
        check_missing_qparam(exported_program, strict=True)

    def test_dynamic_fp_bounds_are_rejected_instead_of_silently_mixed(self) -> None:
        """A quantized Clamp must not retain unquantized runtime bound tensors."""
        exported_program = _fold_fake_quant(
            _DynamicBoundClamp(),
            (
                torch.randn(2, 3),
                torch.tensor(-1.0),
                torch.tensor(1.0),
            ),
        )

        with self.assertRaisesRegex(NotYetSupportedError, "must be a constant"):
            LegalizeQuantizedClamp().call(exported_program)


if __name__ == "__main__":
    unittest.main()
