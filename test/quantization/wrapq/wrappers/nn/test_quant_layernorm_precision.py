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

"""LayerNorm uses wide arithmetic while preserving its floating-point boundary."""

import copy
import io
import unittest

import numpy as np
import tico
import torch
from circle_schema import circle

from tico.circle import CircleDocument
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import no_quant
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_layernorm import QuantLayerNorm


def _without_fake_quant():
    return PTQConfig(activation=no_quant(), weight=no_quant())


class QuantLayerNormPrecisionTest(unittest.TestCase):
    def test_low_precision_matches_native_in_every_mode(self):
        for dtype in (torch.float16, torch.bfloat16):
            for parameter_dtype in (None, dtype, torch.float32):
                with self.subTest(dtype=dtype, parameter_dtype=parameter_dtype):
                    fp = torch.nn.LayerNorm(
                        4,
                        elementwise_affine=parameter_dtype is not None,
                        dtype=parameter_dtype,
                    ).eval()
                    if fp.weight is not None and fp.bias is not None:
                        with torch.no_grad():
                            fp.weight.copy_(torch.tensor([0.5, 1.5, 0.75, 2.0]))
                            fp.bias.copy_(torch.tensor([-0.25, 0.25, 0.5, -0.5]))
                    wrapped = QuantLayerNorm(fp, qcfg=_without_fake_quant())
                    x = torch.tensor(
                        [
                            [1000.0, -1000.0, 500.0, -500.0],
                            [100.0, 100.5, 101.0, 101.5],
                        ],
                        dtype=dtype,
                    )
                    expected = fp(x)
                    for mode in (Mode.NO_QUANT, Mode.CALIB, Mode.QUANT):
                        if mode is Mode.CALIB:
                            wrapped.enable_calibration()
                        elif mode is Mode.QUANT:
                            wrapped.freeze_qparams()
                        actual = wrapped(x)
                        self.assertIs(wrapped._mode, mode)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertTrue(torch.isfinite(actual).all())
                        # These binary-exact inputs/parameters round to the same
                        # low-precision values after native FP32 accumulation.
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_half_calibration_and_quantization_keep_finite_intermediates(self):
        x = torch.tensor([[1000.0, -1000.0, 500.0, -500.0]], dtype=torch.float16)
        wrapped = QuantLayerNorm(torch.nn.LayerNorm(4, dtype=torch.float16))
        wrapped.enable_calibration()
        torch.testing.assert_close(wrapped(x), wrapped.module(x), rtol=0, atol=0)
        wrapped.freeze_qparams()
        self.assertIs(wrapped._mode, Mode.QUANT)
        for _, observer in wrapped.named_observers():
            state = observer.require_frozen()
            assert state.scale is not None and state.zero_point is not None
            self.assertTrue(torch.isfinite(state.scale).all())
            self.assertEqual(state.zero_point.dtype, torch.int32)
        square = wrapped.obs_square.require_frozen()
        assert square.clipping_bounds is not None
        self.assertEqual(square.clipping_bounds[1].dtype, torch.float32)
        self.assertEqual(square.clipping_bounds[1].item(), 1_000_000.0)
        actual = wrapped(x)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertGreater(actual.abs().min().item(), 0.5)
        # The default 8-bit grids introduce a few rounding steps; 0.03 is
        # under three steps of the final approximately 0.01-wide output grid.
        torch.testing.assert_close(actual, wrapped.module(x), rtol=0, atol=0.03)

    def test_small_epsilon_survives_half_move(self):
        fp = torch.nn.LayerNorm(4, eps=1e-8).half().eval()
        wrapped = QuantLayerNorm(fp, qcfg=_without_fake_quant()).half()
        x = torch.ones(1, 4, dtype=torch.float16)
        expected = fp(x)
        for mode in (Mode.NO_QUANT, Mode.CALIB, Mode.QUANT):
            if mode is Mode.CALIB:
                wrapped.enable_calibration()
            elif mode is Mode.QUANT:
                wrapped.freeze_qparams()
            torch.testing.assert_close(wrapped(x), expected, rtol=0, atol=0)

        # Keep the arithmetic observers active while isolating this invariant
        # from low-precision all-zero parameter/output quantization grids.
        active = QuantLayerNorm(
            copy.deepcopy(fp),
            qcfg=PTQConfig(weight=no_quant(), overrides={"act_out": no_quant()}),
        ).half()
        active.enable_calibration()
        active(x)
        active.freeze_qparams()
        epsilon = active.obs_eps.require_frozen()
        assert epsilon.clipping_bounds is not None
        self.assertEqual(epsilon.clipping_bounds[0].dtype, torch.float32)
        self.assertEqual(epsilon.clipping_bounds[0].item(), torch.tensor(fp.eps).item())
        torch.testing.assert_close(active(x), expected, rtol=0, atol=0)

    def test_double_precision_epsilon_and_output(self):
        fp = torch.nn.LayerNorm(4, eps=1.234567890123e-7, dtype=torch.float64).eval()
        wrapped = QuantLayerNorm(fp, qcfg=_without_fake_quant())
        x = torch.tensor([[0.0, 0.0001, 0.0002, 0.0003]], dtype=torch.float64)
        for mode in (Mode.NO_QUANT, Mode.CALIB, Mode.QUANT):
            if mode is Mode.CALIB:
                wrapped.enable_calibration()
            elif mode is Mode.QUANT:
                wrapped.freeze_qparams()
            actual = wrapped(x)
            self.assertEqual(actual.dtype, torch.float64)
            torch.testing.assert_close(actual, fp(x), rtol=1e-14, atol=1e-14)

    def test_invalid_affine_dtype_mixtures_reject_before_observation(self):
        for input_dtype, weight_dtype, bias_dtype in (
            (torch.float32, torch.float64, torch.float64),
            (torch.float64, torch.float32, torch.float32),
            (torch.float16, torch.bfloat16, torch.bfloat16),
            (torch.float16, torch.float32, torch.float16),
        ):
            with self.subTest(
                input_dtype=input_dtype,
                weight_dtype=weight_dtype,
                bias_dtype=bias_dtype,
            ):
                fp = torch.nn.LayerNorm(4, dtype=weight_dtype)
                fp.bias = torch.nn.Parameter(torch.zeros(4, dtype=bias_dtype))
                wrapped = QuantLayerNorm(fp, qcfg=_without_fake_quant())
                x = torch.ones(1, 4, dtype=input_dtype)
                with self.assertRaises(RuntimeError):
                    fp(x)
                for mode in (Mode.NO_QUANT, Mode.CALIB, Mode.QUANT):
                    if mode is Mode.CALIB:
                        wrapped.enable_calibration()
                    elif mode is Mode.QUANT:
                        wrapped.freeze_qparams()
                    with self.assertRaisesRegex(RuntimeError, "LayerNorm affine dtype"):
                        wrapped(x)
                    self.assertIsNone(
                        wrapped.obs_act_in.derive_candidate().observed_shape
                    )

    def test_frozen_checkpoints_preserve_half_execution(self):
        wrapped = QuantLayerNorm(torch.nn.LayerNorm(4, dtype=torch.float16)).eval()
        x = torch.tensor([[1000.0, -1000.0, 500.0, -500.0]], dtype=torch.float16)
        wrapped.enable_calibration()
        wrapped(x)
        wrapped.freeze_qparams()
        expected = wrapped(x)
        receiver = QuantLayerNorm(copy.deepcopy(wrapped.module)).eval()
        receiver.load_state_dict(copy.deepcopy(wrapped.state_dict()))
        torch.testing.assert_close(receiver(x), expected, rtol=0, atol=0)
        stream = io.BytesIO()
        torch.save(wrapped, stream)
        stream.seek(0)
        restored = torch.load(stream, map_location="cpu", weights_only=False)
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)

    def test_legacy_checkpoints_preserve_owned_grids(self):
        wrapped = QuantLayerNorm(torch.nn.LayerNorm(4)).eval()
        x = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
        wrapped.enable_calibration()
        wrapped(x)
        wrapped.freeze_qparams()
        expected = wrapped(x)
        legacy_state = copy.deepcopy(wrapped.state_dict())
        del legacy_state["_extra_state"]["layernorm_eps"]
        legacy_state["eps"] = torch.tensor(wrapped.module.eps)
        original = copy.deepcopy(legacy_state)
        for strict in (True, False):
            with self.subTest(strict=strict):
                receiver = QuantLayerNorm(copy.deepcopy(wrapped.module)).eval()
                receiver.load_state_dict(legacy_state, strict=strict)
                self.assertNotIn("eps", dict(receiver.named_buffers()))
                self.assertIs(receiver._mode, Mode.QUANT)
                torch.testing.assert_close(receiver(x), expected, rtol=0, atol=0)
                for name, observer in wrapped.named_observers():
                    restored = receiver.get_observer(name)
                    assert restored is not None
                    before, after = observer.require_frozen(), restored.require_frozen()
                    torch.testing.assert_close(
                        after.scale, before.scale, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        after.zero_point, before.zero_point, rtol=0, atol=0
                    )
        self.assertEqual(legacy_state["_extra_state"], original["_extra_state"])
        torch.testing.assert_close(legacy_state["eps"], original["eps"], rtol=0, atol=0)
        wrapped.register_buffer("eps", torch.tensor(wrapped.module.eps))
        stream = io.BytesIO()
        torch.save(wrapped, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        self.assertNotIn("eps", dict(restored.named_buffers()))
        torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)

    def test_legacy_checkpoint_epsilon_must_be_valid_and_match(self):
        for configured_epsilon, epsilon in (
            (1e-5, torch.tensor(1e-4)),
            (1e-5, torch.tensor(float("nan"))),
            (1e-5, torch.tensor(float("inf"))),
            (1e-5, torch.tensor([1e-5])),
            (1e-5, torch.tensor(0)),
            (1e-8, torch.tensor(0.0, dtype=torch.float16)),
        ):
            for strict in (True, False):
                with self.subTest(epsilon=epsilon, strict=strict):
                    wrapped = QuantLayerNorm(
                        torch.nn.LayerNorm(4, eps=configured_epsilon)
                    )
                    state = wrapped.state_dict()
                    del state["_extra_state"]["layernorm_eps"]
                    state["eps"] = epsilon
                    with self.assertRaisesRegex(
                        RuntimeError, "legacy epsilon.*reprepare"
                    ):
                        wrapped.load_state_dict(state, strict=strict)
            wrapped = QuantLayerNorm(torch.nn.LayerNorm(4, eps=configured_epsilon))
            wrapped.register_buffer("eps", epsilon)
            stream = io.BytesIO()
            torch.save(wrapped, stream)
            stream.seek(0)
            with self.assertRaisesRegex(RuntimeError, "legacy epsilon.*reprepare"):
                torch.load(stream, weights_only=False)

    def test_checkpoint_epsilon_must_match_source(self):
        wrapped = QuantLayerNorm(torch.nn.LayerNorm(4, eps=1e-8))
        receiver = QuantLayerNorm(torch.nn.LayerNorm(4, eps=1e-5))
        with self.assertRaisesRegex(RuntimeError, "epsilon does not match.*reprepare"):
            receiver.load_state_dict(wrapped.state_dict(), strict=False)

    def test_float_circle_artifact_keeps_dtype_and_values(self):
        """The supported FP32 export still executes the explicit norm graph."""
        wrapped = QuantLayerNorm(
            torch.nn.LayerNorm(4), qcfg=_without_fake_quant()
        ).eval()
        x = torch.tensor([[1000.0, -1000.0, 500.0, -500.0]])
        wrapped.enable_calibration()
        wrapped(x)
        wrapped.freeze_qparams()
        converted = tico.convert(wrapped, (x,))
        document = CircleDocument.from_bytes(converted.circle_binary)
        document.verify(raise_on_error=True)
        graph = document.model.subgraphs[0]
        self.assertEqual(
            graph.tensors[graph.inputs[0]].type, circle.TensorType.TensorType.FLOAT32
        )
        self.assertEqual(
            graph.tensors[graph.outputs[0]].type, circle.TensorType.TensorType.FLOAT32
        )
        opcodes = [
            document.model.operatorCodes[op.opcodeIndex].builtinCode
            for op in graph.operators
        ]
        self.assertIn(circle.BuiltinOperator.BuiltinOperator.MEAN, opcodes)
        self.assertIn(circle.BuiltinOperator.BuiltinOperator.RSQRT, opcodes)
        self.assertNotIn(circle.BuiltinOperator.BuiltinOperator.QUANTIZE, opcodes)
        self.assertNotIn(circle.BuiltinOperator.BuiltinOperator.DEQUANTIZE, opcodes)
        expected = wrapped.module(x).detach().numpy()
        np.testing.assert_allclose(converted(x), expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
