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

"""Signed channel-axis policies have the same numerical and export semantics."""

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization import convert, prepare
from tico.quantization.config import affine, no_quant
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.dtypes import INT16, UINT8
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme


class TestAffineChannelAxis(unittest.TestCase):
    def test_calibrated_negative_axes_match_positive_axes(self):
        for observer_type in (MinMaxObserver, EMAObserver):
            for shape in ((3,), (2, 3), (2, 3, 4)):
                for axis in range(-len(shape), 0):
                    for dtype, qscheme in (
                        (UINT8, QScheme.PER_CHANNEL_ASYMM),
                        (INT16, QScheme.PER_CHANNEL_SYMM),
                    ):
                        with self.subTest(
                            observer=observer_type.__name__,
                            shape=shape,
                            axis=axis,
                            qscheme=qscheme,
                        ):
                            negative = observer_type(
                                name="channel",
                                dtype=dtype,
                                qscheme=qscheme,
                                channel_axis=axis,
                            )
                            positive = observer_type(
                                name="channel",
                                dtype=dtype,
                                qscheme=qscheme,
                                channel_axis=axis + len(shape),
                            )
                            x = (
                                torch.arange(
                                    math.prod(shape), dtype=torch.float32
                                ).reshape(shape)
                                / 8
                                - 0.5
                            )
                            for observer in (negative, positive):
                                observer.collect(x)
                                observer.compute_qparams()
                            before = negative.require_frozen()
                            actual = negative.fake_quant(x)
                            torch.testing.assert_close(
                                actual, positive.fake_quant(x), rtol=0, atol=0
                            )
                            self.assertEqual(actual.shape, x.shape)
                            self.assertEqual(actual.dtype, x.dtype)
                            self.assertEqual(negative.channel_axis, axis)
                            after = negative.require_frozen()
                            self.assertEqual(after.policy, before.policy)
                            torch.testing.assert_close(
                                after.scale, before.scale, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                after.zero_point, before.zero_point, rtol=0, atol=0
                            )

    def test_imported_negative_axes_round_and_saturate_without_changing_readiness(self):
        x = torch.tensor([[-0.5, 0.6, 1.1], [40.0, 70.0, 150.0]])
        # Independent quantize/clamp/dequantize results on binary-exact grids.
        expected = torch.tensor([[-0.25, 0.5, 1.0], [31.625, 63.0, 125.5]])
        for observer_type in (MinMaxObserver, EMAObserver):
            for value, reference, axis in (
                (x, expected, -1),
                (x.T, expected.T, -2),
                (x[0], expected[0], -1),
            ):
                for established in (False, True):
                    with self.subTest(
                        observer=observer_type.__name__,
                        shape=value.shape,
                        axis=axis,
                        established=established,
                    ):
                        observer = observer_type(
                            name="channel",
                            qscheme=QScheme.PER_CHANNEL_ASYMM,
                            channel_axis=axis,
                        )
                        if established:
                            observer.observe_shape(value.shape)
                        observer.load_qparams(
                            torch.tensor([0.125, 0.25, 0.5]),
                            torch.tensor([2, 3, 4], dtype=torch.int32),
                        )
                        before = observer.frozen_state
                        torch.testing.assert_close(
                            observer.fake_quant(value), reference, rtol=0, atol=0
                        )
                        self.assertEqual(observer.channel_axis, axis)
                        self.assertEqual(observer.frozen_state is None, before is None)
                        if before is not None:
                            self.assertEqual(
                                observer.require_frozen().policy, before.policy
                            )

    def test_checkpoint_preserves_signed_policy_and_execution(self):
        x = torch.tensor([[-1.0, 0.25, 0.5], [0.5, 1.0, -0.5]])
        for observer_type in (MinMaxObserver, EMAObserver):
            with self.subTest(observer=observer_type.__name__):
                source = observer_type(
                    name="channel", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=-1
                )
                source.collect(x)
                source.compute_qparams()
                before = source.require_frozen()
                restored = observer_type(
                    name="channel", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=-1
                )
                restored.load_state_dict(
                    copy.deepcopy(source.state_dict()), strict=True
                )
                torch.testing.assert_close(
                    restored.fake_quant(x), source.fake_quant(x), rtol=0, atol=0
                )
                self.assertEqual(restored.channel_axis, -1)
                self.assertEqual(restored.require_frozen().policy, before.policy)
                self.assertEqual(
                    restored.require_frozen().observed_shape, before.observed_shape
                )

    def test_invalid_axes_and_channel_extents_still_reject(self):
        for axis, shape in ((-3, (2, 3)), (2, (2, 3)), (-1, ()), (-1, (2, 4))):
            with self.subTest(axis=axis, shape=shape):
                observer = MinMaxObserver(
                    name="channel", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=axis
                )
                observer.load_qparams(
                    torch.tensor([0.125, 0.25, 0.5]),
                    torch.tensor([2, 3, 4], dtype=torch.int32),
                )
                with self.assertRaisesRegex(RuntimeError, "channel extent"):
                    observer.fake_quant(torch.ones(shape))
                self.assertIsNone(observer.frozen_state)

    def test_public_linear_negative_axis_exports_the_same_circle_weight_domain(self):
        sample = torch.tensor([[0.5, -1.0, 2.0]])
        weight = torch.tensor([[0.25, 0.5, -0.125], [-0.25, 1.0, 0.5]])
        expected = np.array([[-0.625, -0.125]], dtype=np.float32)
        for axis in (-2, 0):
            with self.subTest(axis=axis), tempfile.TemporaryDirectory() as directory:
                module = torch.nn.Linear(3, 2, bias=False).eval()
                with torch.no_grad():
                    module.weight.copy_(weight)
                model = prepare(
                    module,
                    PTQConfig(
                        activation=no_quant(),
                        weight=affine(UINT8),
                        overrides={"weight": {"channel_axis": axis}},
                    ),
                )
                model(sample)
                observer = model.get_observer("weight")
                assert isinstance(observer, AffineObserverBase)
                observer.load_qparams(
                    torch.tensor([0.125, 0.25]), torch.tensor([1, 3], dtype=torch.int32)
                )
                model = convert(model).eval()
                np.testing.assert_array_equal(model(sample).detach().numpy(), expected)
                policy = observer.require_frozen().policy
                # Use public conversion with the default Circle O1 pipeline.
                with torch.no_grad():
                    converted = tico.convert(model, (sample,))
                path = Path(directory) / "linear.circle"
                converted.save(path)
                document = CircleDocument.load(path)
                self.assertFalse(document.verify().issues)
                runtime = CircleReferenceRuntime(document)
                self.assertEqual(runtime.input_tensors[0].dtype, np.dtype("float32"))
                self.assertEqual(runtime.output_tensors[0].dtype, np.dtype("float32"))
                self.assertEqual(
                    [op.name for op in runtime.program.operators], ["FULLY_CONNECTED"]
                )
                (weight_tensor,) = [
                    tensor for tensor in runtime.program.tensors if tensor.is_constant
                ]
                self.assertEqual(weight_tensor.shape, (2, 3))
                self.assertEqual(weight_tensor.dtype, np.dtype("uint8"))
                qparam = weight_tensor.quantization
                assert qparam is not None
                self.assertEqual(qparam.quantized_dimension, 0)
                self.assertEqual(qparam.scale, (0.125, 0.25))
                self.assertEqual(qparam.zero_point, (1, 3))
                np.testing.assert_array_equal(
                    weight_tensor.constant, [[3, 5, 0], [2, 7, 5]]
                )
                actual = runtime.run(
                    [sample.numpy()], mode=ExecutionMode.FAKE_QUANTIZE
                ).outputs[0]
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(observer.channel_axis, axis)
                self.assertEqual(observer.require_frozen().policy, policy)
