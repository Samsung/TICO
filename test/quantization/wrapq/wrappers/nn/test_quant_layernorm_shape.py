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

"""Native LayerNorm shape contracts at the WrapQ boundary."""

import tempfile
import unittest
from pathlib import Path

import tico
import torch
from circle_schema import circle
from tico.circle import CircleDocument
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import no_quant
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.wrappers.nn.quant_layernorm import QuantLayerNorm
from tico.utils.model import CircleModel
from torch import nn
from torch.nn.utils import parametrize


class _CountingWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, weight):
        self.calls += 1
        return weight * self.calls


def _set_mode(wrapper, mode, sample):
    if mode is not Mode.NO_QUANT:
        wrapper.enable_calibration()
    if mode is Mode.QUANT:
        if sample is not None:
            wrapper(sample)
        wrapper.freeze_qparams()


class TestQuantLayerNormShape(unittest.TestCase):
    def test_invalid_input_shapes_reject_before_observation(self):
        cases = (
            ((3, 4), (2, 1, 4), False),
            ((3, 4), (2, 3), False),
            ((1, 4), (2, 3, 4), True),
            ((3, 4), (4,), False),
            ((4,), (), False),
        )
        for normalized_shape, input_shape, affine in cases:
            for mode in Mode:
                with self.subTest(
                    normalized_shape=normalized_shape,
                    input_shape=input_shape,
                    affine=affine,
                    mode=mode,
                ):
                    native = nn.LayerNorm(normalized_shape, elementwise_affine=affine)
                    wrapper = QuantLayerNorm(native)
                    valid = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
                    if normalized_shape == (1, 4):
                        valid = valid[:, :1]
                    elif normalized_shape == (4,):
                        valid = valid[:, 0]
                    _set_mode(wrapper, mode, valid)
                    buffers = {n: t.clone() for n, t in wrapper.named_buffers()}
                    observed_shape = wrapper.obs_act_in.get_extra_state()[
                        "observed_shape"
                    ]
                    invalid = torch.full(input_shape, 1000.0)
                    with self.assertRaisesRegex(RuntimeError, "Given normalized_shape"):
                        native(invalid)
                    with self.assertRaisesRegex(RuntimeError, "Given normalized_shape"):
                        wrapper(invalid)
                    for name, value in wrapper.named_buffers():
                        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
                    self.assertEqual(
                        wrapper.obs_act_in.get_extra_state()["observed_shape"],
                        observed_shape,
                    )

    def test_empty_normalized_shape_rejects_in_all_modes(self):
        for mode in Mode:
            with self.subTest(mode=mode):
                native = nn.LayerNorm((), elementwise_affine=False)
                wrapper = QuantLayerNorm(
                    native,
                    qcfg=PTQConfig(activation=no_quant(), weight=no_quant()),
                )
                _set_mode(wrapper, mode, None)
                for module in (native, wrapper):
                    with self.assertRaisesRegex(
                        RuntimeError, "normalized_shape to be at least 1-dimensional"
                    ):
                        module(torch.ones(2, 3))

    def test_affine_parameters_cannot_broadcast(self):
        for name in ("weight", "bias"):
            for mode in Mode:
                with self.subTest(parameter=name, mode=mode):
                    native = nn.LayerNorm((3, 4))
                    wrapper = QuantLayerNorm(native)
                    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
                    _set_mode(wrapper, mode, x)
                    setattr(native, name, nn.Parameter(torch.ones(1, 4)))
                    buffers = {n: t.clone() for n, t in wrapper.named_buffers()}
                    for module in (native, wrapper):
                        with self.assertRaisesRegex(
                            RuntimeError, f"Expected {name} to be of same shape"
                        ):
                            module(x)
                    for key, value in wrapper.named_buffers():
                        torch.testing.assert_close(value, buffers[key], rtol=0, atol=0)

    def test_valid_ranks_and_empty_dimensions_match_native(self):
        for normalized_shape, input_shape in (
            ((4,), (4,)),
            ((3, 4), (3, 4)),
            ((3, 4), (2, 3, 4)),
            ((3, 4), (2, 5, 3, 4)),
            ((3, 4), (0, 3, 4)),
            ((0,), (2, 0)),
        ):
            for affine in (False, True):
                with self.subTest(
                    normalized_shape=normalized_shape,
                    input_shape=input_shape,
                    affine=affine,
                ):
                    native = nn.LayerNorm(normalized_shape, elementwise_affine=affine)
                    x = torch.arange(
                        torch.Size(input_shape).numel(), dtype=torch.float32
                    )
                    x = x.reshape(input_shape)
                    actual = QuantLayerNorm(native)(x)
                    self.assertEqual(actual.shape, x.shape)
                    torch.testing.assert_close(actual, native(x), rtol=1e-6, atol=1e-6)

    def test_reduction_uses_current_normalized_shape(self):
        native = nn.LayerNorm(4, elementwise_affine=False)
        wrapper = QuantLayerNorm(native)
        native.normalized_shape = (2, 2)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        torch.testing.assert_close(wrapper(x), native(x), rtol=1e-6, atol=1e-6)

    def test_validated_parametrization_is_the_executed_weight(self):
        native = nn.LayerNorm(4, bias=False)
        weight = _CountingWeight()
        parametrize.register_parametrization(native, "weight", weight)
        wrapper = QuantLayerNorm(native)
        x = torch.tensor([[1.0, 3.0, -2.0, -4.0]])
        weight.calls = 0
        expected = native(x)
        weight.calls = 0
        actual = wrapper(x)
        self.assertEqual(weight.calls, 1)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_dynamic_leading_dimensions_survive_saved_circle_export(self):
        native = nn.LayerNorm((3, 4), elementwise_affine=False).eval()
        wrapper = QuantLayerNorm(native).eval()
        x = torch.arange(72, dtype=torch.float32).reshape(2, 3, 3, 4)
        artifact = tico.convert(
            wrapper,
            (x,),
            dynamic_shapes={
                "x": {
                    0: torch.export.Dim("batch", min=1, max=4),
                    1: torch.export.Dim("tokens", min=1, max=5),
                },
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layernorm.circle"
            artifact.save(path)
            restored = CircleModel.load(str(path))
            document = CircleDocument.from_bytes(restored.circle_binary)
            self.assertTrue(document.verify().ok)
            subgraph = document.model.subgraphs[0]
            for index in (subgraph.inputs[0], subgraph.outputs[0]):
                tensor = subgraph.tensors[index]
                self.assertEqual(tuple(tensor.shapeSignature), (-1, -1, 3, 4))
                self.assertEqual(tensor.type, circle.TensorType.TensorType.FLOAT32)
            for batch, tokens in ((1, 4), (3, 2)):
                sample = torch.arange(batch * tokens * 12, dtype=torch.float32)
                sample = sample.reshape(batch, tokens, 3, 4)
                torch.testing.assert_close(
                    torch.from_numpy(restored(sample)),
                    native(sample),
                    rtol=1e-6,
                    atol=1e-6,
                )
