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

"""Frozen grids are storage state; kernel support depends on the actual input."""

import copy
import io
import unittest
from unittest.mock import patch

import torch

from tico.quantization.wrapq.lifecycle import freeze_qparams, validate_frozen_model
from tico.quantization.wrapq.observers.base import (
    FrozenObserverState,
    ObserverNotReadyError,
)
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase


class _ChannelGridModule(QuantModuleBase):
    def __init__(self, observer_type=MinMaxObserver):
        super().__init__()
        self.weight = observer_type(
            name="weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )

    def _all_observers(self):
        return (self.weight,)

    def forward(self, x):
        return self._fq(x, self.weight)


def _import_grid(observer, *, establish_extent=True, scale_dtype=torch.bfloat16):
    if establish_extent:
        observer.observe_shape((3, 2))
    observer.load_qparams(
        torch.tensor([0.125, 0.25, 0.5], dtype=scale_dtype),
        torch.tensor([3, 7, 11], dtype=torch.int32),
        lock=True,
    )
    return observer


class TestAffineExecutionCompatibility(unittest.TestCase):
    def assert_snapshot_equal(
        self, expected: FrozenObserverState, actual: FrozenObserverState
    ):
        for field in (
            "kind",
            "policy",
            "provenance",
            "locked",
            "fake_quant_enabled",
            "observed_shape",
        ):
            self.assertEqual(getattr(actual, field), getattr(expected, field), field)
        for field in ("scale", "zero_point"):
            before, after = getattr(expected, field), getattr(actual, field)
            self.assertIsNotNone(before)
            self.assertIsNotNone(after)
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        self.assertIsNone(actual.clipping_bounds)
        self.assertIsNone(expected.clipping_bounds)

    def test_cpu_bfloat16_grid_is_valid_frozen_model_state(self):
        for observer_type in (MinMaxObserver, EMAObserver):
            with self.subTest(observer=observer_type.__name__):
                model = _ChannelGridModule(observer_type)
                observer = _import_grid(model.weight)
                expected = observer.require_frozen()
                self.assertEqual(expected.scale.dtype, torch.bfloat16)
                self.assertEqual(expected.scale.device.type, "cpu")
                self.assertEqual(expected.zero_point.dtype, torch.int32)
                self.assertEqual(expected.observed_shape, (3, 2))
                self.assertEqual(expected.provenance, "imported")
                self.assertTrue(expected.locked)
                self.assertTrue(expected.fake_quant_enabled)

                candidate = observer.derive_candidate()
                observer.validate_candidate(candidate)
                observer.install_candidate(candidate)
                freeze_qparams(model)
                validate_frozen_model(model)
                freeze_qparams(model)
                self.assert_snapshot_equal(expected, observer.require_frozen())
                self.assertTrue(bool(torch.isposinf(observer.min_val).all()))
                self.assertTrue(bool(torch.isneginf(observer.max_val).all()))

                receiver = _ChannelGridModule(observer_type)
                receiver.load_state_dict(copy.deepcopy(model.state_dict()))
                validate_frozen_model(receiver)
                freeze_qparams(receiver)
                self.assert_snapshot_equal(expected, receiver.weight.require_frozen())

                stream = io.BytesIO()
                torch.save(model, stream)
                stream.seek(0)
                restored = torch.load(stream, map_location="cpu", weights_only=False)
                validate_frozen_model(restored)
                freeze_qparams(restored)
                self.assert_snapshot_equal(expected, restored.weight.require_frozen())

    def test_cpu_bfloat16_pending_import_does_not_fabricate_extent(self):
        for observer_type in (MinMaxObserver, EMAObserver):
            with self.subTest(observer=observer_type.__name__):
                observer = _import_grid(
                    _ChannelGridModule(observer_type).weight, establish_extent=False
                )
                self.assertIsNone(observer.frozen_state)
                with self.assertRaisesRegex(ObserverNotReadyError, "extent is unknown"):
                    observer.derive_candidate()
                restored = _ChannelGridModule(observer_type).weight
                restored.load_state_dict(copy.deepcopy(observer.state_dict()))
                self.assertIsNone(restored.frozen_state)
                with self.assertRaisesRegex(ObserverNotReadyError, "extent is unknown"):
                    restored.derive_candidate()
                restored.collect(torch.empty(3, 0))
                state = restored.derive_candidate()
                self.assertEqual(state.observed_shape, (3, 0))
                self.assertEqual(state.scale.dtype, torch.bfloat16)
                self.assertTrue(state.locked)
                self.assertEqual(state.provenance, "imported")
                restored.install_candidate(state)
                restored.validate_loaded_state()

    def test_corrupt_cpu_bfloat16_grid_rejects_including_non_strict_restore(self):
        source = _import_grid(_ChannelGridModule().weight)
        for strict in (True, False):
            for corruption in ("scale", "policy", "extent", "missing_grid"):
                with self.subTest(strict=strict, corruption=corruption):
                    state = copy.deepcopy(source.state_dict())
                    if corruption == "scale":
                        state["_cached_scale"][0] = 0
                    elif corruption == "policy":
                        state["_extra_state"]["policy"] = ("corrupt",)
                    elif corruption == "extent":
                        state["_extra_state"]["observed_shape"] = (4, 2)
                    else:
                        del state["_cached_scale"]
                    diagnostic = {
                        "scale": "finite positive scale",
                        "policy": "incompatible frozen policy",
                        "extent": "channel extent",
                        "missing_grid": "incomplete frozen grid",
                    }[corruption]
                    with self.assertRaisesRegex(RuntimeError, diagnostic):
                        _ChannelGridModule().weight.load_state_dict(
                            state, strict=strict
                        )

    def test_enabled_cpu_bfloat16_scale_rejects_before_native_operation(self):
        observer = _import_grid(_ChannelGridModule().weight)
        expected = observer.require_frozen()
        for input_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(input_dtype=input_dtype):
                x = torch.ones(3, 2, dtype=input_dtype)
                with patch("torch.fake_quantize_per_channel_affine") as native:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "fake_quantize_per_channel_affine.*execution device cpu.*torch.bfloat16",
                    ):
                        observer.fake_quant(x)
                    native.assert_not_called()
                self.assert_snapshot_equal(expected, observer.require_frozen())

    def test_execution_check_is_metadata_only_and_uses_requested_device(self):
        observer = _import_grid(_ChannelGridModule().weight)
        expected = observer.require_frozen()
        with patch.object(observer, "derive_candidate") as derive:
            observer.validate_execution(torch.device("cuda"))
            with self.assertRaisesRegex(
                RuntimeError, "decoder.weight.*fake_quantize_per_channel_affine"
            ):
                observer.validate_execution(
                    torch.device("cpu"), module_path="decoder.weight"
                )
            derive.assert_not_called()
        self.assert_snapshot_equal(expected, observer.require_frozen())
        # This checks the scope of the CPU rule, not CUDA kernel support.
        self.assertEqual(observer.require_frozen().scale.device.type, "cpu")

    def test_disabled_bfloat16_scale_returns_original_input(self):
        observer = _import_grid(_ChannelGridModule().weight)
        observer.disable_fake_quant()
        expected = observer.require_frozen()
        x = torch.ones(3, 2)
        with patch("torch.fake_quantize_per_channel_affine") as native:
            observer.validate_execution(torch.device("cpu"))
            self.assertIs(observer.fake_quant(x), x)
            native.assert_not_called()
        self.assertFalse(observer.require_frozen().fake_quant_enabled)
        self.assert_snapshot_equal(expected, observer.require_frozen())

    def test_native_cpu_fp32_scales_support_float_and_bfloat16_inputs(self):
        observer = _import_grid(_ChannelGridModule().weight, scale_dtype=torch.float32)
        expected = observer.require_frozen()
        for input_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(input_dtype=input_dtype):
                x = torch.tensor(
                    [[-0.125, 0.25], [-0.25, 0.5], [-0.5, 1.0]], dtype=input_dtype
                )
                observer.validate_execution(torch.device("cpu"))
                actual = observer.fake_quant(x)
                torch.testing.assert_close(actual, x, rtol=0, atol=0)
                self.assert_snapshot_equal(expected, observer.require_frozen())

    def test_native_cpu_bfloat16_scales_are_unsupported_per_channel(self):
        scale = torch.tensor([0.125, 0.25, 0.5], dtype=torch.bfloat16)
        zero_point = torch.tensor([3, 7, 11], dtype=torch.int32)
        for input_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(input_dtype=input_dtype):
                # Exercise the installed Torch kernel separately from TICO's
                # compatibility error; this is not a mocked backend result.
                with self.assertRaises(RuntimeError):
                    torch.fake_quantize_per_channel_affine(
                        torch.ones(3, 2, dtype=input_dtype),
                        scale,
                        zero_point,
                        0,
                        0,
                        255,
                    )

    def test_per_tensor_bfloat16_scales_do_not_inherit_channel_restriction(self):
        observer = MinMaxObserver(name="activation")
        observer.load_qparams(
            torch.tensor(0.125, dtype=torch.bfloat16),
            torch.tensor(3, dtype=torch.int32),
        )
        expected = observer.require_frozen()
        for input_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(input_dtype=input_dtype):
                x = torch.tensor([-0.125, 0.0, 0.25], dtype=input_dtype)
                observer.validate_execution(torch.device("cpu"))
                torch.testing.assert_close(observer.fake_quant(x), x, rtol=0, atol=0)
                self.assert_snapshot_equal(expected, observer.require_frozen())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_real_cuda_execution_and_cpu_checkpoint_storage_are_distinct(self):
        observer = _import_grid(_ChannelGridModule().weight)
        expected = observer.require_frozen()
        for input_dtype in (torch.float32, torch.bfloat16):
            with self.subTest(input_dtype=input_dtype):
                x = torch.tensor(
                    [[-0.125, 0.25], [-0.25, 0.5], [-0.5, 1.0]],
                    dtype=input_dtype,
                    device="cuda",
                )
                native = torch.fake_quantize_per_channel_affine(
                    x,
                    expected.scale.to("cuda"),
                    expected.zero_point.to("cuda"),
                    0,
                    0,
                    255,
                )
                torch.testing.assert_close(
                    observer.fake_quant(x), native, rtol=0, atol=0
                )
                torch.testing.assert_close(native, x, rtol=0, atol=0)
                self.assert_snapshot_equal(expected, observer.require_frozen())

        observer.to("cuda")
        observer.validate_loaded_state()
        observer.to("cpu")
        observer.validate_loaded_state()
        self.assert_snapshot_equal(expected, observer.require_frozen())
        with self.assertRaisesRegex(RuntimeError, "execution device cpu"):
            observer.fake_quant(torch.ones(3, 2))
        self.assert_snapshot_equal(expected, observer.require_frozen())
