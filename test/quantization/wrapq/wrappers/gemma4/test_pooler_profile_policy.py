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

"""Generated pooling grids retain their producer's execution policy."""

import copy
import unittest
from unittest import mock

import torch

from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx, no_quant
from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_pooler import (
    QuantGemma4VisionPooler,
)
from torch import nn


class _PoolerMetadata(nn.Module):
    """The wrapper owns the real pooling arithmetic; the float owner supplies size."""

    hidden_size = 8
    root_hidden_size = 8**0.5


def _pooler(qcfg=None):
    return QuantGemma4VisionPooler(_PoolerMetadata(), qcfg=qcfg).eval()


def _inputs():
    indices = torch.arange(16)
    positions = torch.stack((indices % 4, indices // 4), dim=-1).unsqueeze(0)
    hidden = torch.linspace(-1, 1, 16 * 8).reshape(1, 16, 8)
    return hidden, positions, torch.zeros(1, 16, dtype=torch.bool)


def _freeze(pooler, *, disable_before=False):
    pooler.enable_calibration()
    if disable_before:
        pooler.obs_pool_weight.disable_fake_quant()
    hidden, positions, padding = _inputs()
    with torch.no_grad():
        pooler(hidden, positions, padding, 4)
    pooler.freeze_qparams()
    return pooler


class TestPoolerProfilePolicy(unittest.TestCase):
    def _assert_profiles(self, source, *, enabled):
        initial = _state_contract(source)
        hidden, positions, _ = _inputs()
        profiles = []
        original_install = ObserverBase.install_candidate
        installed_policies = []

        def install(observer, candidate):
            if (
                observer.name == "pool_weight"
                and observer is not source.obs_pool_weight
            ):
                installed_policies.append(candidate.fake_quant_enabled)
            return original_install(observer, candidate)

        with mock.patch.object(
            ObserverBase, "install_candidate", autospec=True, side_effect=install
        ):
            for length in (4, 16):
                with self.subTest(length=length):
                    adapter = source.as_export_module(
                        output_length=length, pixel_position_ids=positions
                    ).eval()
                    observer = adapter.wrapped.obs_pool_weight
                    snapshot = observer.require_frozen()
                    self.assertIs(observer.fake_quant_enabled, enabled)
                    self.assertIs(snapshot.fake_quant_enabled, enabled)
                    self.assertEqual(snapshot.kind, "affine")
                    self.assertEqual(snapshot.provenance, "calibrated")
                    self.assertFalse(snapshot.locked)
                    self.assertEqual(snapshot.observed_shape, (1, length, 16))
                    weights = adapter.wrapped.pool_weights
                    expected_scale = weights.max() / 255
                    self.assertTrue(torch.equal(snapshot.scale, expected_scale))
                    self.assertTrue(
                        torch.equal(
                            snapshot.zero_point, torch.tensor(0, dtype=torch.int32)
                        )
                    )
                    with torch.no_grad():
                        output = adapter(hidden)
                    if not enabled:
                        with mock.patch(
                            "torch.fake_quantize_per_tensor_affine",
                            side_effect=AssertionError("disabled weight quantized"),
                        ):
                            # Calling fake_quant is valid: the disabled policy must bypass its operation.
                            self.assertIs(observer.fake_quant(weights), weights)
                    profiles.append(
                        (adapter, snapshot, output, _state_contract(adapter))
                    )
                    self.assertEqual(initial, _state_contract(source))

        self.assertEqual(installed_policies, [enabled, enabled])
        first, first_state, first_output, first_contract = profiles[0]
        second, second_state, _, _ = profiles[1]
        self.assertIsNot(first.wrapped.obs_pool_weight, second.wrapped.obs_pool_weight)
        self.assertFalse(torch.equal(first_state.scale, second_state.scale))
        with self.assertRaisesRegex(ValueError, "Cannot pool"):
            source.as_export_module(output_length=3, pixel_position_ids=positions)
        with torch.no_grad():
            torch.testing.assert_close(first(hidden), first_output, rtol=0, atol=0)
        self.assertEqual(first_contract, _state_contract(first))
        self.assertEqual(initial, _state_contract(source))

    def test_disabled_after_freeze_survives_profile_grid_installation(self):
        source = _freeze(_pooler())
        source.obs_pool_weight.disable_fake_quant()
        self._assert_profiles(source, enabled=False)

    def test_disabled_before_freeze_keeps_legitimate_calibrated_profile_grids(self):
        source = _freeze(_pooler(), disable_before=True)
        self.assertEqual(
            source.obs_pool_weight.require_frozen().provenance, "calibrated"
        )
        self._assert_profiles(source, enabled=False)

    def test_restored_disabled_checkpoint_preserves_profile_execution_policy(self):
        source = _freeze(_pooler())
        source.obs_pool_weight.disable_fake_quant()
        restored = _pooler()
        restored.load_state_dict(copy.deepcopy(source.state_dict()))
        self.assertEqual(_state_contract(source), _state_contract(restored))
        self._assert_profiles(restored, enabled=False)

    def test_enabled_profiles_derive_separate_numerical_grids(self):
        self._assert_profiles(_freeze(_pooler()), enabled=True)

    def test_imported_grid_and_lock_are_not_transferred_with_disabled_policy(self):
        source = _pooler()
        source.enable_calibration()
        source.obs_pool_weight.load_qparams(
            torch.tensor(0.125), torch.tensor(113), lock=True
        )
        hidden, positions, padding = _inputs()
        with torch.no_grad():
            source(hidden, positions, padding, 4)
        source.freeze_qparams()
        source.obs_pool_weight.disable_fake_quant()
        state = source.obs_pool_weight.require_frozen()
        self.assertTrue(state.locked)
        self.assertEqual(state.provenance, "imported")
        self._assert_profiles(source, enabled=False)

    def test_existing_weight_policy_is_a_snapshot_not_a_live_boolean(self):
        source = _freeze(_pooler())
        source.obs_pool_weight.disable_fake_quant()
        _, positions, _ = _inputs()
        first = source.as_export_module(output_length=4, pixel_position_ids=positions)
        source.obs_pool_weight.enable_fake_quant()
        second = source.as_export_module(output_length=16, pixel_position_ids=positions)
        self.assertFalse(
            first.wrapped.obs_pool_weight.require_frozen().fake_quant_enabled
        )
        self.assertTrue(
            second.wrapped.obs_pool_weight.require_frozen().fake_quant_enabled
        )

    def test_stateless_profile_kinds_preserve_policy_without_affine_grids(self):
        for spec, kind in ((no_quant(), "identity"), (mx("int8", axis=-1), "mx")):
            for enabled in (False, True):
                with self.subTest(kind=kind, enabled=enabled):
                    source = _freeze(
                        _pooler(PTQConfig(overrides={"pool_weight": spec}))
                    )
                    source.obs_pool_weight.set_fake_quant_enabled(enabled)
                    initial = _state_contract(source)
                    _, positions, _ = _inputs()
                    for length in (4, 16):
                        adapter = source.as_export_module(
                            output_length=length, pixel_position_ids=positions
                        )
                        observer = adapter.wrapped.obs_pool_weight
                        snapshot = observer.require_frozen()
                        self.assertEqual(snapshot.kind, kind)
                        self.assertEqual(
                            snapshot.policy, source.obs_pool_weight.policy_signature()
                        )
                        self.assertEqual(snapshot.provenance, "stateless")
                        self.assertIs(snapshot.fake_quant_enabled, enabled)
                        self.assertIsNone(snapshot.scale)
                        self.assertIsNone(snapshot.zero_point)
                        if not enabled:
                            with mock.patch(
                                "tico.quantization.wrapq.observers.mx.quantize_mx",
                                side_effect=AssertionError(
                                    "disabled MX weight quantized"
                                ),
                            ):
                                self.assertIs(
                                    observer.fake_quant(adapter.wrapped.pool_weights),
                                    adapter.wrapped.pool_weights,
                                )
                    self.assertEqual(initial, _state_contract(source))

    def test_percentile_activation_snapshots_do_not_copy_calibration_reservoirs(self):
        spec = affine(
            DType.uint(8),
            observer=PercentileObserver,
            max_samples=64,
            samples_per_batch=32,
        )
        source = _pooler(PTQConfig(activation=spec, weight=spec))
        source.enable_calibration()
        source.obs_pool_in.disable_fake_quant()
        hidden, positions, padding = _inputs()
        with torch.no_grad():
            source(hidden, positions, padding, 4)
        source.freeze_qparams()
        initial = _state_contract(source)
        names = (
            "obs_act_in",
            "obs_pool_in",
            "obs_pool_out",
            "obs_root_hidden_size",
            "obs_pool_matmul_out",
        )
        samples = {name: getattr(source, name)._samples_cpu.clone() for name in names}
        for length in (4, 16):
            adapter = source.as_export_module(
                output_length=length, pixel_position_ids=positions
            )
            actual = _state_contract(adapter.wrapped)
            for name in names:
                producer = getattr(source, name)
                profile = getattr(adapter.wrapped, name)
                self.assertIsNot(producer, profile)
                self.assertIsInstance(profile, PercentileObserver)
                self.assertGreater(producer.sampled_value_count, 0)
                self.assertEqual(profile.sampled_value_count, 0)
                self.assertEqual(
                    initial["observers"][name]["snapshot"],
                    actual["observers"][name]["snapshot"],
                )
                torch.testing.assert_close(
                    producer._samples_cpu, samples[name], rtol=0, atol=0
                )
            self.assertEqual(initial, _state_contract(source))

    def test_no_quant_profile_keeps_float_arithmetic_without_freezing_source(self):
        source = _pooler()
        source.obs_pool_weight.disable_fake_quant()
        hidden, positions, _ = _inputs()
        initial = copy.deepcopy(source.state_dict())
        for length in (4, 16):
            adapter = source.as_export_module(
                output_length=length, pixel_position_ids=positions
            )
            expected = adapter.wrapped.pool_weights @ hidden * source.root_hidden_size
            with mock.patch(
                "torch.fake_quantize_per_tensor_affine",
                side_effect=AssertionError("NO_QUANT profile quantized"),
            ):
                torch.testing.assert_close(adapter(hidden), expected, rtol=0, atol=0)
            self.assertIs(source._mode, Mode.NO_QUANT)
            self.assertIsNone(source.obs_pool_weight.frozen_state)
            current = source.state_dict()
            self.assertEqual(initial.keys(), current.keys())
            for name, value in initial.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(current[name], value, rtol=0, atol=0)
                else:
                    self.assertEqual(current[name], value)


if __name__ == "__main__":
    unittest.main()
