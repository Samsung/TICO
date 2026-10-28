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

import copy
import io
import unittest
from dataclasses import replace
from typing import Any, Optional

import torch

from tico.quantization.wrapq.dtypes import INT16, UINT8
from tico.quantization.wrapq.observers.base import ObserverNotReadyError
from tico.quantization.wrapq.observers.ema import EMAObserver
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mx import MXObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.qscheme import QScheme


def _tensor(value: Optional[torch.Tensor]) -> torch.Tensor:
    assert value is not None
    return value


class FrozenObserverStateTest(unittest.TestCase):
    def test_missing_empty_and_zero_are_distinct(self):
        for cls in (MinMaxObserver, EMAObserver, PercentileObserver):
            for dtype, scheme in (
                (UINT8, QScheme.PER_TENSOR_ASYMM),
                (INT16, QScheme.PER_TENSOR_SYMM),
            ):
                with self.subTest(observer=cls.__name__, scheme=scheme):
                    obs = cls(name="activation", dtype=dtype, qscheme=scheme)
                    with self.assertRaises(ObserverNotReadyError):
                        obs.derive_candidate()
                    obs.collect(torch.empty(0))
                    with self.assertRaises(ObserverNotReadyError):
                        obs.derive_candidate()
                    obs.collect(torch.zeros(3))
                    state = obs.derive_candidate()
                    self.assertTrue(obs.enabled)
                    self.assertIsNone(obs.frozen_state)
                    self.assertTrue(bool(torch.isfinite(_tensor(state.scale)).all()))
                    self.assertTrue(bool((_tensor(state.scale) > 0).all()))
                    self.assertEqual(state.provenance, "calibrated")

    def test_nonfinite_is_invalid_not_missing(self):
        for values in ([float("nan"), 1.0], [float("-inf"), float("inf")]):
            for cls in (MinMaxObserver, EMAObserver):
                with self.subTest(observer=cls.__name__, values=values):
                    obs = cls(name="bad")
                    obs.collect(torch.tensor(values))
                    with self.assertRaises(RuntimeError) as caught:
                        obs.derive_candidate()
                    self.assertNotIsInstance(caught.exception, ObserverNotReadyError)
                    self.assertIsNone(obs.frozen_state)
                    self.assertTrue(obs.enabled)
        percentile = PercentileObserver(name="filtered")
        percentile.collect(torch.tensor([float("nan"), -1.0, 2.0, float("inf")]))
        self.assertEqual(percentile.derive_candidate().provenance, "calibrated")

    def test_every_affine_subtype_prefers_locked_grid(self):
        for cls in (MinMaxObserver, EMAObserver, PercentileObserver):
            with self.subTest(observer=cls.__name__):
                obs = cls(name="locked")
                scale, zp = torch.tensor(0.3), torch.tensor(7)
                obs.load_qparams(scale, zp)
                scale.fill_(100)
                zp.fill_(99)
                obs.collect(torch.tensor([-100.0, 100.0]))
                candidate = obs.derive_candidate()
                self.assertEqual(
                    _tensor(candidate.scale).item(), torch.tensor(0.3).item()
                )
                self.assertEqual(_tensor(candidate.zero_point).item(), 7)
                self.assertTrue(candidate.locked)
                self.assertEqual(candidate.provenance, "imported")
                obs.compute_qparams()
                _tensor(candidate.scale).fill_(123)
                self.assertEqual(
                    _tensor(obs.require_frozen().scale).item(), torch.tensor(0.3).item()
                )

    def test_invalid_imports_leave_existing_snapshot_unchanged(self):
        for scale, zp in (
            (0.0, 0),
            (-1.0, 0),
            (float("nan"), 0),
            (float("inf"), 0),
            (0.1, 1.5),
            (0.1, float("nan")),
            (0.1, 256),
            (0.1, -1),
        ):
            with self.subTest(scale=scale, zp=zp):
                obs = MinMaxObserver(name="import")
                obs.load_qparams(torch.tensor(0.25), torch.tensor(3))
                with self.assertRaises(RuntimeError):
                    obs.load_qparams(torch.tensor(scale), torch.tensor(zp))
                self.assertEqual(_tensor(obs.require_frozen().scale).item(), 0.25)
                self.assertEqual(_tensor(obs.require_frozen().zero_point).item(), 3)

    def test_per_channel_import_requires_semantic_extent(self):
        obs = MinMaxObserver(
            name="weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )
        obs.load_qparams(torch.tensor([[0.1], [0.2]]), torch.tensor([[2], [3]]))
        self.assertIsNone(obs.frozen_state)
        self.assertEqual(obs.fake_quant(torch.ones(2, 4)).shape, (2, 4))
        self.assertIsNone(obs.frozen_state)
        with self.assertRaisesRegex(RuntimeError, "channel extent"):
            obs.fake_quant(torch.ones(3, 4))
        with self.assertRaisesRegex(ObserverNotReadyError, "extent is unknown"):
            obs.derive_candidate()
        with self.assertRaisesRegex(RuntimeError, "channel extent"):
            obs.observe_shape((3, 4))
        with self.assertRaisesRegex(ObserverNotReadyError, "extent is unknown"):
            obs.derive_candidate()
        obs.reset()
        obs.observe_shape((2, 4))
        obs.load_qparams(torch.tensor([0.1, 0.2]), torch.tensor([2, 3]))
        self.assertEqual(_tensor(obs.derive_candidate().scale).shape, (2,))

    def test_empty_input_supplies_locked_import_channel_extent(self):
        scale = torch.tensor([0.125, 0.25, 0.5])
        zero_point = torch.tensor([7, 11, 13], dtype=torch.int)
        for cls in (MinMaxObserver, EMAObserver):
            for axis in (1, -3):
                with self.subTest(observer=cls.__name__, axis=axis):
                    obs = cls(
                        name="imported_empty",
                        qscheme=QScheme.PER_CHANNEL_ASYMM,
                        channel_axis=axis,
                    )
                    obs.load_qparams(scale, zero_point, lock=True)
                    obs.collect(torch.empty(1, 3, 0, 4))

                    candidate = obs.derive_candidate()
                    self.assertEqual(candidate.observed_shape, (1, 3, 0, 4))
                    self.assertEqual(candidate.provenance, "imported")
                    self.assertTrue(candidate.locked)
                    self.assertIsNone(candidate.clipping_bounds)
                    torch.testing.assert_close(
                        _tensor(candidate.scale), scale, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        _tensor(candidate.zero_point), zero_point, rtol=0, atol=0
                    )
                    self.assertTrue(bool(torch.isposinf(obs.min_val).all()))
                    self.assertTrue(bool(torch.isneginf(obs.max_val).all()))
                    self.assertIsNone(obs.frozen_state)
                    obs.install_candidate(candidate)
                    self.assertEqual(obs.require_frozen().observed_shape, (1, 3, 0, 4))

    def test_empty_input_rejects_mismatched_locked_import_extent(self):
        for cls in (MinMaxObserver, EMAObserver):
            for axis in (1, -3):
                with self.subTest(observer=cls.__name__, axis=axis):
                    obs = cls(
                        name="mismatched_empty",
                        qscheme=QScheme.PER_CHANNEL_ASYMM,
                        channel_axis=axis,
                    )
                    obs.load_qparams(torch.ones(3), torch.zeros(3, dtype=torch.int))
                    with self.assertRaisesRegex(RuntimeError, "channel extent"):
                        obs.collect(torch.empty(1, 4, 0, 4))
                    self.assertIsNone(obs.frozen_state)
                    self.assertTrue(bool(torch.isposinf(obs.min_val).all()))
                    self.assertTrue(bool(torch.isneginf(obs.max_val).all()))
                    # A rejected shape must not poison a later compatible input.
                    obs.collect(torch.empty(1, 3, 0, 4))
                    self.assertEqual(
                        obs.derive_candidate().observed_shape, (1, 3, 0, 4)
                    )

    def test_empty_input_cannot_calibrate_unlocked_or_ordinary_channels(self):
        for cls in (MinMaxObserver, EMAObserver):
            for imported in (False, True):
                with self.subTest(observer=cls.__name__, imported=imported):
                    obs = cls(
                        name="uncalibrated_empty",
                        qscheme=QScheme.PER_CHANNEL_ASYMM,
                        channel_axis=1,
                    )
                    if imported:
                        obs.load_qparams(
                            torch.ones(3), torch.zeros(3, dtype=torch.int), lock=False
                        )
                    obs.collect(torch.empty(1, 3, 0, 4))
                    with self.assertRaises(ObserverNotReadyError):
                        obs.derive_candidate()
                    self.assertIsNone(obs.frozen_state)
                    self.assertTrue(bool(torch.isposinf(obs.min_val).all()))
                    self.assertTrue(bool(torch.isneginf(obs.max_val).all()))

    def test_empty_import_extent_roundtrips_exact_state(self):
        scale = torch.tensor([0.125, 0.25, 0.5])
        zero_point = torch.tensor([7, 11, 13], dtype=torch.int)
        for cls in (MinMaxObserver, EMAObserver):
            for finalized in (False, True):
                with self.subTest(observer=cls.__name__, finalized=finalized):
                    obs = cls(
                        name="restored_empty",
                        qscheme=QScheme.PER_CHANNEL_ASYMM,
                        channel_axis=1,
                    )
                    obs.disable_fake_quant()
                    obs.load_qparams(scale, zero_point, lock=True)
                    obs.collect(torch.empty(1, 3, 0, 4))
                    expected = obs.derive_candidate()
                    if finalized:
                        obs.install_candidate(expected)
                    receiver = cls(
                        name="restored_empty",
                        qscheme=QScheme.PER_CHANNEL_ASYMM,
                        channel_axis=1,
                    )
                    receiver.load_state_dict(copy.deepcopy(obs.state_dict()))

                    actual = receiver.derive_candidate()
                    self.assertEqual(actual.observed_shape, (1, 3, 0, 4))
                    self.assertEqual(actual.policy, expected.policy)
                    self.assertEqual(actual.provenance, "imported")
                    self.assertTrue(actual.locked)
                    self.assertFalse(actual.fake_quant_enabled)
                    self.assertIsNone(actual.clipping_bounds)
                    torch.testing.assert_close(
                        _tensor(actual.scale), scale, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        _tensor(actual.zero_point), zero_point, rtol=0, atol=0
                    )
                    self.assertTrue(bool(torch.isposinf(receiver.min_val).all()))
                    self.assertTrue(bool(torch.isneginf(receiver.max_val).all()))
                    receiver.install_candidate(actual)
                    self.assertEqual(
                        receiver.require_frozen().observed_shape, (1, 3, 0, 4)
                    )

    def test_empty_input_does_not_change_stateless_policy_readiness(self):
        for cls in (IdentityObserver, MXObserver):
            with self.subTest(observer=cls.__name__):
                obs = cls(name="stateless_empty")
                obs.collect(torch.empty(1, 3, 0, 4))
                candidate = obs.derive_candidate()
                self.assertEqual(candidate.provenance, "stateless")
                self.assertIsNone(candidate.observed_shape)
                self.assertIsNone(candidate.scale)
                obs.install_candidate(candidate)
                self.assertEqual(obs.require_frozen().provenance, "stateless")

    def test_grid_shape_axis_and_symmetric_policy_validation(self):
        obs = MinMaxObserver(name="scalar")
        with self.assertRaisesRegex(RuntimeError, "scalar grid"):
            obs.load_qparams(torch.ones(2), torch.zeros(2))
        with self.assertRaisesRegex(RuntimeError, "shape"):
            obs.load_qparams(torch.tensor(0.1), torch.zeros(2))
        symmetric = MinMaxObserver(
            name="symmetric", dtype=INT16, qscheme=QScheme.PER_TENSOR_SYMM
        )
        with self.assertRaisesRegex(RuntimeError, "symmetric zero point"):
            symmetric.load_qparams(torch.tensor(0.1), torch.tensor(1))
        invalid_axis = MinMaxObserver(name="axis", qscheme=QScheme.PER_CHANNEL_ASYMM)
        with self.assertRaisesRegex(RuntimeError, "requires an axis"):
            invalid_axis.load_qparams(torch.ones(2), torch.zeros(2))

    def test_exact_calibrated_and_imported_state_roundtrip(self):
        for cls in (MinMaxObserver, EMAObserver, PercentileObserver):
            for imported in (False, True):
                with self.subTest(observer=cls.__name__, imported=imported):
                    obs = cls(name="roundtrip")
                    obs.collect(torch.tensor([-9.0, -0.1, 0.5, 3.0]))
                    if imported:
                        obs.load_qparams(torch.tensor(0.125), torch.tensor(11))
                    obs.compute_qparams()
                    obs.disable_fake_quant()
                    receiver = cls(name="roundtrip")
                    receiver.load_state_dict(copy.deepcopy(obs.state_dict()))
                    expected, actual = obs.require_frozen(), receiver.require_frozen()
                    self.assertTrue(
                        torch.equal(_tensor(actual.scale), _tensor(expected.scale))
                    )
                    self.assertTrue(
                        torch.equal(
                            _tensor(actual.zero_point), _tensor(expected.zero_point)
                        )
                    )
                    self.assertEqual(actual.locked, expected.locked)
                    self.assertEqual(actual.policy, expected.policy)
                    self.assertEqual(actual.provenance, expected.provenance)
                    self.assertFalse(actual.fake_quant_enabled)
                    receiver.compute_qparams()
                    self.assertTrue(
                        torch.equal(
                            _tensor(receiver.require_frozen().scale),
                            _tensor(expected.scale),
                        )
                    )

    def test_dynamic_per_channel_buffers_and_owned_reads(self):
        obs = MinMaxObserver(
            name="weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )
        obs.collect(torch.tensor([[-1.0, 2.0], [-0.5, 0.75]]))
        obs.compute_qparams()
        receiver = MinMaxObserver(
            name="weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )
        receiver.load_state_dict(copy.deepcopy(obs.state_dict()))
        expected = obs.require_frozen()
        actual = receiver.to("cpu").require_frozen()
        self.assertTrue(torch.equal(_tensor(actual.scale), _tensor(expected.scale)))
        _tensor(actual.scale).zero_()
        self.assertTrue(
            torch.equal(
                _tensor(receiver.require_frozen().scale), _tensor(expected.scale)
            )
        )
        receiver.reset()
        self.assertIsNone(receiver.frozen_state)
        self.assertFalse(receiver._qparams_locked)

    def test_nonstrict_cannot_bypass_policy_version_or_grid_validation(self):
        obs = MinMaxObserver(name="policy")
        obs.collect(torch.tensor([-1.0, 2.0]))
        obs.compute_qparams()
        for strict in (False, True):
            for corrupt in (
                "policy",
                "version",
                "scale",
                "missing_grid",
                "fractional_zp",
                "provenance",
                "clipping",
            ):
                with self.subTest(strict=strict, corrupt=corrupt):
                    state = copy.deepcopy(obs.state_dict())
                    if corrupt == "policy":
                        state["_extra_state"]["policy"] = ()
                    elif corrupt == "version":
                        state["_extra_state"]["version"] = 999
                    elif corrupt == "scale":
                        state["_cached_scale"].zero_()
                    elif corrupt == "missing_grid":
                        del state["_cached_zp"]
                    elif corrupt == "fractional_zp":
                        state["_cached_zp"] = torch.tensor(1.5)
                    elif corrupt == "clipping":
                        state["_extra_state"]["clipping_bounds"] = (
                            torch.tensor(float("nan")),
                            torch.tensor(1.0),
                        )
                    else:
                        state["_extra_state"]["provenance"] = "guessed"
                    with self.assertRaises(RuntimeError):
                        MinMaxObserver(name="policy").load_state_dict(
                            state, strict=strict
                        )
        with self.assertRaisesRegex(RuntimeError, "incompatible frozen policy"):
            MinMaxObserver(name="policy", dtype=INT16).load_state_dict(
                obs.state_dict(), strict=False
            )

    def test_legacy_ranges_are_not_frozen_and_percentile_needs_samples(self):
        for cls in (MinMaxObserver, PercentileObserver):
            obs = cls(name="legacy")
            obs.load_state_dict(
                {"min_val": torch.tensor(-1.0), "max_val": torch.tensor(2.0)}
            )
            self.assertIsNone(obs.frozen_state)
            self.assertTrue(obs.enabled)
            with self.assertRaisesRegex(ObserverNotReadyError, "fresh calibration"):
                obs.derive_candidate()
            obs.reset()
            obs.collect(torch.tensor([-1.0, 1.0]))
            self.assertEqual(obs.derive_candidate().provenance, "calibrated")

    def test_bypass_roundtrip_cannot_enable_an_unobserved_grid(self):
        obs = MinMaxObserver(name="bypass")
        obs.disable_fake_quant()
        obs.compute_qparams()
        receiver = MinMaxObserver(name="bypass")
        receiver.load_state_dict(copy.deepcopy(obs.state_dict()))
        self.assertEqual(receiver.require_frozen().kind, "bypass")
        x = torch.ones(2)
        self.assertIs(receiver.fake_quant(x), x)
        with self.assertRaisesRegex(RuntimeError, "recalibrate"):
            receiver.enable_fake_quant()

    def test_stateless_policies_roundtrip_without_affine_sentinels(self):
        for cls in (IdentityObserver, MXObserver):
            with self.subTest(observer=cls.__name__):
                obs = cls(name="stateless")
                obs.collect(torch.ones(2, 3))
                obs.compute_qparams()
                receiver = cls(name="stateless")
                receiver.load_state_dict(copy.deepcopy(obs.state_dict()))
                self.assertEqual(
                    receiver.require_frozen().kind,
                    "identity" if cls is IdentityObserver else "mx",
                )
                self.assertIsNone(receiver.require_frozen().scale)
                self.assertIsNone(receiver.require_frozen().zero_point)

    def test_invalid_mx_policy_is_rejected_by_common_protocol(self):
        invalid_options: tuple[dict[str, Any], ...] = (
            {"elem_format": "bad"},
            {"axis": "bad"},
            {"round": "bad"},
            {"shared_exp_method": "bad"},
        )
        for kwargs in invalid_options:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(RuntimeError, "invalid dynamic MX"):
                    MXObserver(name="mx", **kwargs).derive_candidate()

    def test_whole_module_roundtrip_is_exact(self):
        obs = PercentileObserver(name="pickle")
        obs.load_qparams(torch.tensor(0.3), torch.tensor(7))
        payload = io.BytesIO()
        torch.save(obs, payload)
        payload.seek(0)
        restored = torch.load(payload, weights_only=False)
        restored.validate_loaded_state()
        self.assertTrue(
            torch.equal(
                restored.require_frozen().scale, _tensor(obs.require_frozen().scale)
            )
        )
        self.assertTrue(restored.require_frozen().locked)

    def test_candidate_policy_cannot_be_changed_at_install(self):
        obs = MinMaxObserver(name="candidate")
        obs.collect(torch.zeros(2))
        candidate = obs.derive_candidate()
        with self.assertRaisesRegex(RuntimeError, "policy"):
            obs.install_candidate(replace(candidate, policy=()))
        self.assertIsNone(obs.frozen_state)
        self.assertTrue(obs.enabled)

    def test_legacy_mutating_compute_override_is_rejected(self):
        class LegacyObserver(MinMaxObserver):
            def compute_qparams(self):
                raise AssertionError("must not run during planning")

        obs = LegacyObserver(name="extension")
        obs.collect(torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, "migrate compute_qparams"):
            obs.derive_candidate()

    def test_frozen_policy_and_observed_schema_are_stable(self):
        obs = MinMaxObserver(name="stable")
        obs.collect(torch.tensor([-1.0, 2.0]))
        obs.compute_qparams()
        state = obs.require_frozen()
        obs.collect(torch.ones(5, 4))
        self.assertEqual(obs.require_frozen().observed_shape, state.observed_shape)
        obs.dtype = INT16
        changed = obs.frozen_state
        assert changed is not None
        self.assertEqual(changed.policy, state.policy)
        with self.assertRaisesRegex(RuntimeError, "frozen policy changed"):
            obs.require_frozen()
        with self.assertRaisesRegex(RuntimeError, "frozen policy changed"):
            obs.fake_quant(torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, "incompatible frozen policy"):
            obs.derive_candidate()

    def test_calibration_only_percentile_restore_requires_explicit_reset(self):
        original = PercentileObserver(name="partial")
        original.collect(torch.tensor([-1.0, 3.0]))
        restored = PercentileObserver(name="partial")
        restored.load_state_dict(copy.deepcopy(original.state_dict()))
        with self.assertRaisesRegex(ObserverNotReadyError, "fresh calibration"):
            restored.derive_candidate()
        with self.assertRaisesRegex(ObserverNotReadyError, "fresh calibration"):
            restored.collect(torch.tensor([0.0, 1.0]))
        restored.reset()
        restored.collect(torch.tensor([0.0, 1.0]))
        self.assertEqual(restored.derive_candidate().provenance, "calibrated")

    def test_clipping_provenance_moves_as_owned_buffers(self):
        obs = PercentileObserver(name="clipping")
        obs.collect(torch.tensor([-1.0, 2.0]))
        obs.compute_qparams()
        expected = obs.require_frozen()
        obs.to(dtype=torch.float64)
        actual = obs.require_frozen()
        assert (
            actual.clipping_bounds is not None and expected.clipping_bounds is not None
        )
        self.assertEqual(actual.clipping_bounds[0].dtype, torch.float64)
        self.assertTrue(
            torch.equal(actual.clipping_bounds[0], expected.clipping_bounds[0].double())
        )

    def test_stateless_refreeze_cannot_silently_change_frozen_policy(self):
        for observer in (IdentityObserver(name="float"), MXObserver(name="mx")):
            with self.subTest(observer=type(observer).__name__):
                observer.compute_qparams()
                before = observer.require_frozen()
                if isinstance(observer, MXObserver):
                    observer.round = "floor"
                else:
                    observer.dtype = INT16
                with self.assertRaisesRegex(RuntimeError, "frozen policy changed"):
                    observer.derive_candidate()
                with self.assertRaisesRegex(RuntimeError, "frozen policy changed"):
                    observer.compute_qparams()
                actual = observer.frozen_state
                assert actual is not None
                self.assertEqual(actual.policy, before.policy)
                observer.reset()
                observer.compute_qparams()
                self.assertNotEqual(observer.require_frozen().policy, before.policy)

    def test_per_channel_import_rejects_unsupported_scale_dtype_without_casting(self):
        obs = MinMaxObserver(
            name="weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )
        obs.observe_shape((2, 2))
        obs.load_qparams(torch.tensor([0.1, 0.2]), torch.tensor([2, 3]))
        before = obs.require_frozen()
        for dtype in (torch.float16, torch.float64):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(RuntimeError, "per-channel scale dtype"):
                    obs.load_qparams(
                        torch.tensor([0.1, 0.2], dtype=dtype), torch.tensor([2, 3])
                    )
                self.assertTrue(
                    torch.equal(
                        _tensor(obs.require_frozen().scale), _tensor(before.scale)
                    )
                )
                self.assertEqual(
                    _tensor(obs.require_frozen().scale).dtype, torch.float32
                )
        self.assertEqual(
            obs.fake_quant(torch.ones(2, 2, dtype=torch.bfloat16)).dtype, torch.bfloat16
        )

    def test_per_channel_refreeze_rejects_unsupported_dtype_after_module_move(self):
        for dtype in (torch.float16, torch.float64):
            with self.subTest(dtype=dtype):
                obs = MinMaxObserver(
                    name="weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
                )
                obs.collect(torch.tensor([[-1.0, 2.0], [-0.5, 0.75]]))
                obs.compute_qparams()
                obs.to(dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, "per-channel scale dtype"):
                    obs.derive_candidate()
                state = obs.frozen_state
                assert state is not None
                self.assertEqual(_tensor(state.scale).dtype, dtype)

    def test_cpu_per_channel_bfloat16_scales_store_but_reject_execution(self):
        obs = MinMaxObserver(
            name="weight", qscheme=QScheme.PER_CHANNEL_ASYMM, channel_axis=0
        )
        obs.observe_shape((2, 2))
        obs.load_qparams(
            torch.tensor([0.125, 0.25], dtype=torch.bfloat16), torch.tensor([2, 3])
        )
        obs.install_candidate(obs.derive_candidate())
        state = obs.require_frozen()
        self.assertEqual(_tensor(state.scale).dtype, torch.bfloat16)
        self.assertEqual(state.provenance, "imported")
        self.assertTrue(state.locked)
        with self.assertRaisesRegex(
            RuntimeError, "execution device cpu.*torch.bfloat16"
        ):
            obs.fake_quant(torch.ones(2, 2))
        self.assertTrue(obs.has_qparams)
        torch.testing.assert_close(
            _tensor(obs.require_frozen().scale), _tensor(state.scale), rtol=0, atol=0
        )

    def test_per_tensor_bfloat16_scale_retains_supported_math(self):
        obs = MinMaxObserver(name="activation")
        scale = torch.tensor(0.125, dtype=torch.bfloat16)
        zp = torch.tensor(3, dtype=torch.int)
        obs.load_qparams(scale, zp)
        x = torch.tensor([-0.125, 0.0, 0.25], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(obs.fake_quant(x), x))
        self.assertEqual(_tensor(obs.require_frozen().scale).dtype, torch.bfloat16)

    def test_affine_install_rejects_stateless_provenance_and_calibrated_lock(self):
        obs = MinMaxObserver(name="candidate")
        obs.collect(torch.tensor([-1.0, 2.0]))
        candidate = obs.derive_candidate()
        for invalid in (
            replace(candidate, provenance="stateless"),
            replace(candidate, locked=True),
        ):
            with self.subTest(provenance=invalid.provenance, locked=invalid.locked):
                with self.assertRaisesRegex(
                    RuntimeError, "invalid affine provenance or lock"
                ):
                    obs.install_candidate(invalid)
                self.assertTrue(obs.enabled)
                self.assertFalse(obs.has_qparams)
                self.assertIsNone(obs.frozen_state)

    def test_unlocked_import_remains_pending_after_state_restore(self):
        obs = MinMaxObserver(name="pending")
        obs.collect(torch.tensor([-1.0, 2.0]))
        obs.load_qparams(torch.tensor(123.0), torch.tensor(7), lock=False)
        self.assertTrue(obs.enabled)
        self.assertTrue(obs.has_qparams)
        self.assertIsNone(obs.frozen_state)
        restored = MinMaxObserver(name="pending")
        restored.load_state_dict(copy.deepcopy(obs.state_dict()))
        self.assertTrue(restored.enabled)
        self.assertIsNone(restored.frozen_state)
        scale, _ = restored.compute_qparams()
        self.assertNotEqual(scale.item(), 123.0)
        self.assertEqual(restored.require_frozen().provenance, "calibrated")
        self.assertFalse(restored.require_frozen().locked)
