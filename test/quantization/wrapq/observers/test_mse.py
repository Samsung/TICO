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

"""Unit tests for the MSE grid-search observer."""

import copy
import unittest
from typing import Optional

import torch

from tico.quantization.wrapq.dtypes import DType, INT8, UINT8
from tico.quantization.wrapq.observers.base import ObserverNotReadyError
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.mse import MSEObserver
from tico.quantization.wrapq.qscheme import QScheme


def _tensor(value: Optional[torch.Tensor]) -> torch.Tensor:
    assert value is not None
    return value


def _mean_squared_fake_quant_error(observer, values: torch.Tensor) -> float:
    """Return the observer's fake-quantization MSE on ``values``."""
    quantized = observer.fake_quant(values.clone().float())
    return float((values.float() - quantized).square().mean())


class MSEObserverTest(unittest.TestCase):
    def setUp(self) -> None:
        """Make every deterministic search reproducible."""
        torch.manual_seed(0)

    def _outlier_tensor(self, count: int = 1000) -> torch.Tensor:
        """Return seeded gaussian data with one dominant outlier."""
        values = torch.randn(count)
        values[0] = 40.0
        return values

    def _uint4(self, **kwargs) -> MSEObserver:
        """Return a uint4 per-tensor asymmetric MSE observer."""
        return MSEObserver(name="activation", dtype=DType.uint(4), **kwargs)

    def _minmax(self, dtype: DType = DType.uint(4), **kwargs) -> MinMaxObserver:
        """Return a matching MinMax observer for baseline comparisons."""
        return MinMaxObserver(name="minmax", dtype=dtype, **kwargs)

    def test_constructor_rejects_invalid_search_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_grid"):
            MSEObserver(name="bad", num_grid=1)
        with self.assertRaisesRegex(ValueError, "num_zp_grid"):
            MSEObserver(name="bad", num_zp_grid=2)
        with self.assertRaisesRegex(ValueError, "max_alternating_iters"):
            MSEObserver(name="bad", max_alternating_iters=-1)
        with self.assertRaisesRegex(ValueError, "search_chunk_size"):
            MSEObserver(name="bad", search_chunk_size=0)

    def test_missing_stats_are_not_ready(self) -> None:
        observer = self._uint4()
        with self.assertRaises(ObserverNotReadyError):
            observer.derive_candidate()
        observer.collect(torch.empty(0))
        with self.assertRaises(ObserverNotReadyError):
            observer.derive_candidate()

    def test_searched_grid_beats_minmax_for_low_bits(self) -> None:
        values = self._outlier_tensor()

        minmax = self._minmax(dtype=DType.uint(4))
        minmax.collect(values)
        minmax.compute_qparams()

        observer = self._uint4()
        observer.collect(values)
        observer.compute_qparams()

        minmax_error = _mean_squared_fake_quant_error(minmax, values)
        mse_error = _mean_squared_fake_quant_error(observer, values)
        self.assertLess(mse_error, minmax_error)
        # The alpha = 1 candidate equals the MinMax grid, so the searched
        # scale can never exceed the MinMax scale on a single batch.
        self.assertLessEqual(
            float(observer._cached_scale), float(minmax._cached_scale) + 1e-12
        )

    def test_symmetric_scheme_keeps_zero_point(self) -> None:
        values = torch.randn(1000)
        values[5] = -30.0

        observer = MSEObserver(
            name="activation", dtype=DType.int(4), qscheme=QScheme.PER_TENSOR_SYMM
        )
        observer.collect(values)
        scale, zero_point = observer.compute_qparams()

        self.assertEqual(int(zero_point), 0)
        self.assertGreater(float(scale), 0.0)

        minmax = self._minmax(dtype=DType.int(4), qscheme=QScheme.PER_TENSOR_SYMM)
        minmax.collect(values)
        minmax.compute_qparams()
        self.assertLess(
            _mean_squared_fake_quant_error(observer, values),
            _mean_squared_fake_quant_error(minmax, values),
        )
        self.assertLessEqual(float(scale), float(minmax._cached_scale) + 1e-12)

    def test_alternating_search_never_worsens_the_plain_candidate(self) -> None:
        values = self._outlier_tensor()

        plain = self._uint4(alternating_search=False)
        plain.collect(values)
        plain.compute_qparams()

        alternating = self._uint4(alternating_search=True, max_alternating_iters=2)
        alternating.collect(values)
        alternating.compute_qparams()

        # Each refinement round keeps the previous candidate in its grid, so
        # the alternating result cannot be worse than the plain scale search.
        self.assertLessEqual(
            _mean_squared_fake_quant_error(alternating, values),
            _mean_squared_fake_quant_error(plain, values) + 1e-9,
        )

    def test_multi_batches_merge_by_interval_union(self) -> None:
        first = torch.randn(64)
        second = torch.randn(64) * 3.0

        merged = self._uint4()
        merged.collect(first)
        merged.collect(second)

        first_only = self._uint4()
        first_only.collect(first)

        self.assertLessEqual(float(merged.clip_min_val), float(first_only.clip_min_val))
        self.assertGreaterEqual(
            float(merged.clip_max_val), float(first_only.clip_max_val)
        )

        # An interval union commutes, so collection order cannot change grid.
        reversed_order = self._uint4()
        reversed_order.collect(second)
        reversed_order.collect(first)
        merged.compute_qparams()
        reversed_order.compute_qparams()
        self.assertTrue(torch.equal(merged._cached_scale, reversed_order._cached_scale))
        self.assertTrue(torch.equal(merged._cached_zp, reversed_order._cached_zp))

    def test_per_channel_search_matches_minmax_contract(self) -> None:
        values = torch.randn(16, 8)
        values[:, 3] *= 8.0

        minmax = self._minmax(
            dtype=DType.uint(4),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=-1,
        )
        minmax.collect(values)
        minmax.compute_qparams()
        minmax._cached_scale = minmax._cached_scale.float()
        minmax._cached_zp = minmax._cached_zp.to(torch.int)

        observer = MSEObserver(
            name="activation",
            dtype=DType.uint(4),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=-1,
        )
        observer.collect(values)
        scale, zero_point = observer.compute_qparams()

        self.assertEqual(scale.shape, (8,))
        self.assertEqual(scale.dtype, torch.float32)
        self.assertEqual(zero_point.shape, (8,))
        self.assertTrue(bool(((zero_point >= 0) & (zero_point <= 15)).all()))
        self.assertTrue(
            bool((observer._cached_scale <= minmax._cached_scale + 1e-9).all())
        )
        self.assertLessEqual(
            _mean_squared_fake_quant_error(observer, values),
            _mean_squared_fake_quant_error(minmax, values) + 1e-9,
        )

    def test_negative_and_positive_channel_axis_agree(self) -> None:
        values = torch.randn(4, 8)

        negative = MSEObserver(
            name="negative",
            dtype=DType.uint(4),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=-1,
        )
        positive = MSEObserver(
            name="positive",
            dtype=DType.uint(4),
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=1,
        )
        negative.collect(values)
        positive.collect(values)

        self.assertTrue(torch.equal(negative.clip_min_val, positive.clip_min_val))
        self.assertTrue(torch.equal(negative.clip_max_val, positive.clip_max_val))

    def test_chunk_size_does_not_change_the_result(self) -> None:
        values = torch.randn(1000)

        large = self._uint4(search_chunk_size=1 << 20)
        large.collect(values)
        small = self._uint4(search_chunk_size=64)
        small.collect(values)

        self.assertTrue(torch.equal(large.clip_min_val, small.clip_min_val))
        self.assertTrue(torch.equal(large.clip_max_val, small.clip_max_val))

    def test_constant_zero_batch_yields_valid_degenerate_grid(self) -> None:
        observer = self._uint4()
        observer.collect(torch.zeros(5))
        scale, _ = observer.compute_qparams()
        self.assertTrue(bool(torch.isfinite(scale)))
        self.assertGreater(float(scale), 0.0)
        self.assertTrue(bool((observer.fake_quant(torch.zeros(5)) == 0).all()))

        symmetric = MSEObserver(
            name="symmetric", dtype=INT8, qscheme=QScheme.PER_TENSOR_SYMM
        )
        symmetric.collect(torch.zeros(5))
        scale, zero_point = symmetric.compute_qparams()
        self.assertAlmostEqual(float(scale), 1e-8, delta=1e-9)
        self.assertEqual(int(zero_point), 0)

    def test_nonfinite_statistics_are_invalid_not_missing(self) -> None:
        observer = self._uint4()
        observer.collect(torch.tensor([float("nan"), 1.0]))
        with self.assertRaises(RuntimeError) as caught:
            observer.derive_candidate()
        self.assertNotIsInstance(caught.exception, ObserverNotReadyError)
        self.assertIsNone(observer.frozen_state)
        self.assertTrue(observer.enabled)

    def test_state_dict_resumes_an_interrupted_calibration(self) -> None:
        first = torch.randn(64)
        second = torch.randn(64) * 3.0

        source = self._uint4()
        source.collect(first)

        resumed = self._uint4()
        resumed.load_state_dict(copy.deepcopy(source.state_dict()))
        resumed.collect(second)
        resumed.compute_qparams()

        direct = self._uint4()
        direct.collect(first)
        direct.collect(second)
        direct.compute_qparams()

        self.assertTrue(torch.equal(resumed._cached_scale, direct._cached_scale))
        self.assertTrue(torch.equal(resumed._cached_zp, direct._cached_zp))

    def test_frozen_grid_roundtrips_exactly(self) -> None:
        values = self._outlier_tensor()

        observer = self._uint4()
        observer.collect(values)
        observer.compute_qparams()
        observer.disable_fake_quant()

        receiver = self._uint4()
        receiver.load_state_dict(copy.deepcopy(observer.state_dict()))

        self.assertTrue(receiver._frozen)
        self.assertTrue(torch.equal(receiver._cached_scale, observer._cached_scale))
        self.assertTrue(torch.equal(receiver._cached_zp, observer._cached_zp))
        receiver.compute_qparams()
        self.assertTrue(
            torch.equal(
                _tensor(receiver.require_frozen().scale),
                _tensor(observer.require_frozen().scale),
            )
        )

    def test_search_policy_mismatch_rejects_state_load(self) -> None:
        source = self._uint4()
        source.collect(torch.randn(32))
        source.compute_qparams()

        mismatched_grid = MSEObserver(
            name="activation", dtype=DType.uint(4), num_grid=40
        )
        with self.assertRaisesRegex(RuntimeError, "incompatible frozen policy"):
            mismatched_grid.load_state_dict(copy.deepcopy(source.state_dict()))

        mismatched_search = MSEObserver(
            name="activation", dtype=DType.uint(4), alternating_search=False
        )
        with self.assertRaisesRegex(RuntimeError, "incompatible frozen policy"):
            mismatched_search.load_state_dict(copy.deepcopy(source.state_dict()))

    def test_calibration_is_deterministic(self) -> None:
        values = self._outlier_tensor()

        first = self._uint4()
        first.collect(values)
        first_scale, first_zp = first.compute_qparams()

        second = self._uint4()
        second.collect(values)
        second_scale, second_zp = second.compute_qparams()

        self.assertTrue(torch.equal(first_scale, second_scale))
        self.assertTrue(torch.equal(first_zp, second_zp))

    def test_locked_import_suppresses_mse_collection(self) -> None:
        observer = self._uint4()
        observer.load_qparams(torch.tensor(0.25), torch.tensor(6))
        observer.collect(self._outlier_tensor())

        candidate = observer.derive_candidate()
        self.assertEqual(candidate.provenance, "imported")
        self.assertAlmostEqual(float(_tensor(candidate.scale)), 0.25)
        self.assertTrue(bool(torch.isposinf(observer.clip_min_val)))

    def test_bfloat16_input_produces_float32_grid(self) -> None:
        observer = MSEObserver(
            name="activation",
            dtype=UINT8,
            qscheme=QScheme.PER_CHANNEL_ASYMM,
            channel_axis=-1,
        )
        observer.collect(torch.randn(8, 4).to(torch.bfloat16))
        scale, zero_point = observer.compute_qparams()

        self.assertEqual(scale.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(scale).all()))
        self.assertTrue(bool((scale > 0).all()))
        self.assertTrue(bool(torch.isfinite(zero_point).all()))

    def test_extra_repr_reports_search_settings(self) -> None:
        observer = MSEObserver(
            name="activation", num_grid=16, num_zp_grid=9, max_alternating_iters=1
        )
        summary = observer.extra_repr()
        self.assertIn("num_grid=16", summary)
        self.assertIn("num_zp_grid=9", summary)
        self.assertIn("max_alternating_iters=1", summary)

    def test_reset_clears_merged_coverage(self) -> None:
        observer = self._uint4()
        observer.collect(self._outlier_tensor())
        observer.reset()

        self.assertTrue(bool(torch.isposinf(observer.clip_min_val)))
        self.assertTrue(bool(torch.isneginf(observer.clip_max_val)))
        with self.assertRaises(ObserverNotReadyError):
            observer.derive_candidate()


if __name__ == "__main__":
    unittest.main()
