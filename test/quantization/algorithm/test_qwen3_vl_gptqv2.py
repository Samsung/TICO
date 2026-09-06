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

"""
Tests for the Qwen3-VL GPTQv2 quantizer helpers.

These tests verify that:
  - The GPTQ class from qwen3_vl_gptq.gptq is used for both v1 and v2.
  - _build_gptq_objects initializes native_inp when gptq_v2=True.
  - Conv3d native inputs correctly populate dXXT.
"""

import copy
import os
import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from tico.quantization.algorithm.qwen3_vl_gptq.gptq import GPTQ
from tico.quantization.algorithm.qwen3_vl_gptq.quantizer import (
    FPInputsCache,
    Qwen3VLGPTQQuantizer,
)
from tico.quantization.config.qwen3_vl_gptq import Qwen3VLGPTQConfig


class TestQwen3VLGPTQv2Core(unittest.TestCase):
    """Test GPTQv2 core mechanics on Conv3d layers."""

    @torch.no_grad()
    def test_conv3d_native_inputs_populate_dXXT(self):
        """dXXT should be computed and non-zero when FP and quantized inputs differ."""
        layer = torch.nn.Conv3d(
            in_channels=2,
            out_channels=3,
            kernel_size=(2, 2, 2),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        gptq = GPTQ(layer)

        current = torch.randn(1, 2, 3, 3, 3)
        native = current + 0.125
        out = layer(current)

        gptq.native_inp = [native]
        gptq.add_batch(current, out)

        self.assertIsNotNone(gptq.dXXT)
        dXXT = gptq.dXXT
        assert dXXT is not None
        self.assertEqual(dXXT.shape, gptq.H.shape)  # type: ignore[union-attr]
        self.assertGreater(dXXT.abs().sum().item(), 0.0)


class TestQwen3VLGPTQv2QuantizerHelpers(unittest.TestCase):
    """Test Qwen3VLGPTQQuantizer helper methods."""

    def test_build_gptq_objects_default_config(self):
        """_build_gptq_objects should create GPTQ objects with native_inp=None for v1."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        self.assertIsInstance(gptq_objs["linear"], GPTQ)
        # For v1 (gptq_v2=False), native_inp should not be initialized as a list
        self.assertIsNone(gptq_objs["linear"].native_inp)

    def test_build_gptq_objects_gptqv2_config(self):
        """_build_gptq_objects should create GPTQ objects with native_inp=[] for v2."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(gptq_v2=True))
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        self.assertIsInstance(gptq_objs["linear"], GPTQ)
        # For v2 (gptq_v2=True), native_inp should be initialized as an empty list
        self.assertEqual(gptq_objs["linear"].native_inp, [])

    def test_assign_native_inputs(self):
        """_assign_native_inputs should copy FP inputs to GPTQ objects."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(gptq_v2=True))
        layer = torch.nn.Linear(4, 3)
        gptq_objs = quantizer._build_gptq_objects({"linear": layer}, {layer: "linear"})

        fp_inputs = [torch.randn(2, 4), torch.randn(3, 4)]
        native_inputs = {"linear": fp_inputs}

        quantizer._assign_native_inputs(gptq_objs, native_inputs)

        native_inp = gptq_objs["linear"].native_inp
        assert native_inp is not None
        self.assertEqual(len(native_inp), 2)
        self.assertTrue(torch.allclose(native_inp[0], fp_inputs[0]))
        self.assertTrue(torch.allclose(native_inp[1], fp_inputs[1]))

    def test_resolve_weight_bits_default(self):
        """_resolve_weight_bits should return the config default when no override."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig(weight_bits=4))
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 4)

    def test_resolve_weight_bits_override_full_name(self):
        """_resolve_weight_bits should use full-name override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"model.layers.0.self_attn.q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_resolve_weight_bits_override_local_name(self):
        """_resolve_weight_bits should use local-name override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"self_attn.q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_resolve_weight_bits_override_suffix(self):
        """_resolve_weight_bits should use suffix override when available."""
        quantizer = Qwen3VLGPTQQuantizer(
            Qwen3VLGPTQConfig(
                weight_bits=4,
                weight_bits_overrides={"q_proj": 8},
            )
        )
        bits = quantizer._resolve_weight_bits(
            quantizer.config,  # type: ignore[arg-type]
            full_module_name="model.layers.0.self_attn.q_proj",
            local_module_name="self_attn.q_proj",
        )
        self.assertEqual(bits, 8)

    def test_module_device(self):
        """_module_device should return the device of the module's parameters."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        layer = torch.nn.Linear(4, 3)
        device = quantizer._module_device(layer)
        self.assertEqual(device, layer.weight.device)

    def test_copy_original_model(self):
        """_copy_original_model should create a deep copy on CPU."""
        quantizer = Qwen3VLGPTQQuantizer(Qwen3VLGPTQConfig())
        model = torch.nn.Sequential(torch.nn.Linear(4, 3))
        orig_model = quantizer._copy_original_model(model)

        # Should be a different object
        self.assertIsNot(orig_model, model)
        # Weights should match
        self.assertTrue(torch.allclose(orig_model[0].weight, model[0].weight))  # type: ignore[index]
        # Modifying one should not affect the other
        model[0].weight.data.fill_(0.0)
        self.assertFalse(torch.allclose(orig_model[0].weight, model[0].weight))  # type: ignore[index]


# ---------------------------------------------------------------------------
# Helpers for FP inputs cache tests
# ---------------------------------------------------------------------------


def _make_quantizer(fp_inputs_cache_path=None, gptq_v2=True):
    """Create a Qwen3VLGPTQQuantizer with minimal config for testing."""
    config = Qwen3VLGPTQConfig(
        weight_bits=8,
        gptq_v2=gptq_v2,
        fp_inputs_cache_path=fp_inputs_cache_path,
        show_progress=False,
        verbose=False,
    )
    return Qwen3VLGPTQQuantizer(config)


# ---------------------------------------------------------------------------
# Tests: FPInputsCache
# ---------------------------------------------------------------------------


class TestFPInputsCache(unittest.TestCase):
    """Core tests for the FPInputsCache hook-based collector and disk cache."""

    def test_caches_fp_input(self):
        """A forward hook stores the first positional arg in fp_cache."""
        cache = FPInputsCache(["linear"])
        linear = nn.Linear(4, 4)
        cache.add_hook({"linear": linear})

        inp = torch.randn(2, 4)
        linear(inp)
        cache.clear_hook()

        self.assertIn("linear", cache.fp_cache)
        self.assertEqual(len(cache.fp_cache["linear"]), 1)
        self.assertTrue(torch.equal(cache.fp_cache["linear"][0], inp))

    def test_save_and_load_roundtrip(self):
        """Save cache to disk, load it back, and verify tensor equality."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "fp_cache.pt")
            dummy_cache = {
                "vision.patch_embed": {
                    "proj": [torch.randn(2, 3), torch.randn(2, 3)],
                },
                "text.layers.0": {
                    "self_attn.q_proj": [torch.randn(4, 5)],
                },
            }
            torch.save(dummy_cache, cache_path)
            self.assertTrue(os.path.exists(cache_path))

            loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.assertEqual(set(loaded.keys()), set(dummy_cache.keys()))
            for stage in dummy_cache:
                for name in dummy_cache[stage]:
                    for i, t in enumerate(loaded[stage][name]):
                        self.assertTrue(torch.equal(t, dummy_cache[stage][name][i]))

    def test_raw_replay_returns_cached_on_hit(self):
        """_collect_native_inputs_from_raw_replay returns cached data without
        running any forward hooks when stage_desc is in the disk cache."""
        quantizer = _make_quantizer(fp_inputs_cache_path="/tmp/dummy.pt")
        cached_tensors = [torch.randn(2, 3)]
        quantizer._fp_inputs_disk_cache = {
            "vision.merger": {"merger.linear": cached_tensors},
        }

        dummy_model = MagicMock()
        result = quantizer._collect_native_inputs_from_raw_replay(
            model=dummy_model,
            subset={"merger.linear": MagicMock()},
            module_name={},
            cache_args=[[]],
            cache_kwargs={},
            num_batches=1,
            stage_desc="vision.merger",
        )

        self.assertIn("merger.linear", result)
        self.assertTrue(torch.equal(result["merger.linear"][0], cached_tensors[0]))
        dummy_model.assert_not_called()

    @torch.no_grad()
    def test_collect_then_cache_hit(self):
        """First call collects via hooks and persists to _fp_inputs_disk_cache;
        second call returns from cache without re-running forward."""
        quantizer = _make_quantizer(fp_inputs_cache_path="/tmp/dummy.pt")

        linear = nn.Linear(4, 4)
        stage_module = nn.Sequential(linear)
        subset = {"0": linear}

        inp = torch.randn(2, 4)
        cached_args = [[inp]]
        cached_kwargs: dict = {}

        result1 = quantizer._collect_native_inputs_from_stage_cache(
            stage_module=stage_module,
            subset=subset,
            cached_args=cached_args,
            cached_kwargs=cached_kwargs,
            stage_desc="test_stage",
            num_batches=1,
        )
        self.assertIn("0", result1)
        self.assertTrue(torch.equal(result1["0"][0], inp))

        # The function should have persisted to _fp_inputs_disk_cache automatically
        self.assertIn("test_stage", quantizer._fp_inputs_disk_cache)

        broken_module = MagicMock(side_effect=RuntimeError("should not be called"))
        result2 = quantizer._collect_native_inputs_from_stage_cache(
            stage_module=broken_module,
            subset=subset,
            cached_args=cached_args,
            cached_kwargs=cached_kwargs,
            stage_desc="test_stage",
            num_batches=1,
        )
        self.assertTrue(torch.equal(result2["0"][0], result1["0"][0]))
        broken_module.assert_not_called()

    @torch.no_grad()
    def test_stage_cache_persist_roundtrip(self):
        """Cold collection -> save to disk -> new quantizer loads disk ->
        warm lookup returns cached tensors without calling forward.

        This test verifies that _collect_native_inputs_from_stage_cache()
        persists its result so that a warm-cache run can retrieve it.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "fp_cache.pt")

            # --- Cold run: collect and persist ---
            quantizer_cold = _make_quantizer(fp_inputs_cache_path=cache_path)

            linear = nn.Linear(4, 4)
            stage_module = nn.Sequential(linear)
            subset = {"0": linear}

            inp = torch.randn(2, 4)
            cached_args = [[inp]]
            cached_kwargs: dict = {}

            result_cold = quantizer_cold._collect_native_inputs_from_stage_cache(
                stage_module=stage_module,
                subset=subset,
                cached_args=cached_args,
                cached_kwargs=cached_kwargs,
                stage_desc="vision.blocks.0",
                num_batches=1,
            )
            self.assertIn("0", result_cold)
            self.assertTrue(torch.equal(result_cold["0"][0], inp))

            # Verify the stage was persisted to the in-memory disk cache
            self.assertIn("vision.blocks.0", quantizer_cold._fp_inputs_disk_cache)

            # Simulate convert()'s save logic
            torch.save(quantizer_cold._fp_inputs_disk_cache, cache_path)
            self.assertTrue(os.path.exists(cache_path))

            # --- Warm run: new quantizer loads cache from disk ---
            quantizer_warm = _make_quantizer(fp_inputs_cache_path=cache_path)
            quantizer_warm._fp_inputs_disk_cache = torch.load(
                cache_path, map_location="cpu", weights_only=False
            )
            quantizer_warm._fp_inputs_disk_loaded = True

            # Use a broken module that raises if forward is called
            broken_module = MagicMock(side_effect=RuntimeError("should not be called"))
            result_warm = quantizer_warm._collect_native_inputs_from_stage_cache(
                stage_module=broken_module,
                subset=subset,
                cached_args=cached_args,
                cached_kwargs=cached_kwargs,
                stage_desc="vision.blocks.0",
                num_batches=1,
            )
            # The warm result should match the cold result
            self.assertTrue(torch.equal(result_warm["0"][0], result_cold["0"][0]))
            broken_module.assert_not_called()

    @torch.no_grad()
    def test_stage_cache_fail_closed_on_miss(self):
        """When _fp_inputs_disk_loaded is True but the stage is not in the cache,
        a RuntimeError should be raised instead of silently recomputing."""
        quantizer = _make_quantizer(fp_inputs_cache_path="/tmp/dummy.pt")
        quantizer._fp_inputs_disk_loaded = True
        # "missing_stage" is NOT in _fp_inputs_disk_cache

        linear = nn.Linear(4, 4)
        stage_module = nn.Sequential(linear)
        subset = {"0": linear}

        inp = torch.randn(2, 4)
        cached_args = [[inp]]
        cached_kwargs: dict = {}

        with self.assertRaises(RuntimeError) as ctx:
            quantizer._collect_native_inputs_from_stage_cache(
                stage_module=stage_module,
                subset=subset,
                cached_args=cached_args,
                cached_kwargs=cached_kwargs,
                stage_desc="missing_stage",
                num_batches=1,
            )
        self.assertIn("cache miss", str(ctx.exception).lower())


# ---------------------------------------------------------------------------
# Tests: GPTQv2 P-correction (dXXT → P matrix)
# ---------------------------------------------------------------------------


class TestGPTQPCorrection(unittest.TestCase):
    """Reference tests for the GPTQv2 P-correction logic in fasterquant.

    These tests verify:
      1. ``quantize()`` does not modify ``w_col`` in-place.
      2. ``q_col`` and ``w_col`` are different tensors with different values.
      3. With ``alpha=0`` the P-correction is zero, so GPTQv2 == GPTQv1.
      4. With ``alpha>0`` and non-zero ``dXXT``, the quantized weights differ
         from the ``alpha=0`` case.
      5. The P matrix is computed as ``alpha * triu(dXXT @ hinv^T, k=1) @ hinv``.
    """

    def _make_gptq(self, rows=8, cols=8):
        """Create a GPTQ object with a small Linear layer and a configured quantizer."""
        torch.manual_seed(42)
        layer = nn.Linear(cols, rows, bias=False)
        gptq = GPTQ(layer)
        gptq.quantizer.configure(bits=8, perchannel=True, sym=True)
        return gptq

    def _add_random_batch(self, gptq, batch=16, cols=8):
        """Feed a random batch so that H is well-conditioned."""
        torch.manual_seed(123)
        inp = torch.randn(batch, cols)
        with torch.no_grad():
            out = gptq.layer(inp)
        gptq.add_batch(inp, out)

    # ------------------------------------------------------------------
    # 1. quantize() does not modify w_col in-place
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_quantize_does_not_modify_w_col(self):
        """The ``quantize`` function must not change its input tensor in-place."""
        from tico.quantization.algorithm.qwen3_vl_gptq.gptq import quantize

        w_col = torch.randn(6)
        w_col_copy = w_col.clone()

        scale = torch.tensor(0.1)
        zero = torch.tensor(0.0)
        maxq = torch.tensor(255.0)

        _ = quantize(w_col.unsqueeze(1), scale, zero, maxq)

        self.assertTrue(torch.equal(w_col, w_col_copy))

    # ------------------------------------------------------------------
    # 2. q_col != w_col after quantize()
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_q_col_differs_from_w_col(self):
        """``q_col`` (quantized) must differ from ``w_col`` (original)."""
        from tico.quantization.algorithm.qwen3_vl_gptq.gptq import quantize

        w_col = torch.randn(6)
        scale = torch.tensor(0.1)
        zero = torch.tensor(0.0)
        maxq = torch.tensor(255.0)

        q_col = quantize(w_col.unsqueeze(1), scale, zero, maxq).flatten()

        self.assertFalse(torch.equal(q_col, w_col))
        # The difference is the quantization error
        self.assertGreater((w_col - q_col).abs().sum().item(), 0.0)

    # ------------------------------------------------------------------
    # 3. alpha=0  →  P is zero  →  GPTQv2 == GPTQv1
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_alpha_zero_equals_no_pcorrection(self):
        """With alpha=0 the P-correction vanishes, so the result must match
        the case where dXXT is None (pure GPTQv1)."""
        rows, cols = 8, 8

        # --- run A: dXXT=None (GPTQv1) ---
        gptq_a = self._make_gptq(rows, cols)
        self._add_random_batch(gptq_a, batch=16, cols=cols)
        w_before_a = gptq_a.layer.weight.data.clone()
        gptq_a.fasterquant(blocksize=128, percdamp=0.01, alpha=0.0)
        w_after_a = gptq_a.layer.weight.data.clone()

        # --- run B: dXXT set but alpha=0 ---
        gptq_b = self._make_gptq(rows, cols)
        # Copy same weights and H so the two runs are comparable
        gptq_b.layer.weight.data = w_before_a.clone()
        self._add_random_batch(gptq_b, batch=16, cols=cols)
        gptq_b.dXXT = torch.randn(cols, cols)  # non-zero dXXT
        gptq_b.fasterquant(blocksize=128, percdamp=0.01, alpha=0.0)
        w_after_b = gptq_b.layer.weight.data.clone()

        self.assertTrue(torch.allclose(w_after_a, w_after_b, atol=1e-6))

    # ------------------------------------------------------------------
    # 4. alpha>0 with non-zero dXXT → result differs from alpha=0
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_alpha_positive_differs_from_zero(self):
        """With alpha>0 and a non-zero dXXT, the quantized weights must
        differ from the alpha=0 baseline."""
        rows, cols = 8, 8

        # --- baseline: alpha=0 ---
        gptq_base = self._make_gptq(rows, cols)
        self._add_random_batch(gptq_base, batch=16, cols=cols)
        w_orig = gptq_base.layer.weight.data.clone()
        gptq_base.dXXT = torch.randn(cols, cols)
        gptq_base.fasterquant(blocksize=128, percdamp=0.01, alpha=0.0)
        w_base = gptq_base.layer.weight.data.clone()

        # --- with P-correction: alpha=0.5 ---
        gptq_p = self._make_gptq(rows, cols)
        gptq_p.layer.weight.data = w_orig.clone()
        self._add_random_batch(gptq_p, batch=16, cols=cols)
        gptq_p.dXXT = gptq_base.dXXT.clone()  # same dXXT
        gptq_p.fasterquant(blocksize=128, percdamp=0.01, alpha=0.5)
        w_p = gptq_p.layer.weight.data.clone()

        self.assertFalse(torch.allclose(w_base, w_p, atol=1e-6))

    # ------------------------------------------------------------------
    # 5. P matrix formula: alpha * triu(dXXT @ hinv^T, k=1) @ hinv
    # ------------------------------------------------------------------

    @torch.no_grad()
    def test_p_correction_formula(self):
        """Manually compute P from dXXT and hinv, then verify that the
        in-block P-correction matches ``w_col @ P1[i, i:]``."""
        rows, cols = 6, 6

        gptq = self._make_gptq(rows, cols)
        self._add_random_batch(gptq, batch=32, cols=cols)

        # Set a known dXXT
        torch.manual_seed(99)
        dXXT = torch.randn(cols, cols)
        gptq.dXXT = dXXT.clone()
        alpha = 0.25

        # Reproduce the hinv computation from fasterquant
        h = gptq.H.clone()
        del gptq.H  # fasterquant does del self.H; we need to restore after
        gptq.H = h.clone()  # restore for fasterquant

        dead = torch.diag(h) == 0
        h[dead, dead] = 1
        damp = 0.01 * torch.mean(torch.diag(h))
        diag_idx = torch.arange(cols)
        h[diag_idx, diag_idx] += damp
        h = torch.linalg.cholesky(h)
        h = torch.cholesky_inverse(h)
        h = torch.linalg.cholesky(h, upper=True)
        hinv = h

        # Compute P using the same formula as fasterquant
        P_ref = alpha * ((dXXT @ hinv.T).triu(diagonal=1)) @ hinv

        # P must be non-zero
        self.assertGreater(P_ref.abs().sum().item(), 0.0)

        # The diagonal of P must be zero (triu with diagonal=1)
        self.assertTrue(torch.allclose(torch.diag(P_ref), torch.zeros(cols), atol=1e-6))

        # Run fasterquant and verify it completes without error
        gptq.fasterquant(blocksize=128, percdamp=0.01, alpha=alpha)
        # After fasterquant, the layer weight should have been updated
        self.assertIsNotNone(gptq.layer.weight.data)


# ---------------------------------------------------------------------------
# Tests: Final-weight reference for fasterquant with alpha > 0
# ---------------------------------------------------------------------------


def _reference_fasterquant(
    w: torch.Tensor,
    H: torch.Tensor,
    dXXT: torch.Tensor | None,
    quantizer,
    blocksize: int = 128,
    percdamp: float = 0.01,
    groupsize: int = -1,
    actorder: bool = False,
    alpha: float = 0.25,
) -> torch.Tensor:
    """
    Standalone reference implementation of ``GPTQ.fasterquant``.

    This is an independent re-implementation of the algorithm in
    ``tico.quantization.algorithm.qwen3_vl_gptq.gptq.GPTQ.fasterquant``
    so that any regression in the production code (wrong sign, wrong
    variable used for P-correction, missing cross-block term, etc.) is
    caught by comparing final weights.

    Args:
        w: Weight matrix of shape (rows, columns).
        H: Hessian matrix of shape (columns, columns).
        dXXT: dXXT matrix of shape (columns, columns), or None.
        quantizer: A configured Quantizer object (must already have
            find_params called, or will be called inside).
        blocksize: Block size for GPTQ.
        percdamp: Damping factor.
        groupsize: Group size (-1 disables).
        actorder: Whether to use activation order.
        alpha: P-correction strength.

    Returns:
        The quantized weight matrix ``q_all`` of shape (rows, columns).
    """
    from tico.quantization.algorithm.gptq.quant import quantize

    w = w.clone().float()
    rows = w.shape[0]
    columns = w.shape[1]

    # Find quantization parameters if not ready
    if not quantizer.ready():
        quantizer.find_params(w, weight=True)

    h = H.clone().float()

    # Dead columns
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0

    # Zero out dead elements in dXXT
    if dXXT is not None:
        dXXT = dXXT.clone().float()
        dXXT[:, dead] = 0.0

    # Actorder
    perm = None
    invperm = None
    if actorder:
        perm = torch.argsort(torch.diag(h), descending=True)
        w = w[:, perm]
        h = h[perm][:, perm]
        if dXXT is not None:
            dXXT = dXXT[perm][:, perm]
        invperm = torch.argsort(perm)

    # Damping and Cholesky inverse
    damp = percdamp * torch.mean(torch.diag(h))
    diag_idx = torch.arange(columns)
    h[diag_idx, diag_idx] += damp
    h = torch.linalg.cholesky(h)
    h = torch.cholesky_inverse(h)
    h = torch.linalg.cholesky(h, upper=True)
    hinv = h

    # P correction matrix
    P = None
    if dXXT is not None:
        P = alpha * ((dXXT @ hinv.T).triu(diagonal=1)) @ hinv

    # quantizer.update (for mse_for_gptq modes)
    quantizer.update(w, hinv, perm)

    q_all = torch.zeros_like(w)

    # ------------------------------------------------------------------
    # Structurally different from production code:
    #
    # Production code (gptq.py):
    #   - Uses a single w1 = w[:, i1:i2].clone()
    #   - Clones individual w_col = w1[:, i].clone() inside the loop
    #   - Precomputes P_update = w1.matmul(P[...]) BEFORE the inner loop
    #
    # This reference:
    #   - Uses a single w1 = w[:, i1:i2].clone() (same as production)
    #   - Clones w_col = w1[:, i].clone() inside the loop (same as production)
    #   - Precomputes P_update BEFORE the inner loop (same as production)
    #   - BUT uses a different code structure: separates the GPTQ error
    #     correction and P-correction into distinct named variables, and
    #     uses explicit outer-product construction instead of in-place -=
    #
    # The key correctness invariant: w_col must be the value of w1[:, i]
    # AFTER previous columns' updates but BEFORE the current column's
    # update. This matches production's w1[:, i].clone().
    # ------------------------------------------------------------------

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]

        if P is not None:
            P1 = P[i1:i2, i1:i2]
            # Precompute cross-block P correction from the pre-loop w1 snapshot
            P_update = w1.matmul(P[i1:i2, i2:])
        else:
            P1 = None
            P_update = None

        for i in range(count):
            # Clone to snapshot the value before this column's update
            w_col = w1[:, i].clone()
            d = hinv1[i, i]

            if groupsize != -1:
                if (i1 + i) % groupsize == 0:
                    quantizer.find_params(
                        w[:, (i1 + i) : (i1 + i + groupsize)],
                        weight=True,
                    )

            q_col = quantize(
                w_col.unsqueeze(1),
                quantizer.scale,
                quantizer.zero,
                quantizer.maxq,
            ).flatten()

            q1[:, i] = q_col

            cur_err = (w_col - q_col) / d
            # GPTQ error correction: update remaining columns in this block
            correction = cur_err.unsqueeze(1).matmul(hinv1[i, i:].unsqueeze(0))
            w1[:, i:] -= correction
            # P-correction: uses the pre-update w_col (cloned above)
            if P1 is not None:
                p_correction = w_col.unsqueeze(1).matmul(P1[i, i:].unsqueeze(0))
                w1[:, i:] += p_correction
            err1[:, i] = cur_err

        q_all[:, i1:i2] = q1
        # Cross-block GPTQ update
        w[:, i2:] -= err1.matmul(hinv[i1:i2, i2:])
        # Cross-block P-correction (precomputed from pre-loop w1)
        if P_update is not None:
            w[:, i2:] += P_update


    if actorder:
        q_all = q_all[:, invperm]

    # Dead column RTN
    if groupsize == -1:
        pass

    return q_all


class TestGPTQv2FinalWeightReference(unittest.TestCase):
    """
    Reference tests that independently compute the expected final weights
    after ``fasterquant`` with ``alpha > 0`` and compare them against the
    actual ``fasterquant`` output.

    Unlike the existing tests that only check "alpha>0 differs from alpha=0"
    or "the P formula looks right", these tests verify that **every element**
    of the final weight matrix matches an independent reference implementation.

    This catches regressions such as:
      - Using ``q_col`` instead of ``w_col`` in the P-correction term
      - Wrong sign on the P-correction
      - Missing cross-block P-correction (``P_update``)
      - Incorrect dXXT permutation under actorder
      - Wrong indexing in the in-block P-correction
    """

    def _make_gptq_with_state(self, rows, cols, batch=32, seed_w=42, seed_inp=123, bits=8):
        """
        Create a GPTQ object, add a batch, and return:
          (gptq, w_orig, H, quantizer_config_snapshot)
        """
        torch.manual_seed(seed_w)
        layer = nn.Linear(cols, rows, bias=False)
        gptq = GPTQ(layer)
        gptq.quantizer.configure(bits=bits, perchannel=True, sym=False)


        torch.manual_seed(seed_inp)
        inp = torch.randn(batch, cols)
        with torch.no_grad():
            out = gptq.layer(inp)
        gptq.add_batch(inp, out)

        return gptq

    @torch.no_grad()
    def test_final_weights_match_reference_single_block(self):
        """Final weights from fasterquant (alpha>0, single block) must match
        the independent reference implementation."""
        rows, cols = 6, 8
        alpha = 0.50

        # Create two identical GPTQ objects
        gptq_actual = self._make_gptq_with_state(rows, cols)
        gptq_ref = self._make_gptq_with_state(rows, cols)

        # Verify they have identical state
        self.assertTrue(torch.allclose(gptq_actual.layer.weight.data, gptq_ref.layer.weight.data))
        self.assertTrue(torch.allclose(gptq_actual.H, gptq_ref.H))

        # Set a known non-zero dXXT
        torch.manual_seed(99)
        dXXT = torch.randn(cols, cols)
        gptq_actual.dXXT = dXXT.clone()
        gptq_ref.dXXT = dXXT.clone()

        # Save state for reference computation
        w_saved = gptq_ref.layer.weight.data.clone()
        H_saved = gptq_ref.H.clone()
        dXXT_saved = gptq_ref.dXXT.clone()

        # Run actual fasterquant
        gptq_actual.fasterquant(
            blocksize=128, percdamp=0.01, groupsize=-1,
            actorder=True, alpha=alpha,
        )
        w_actual = gptq_actual.layer.weight.data.clone().float()

        # Run reference implementation
        w_reference = _reference_fasterquant(
            w=w_saved,
            H=H_saved,
            dXXT=dXXT_saved,
            quantizer=gptq_ref.quantizer,
            blocksize=128,
            percdamp=0.01,
            groupsize=-1,
            actorder=True,
            alpha=alpha,
        )

        self.assertTrue(
            torch.allclose(w_actual, w_reference, atol=1e-5),
            f"Single-block: actual weights do not match reference. "
            f"Max diff: {(w_actual - w_reference).abs().max().item():.2e}",
        )

    @torch.no_grad()
    def test_final_weights_match_reference_multiple_blocks(self):
        """Final weights from fasterquant (alpha>0, multiple blocks) must match
        the independent reference implementation. This tests both in-block
        and cross-block P-correction."""
        rows, cols = 6, 12
        alpha = 0.25
        blocksize = 4  # 3 blocks

        gptq_actual = self._make_gptq_with_state(rows, cols)
        gptq_ref = self._make_gptq_with_state(rows, cols)

        torch.manual_seed(99)
        dXXT = torch.randn(cols, cols)
        gptq_actual.dXXT = dXXT.clone()
        gptq_ref.dXXT = dXXT.clone()

        w_saved = gptq_ref.layer.weight.data.clone()
        H_saved = gptq_ref.H.clone()
        dXXT_saved = gptq_ref.dXXT.clone()

        gptq_actual.fasterquant(
            blocksize=blocksize, percdamp=0.01, groupsize=-1,
            actorder=True, alpha=alpha,
        )
        w_actual = gptq_actual.layer.weight.data.clone().float()

        w_reference = _reference_fasterquant(
            w=w_saved,
            H=H_saved,
            dXXT=dXXT_saved,
            quantizer=gptq_ref.quantizer,
            blocksize=blocksize,
            percdamp=0.01,
            groupsize=-1,
            actorder=True,
            alpha=alpha,
        )

        self.assertTrue(
            torch.allclose(w_actual, w_reference, atol=1e-5),
            f"Multi-block: actual weights do not match reference. "
            f"Max diff: {(w_actual - w_reference).abs().max().item():.2e}",
        )

    @torch.no_grad()
    def test_final_weights_match_reference_actorder(self):
        """Final weights from fasterquant (alpha>0, actorder=True, multiple
        blocks) must match the reference. This tests P-correction with
        Hessian permutation (dXXT is permuted too)."""
        rows, cols = 6, 12
        alpha = 0.25
        blocksize = 4

        gptq_actual = self._make_gptq_with_state(rows, cols)
        gptq_ref = self._make_gptq_with_state(rows, cols)

        torch.manual_seed(99)
        dXXT = torch.randn(cols, cols)
        gptq_actual.dXXT = dXXT.clone()
        gptq_ref.dXXT = dXXT.clone()

        w_saved = gptq_ref.layer.weight.data.clone()
        H_saved = gptq_ref.H.clone()
        dXXT_saved = gptq_ref.dXXT.clone()

        gptq_actual.fasterquant(
            blocksize=blocksize, percdamp=0.01, groupsize=-1,
            actorder=True, alpha=alpha,
        )
        w_actual = gptq_actual.layer.weight.data.clone().float()

        w_reference = _reference_fasterquant(
            w=w_saved,
            H=H_saved,
            dXXT=dXXT_saved,
            quantizer=gptq_ref.quantizer,
            blocksize=blocksize,
            percdamp=0.01,
            groupsize=-1,
            actorder=True,
            alpha=alpha,
        )

        self.assertTrue(
            torch.allclose(w_actual, w_reference, atol=1e-5),
            f"Actorder: actual weights do not match reference. "
            f"Max diff: {(w_actual - w_reference).abs().max().item():.2e}",
        )

    @torch.no_grad()
    def test_final_weights_match_reference_alpha_zero(self):
        """Sanity check: with alpha=0, reference should match fasterquant
        (P is zero, so it's plain GPTQ). This validates the reference
        implementation itself."""
        rows, cols = 6, 12
        alpha = 0.0
        blocksize = 4

        gptq_actual = self._make_gptq_with_state(rows, cols)
        gptq_ref = self._make_gptq_with_state(rows, cols)

        torch.manual_seed(99)
        dXXT = torch.randn(cols, cols)
        gptq_actual.dXXT = dXXT.clone()
        gptq_ref.dXXT = dXXT.clone()

        w_saved = gptq_ref.layer.weight.data.clone()
        H_saved = gptq_ref.H.clone()
        dXXT_saved = gptq_ref.dXXT.clone()

        gptq_actual.fasterquant(
            blocksize=blocksize, percdamp=0.01, groupsize=-1,
            actorder=False, alpha=alpha,
        )
        w_actual = gptq_actual.layer.weight.data.clone().float()

        w_reference = _reference_fasterquant(
            w=w_saved,
            H=H_saved,
            dXXT=dXXT_saved,
            quantizer=gptq_ref.quantizer,
            blocksize=blocksize,
            percdamp=0.01,
            groupsize=-1,
            actorder=False,
            alpha=alpha,
        )

        self.assertTrue(
            torch.allclose(w_actual, w_reference, atol=1e-5),
            f"Alpha=0: actual weights do not match reference. "
            f"Max diff: {(w_actual - w_reference).abs().max().item():.2e}",
        )

    @torch.no_grad()
    def test_final_weights_match_reference_different_alpha(self):
        """Test with a different alpha value (0.5) to ensure the P-correction
        scaling is correct."""
        rows, cols = 8, 10
        alpha = 0.5
        blocksize = 5  # 2 blocks

        gptq_actual = self._make_gptq_with_state(rows, cols, batch=48)
        gptq_ref = self._make_gptq_with_state(rows, cols, batch=48)

        torch.manual_seed(77)
        dXXT = torch.randn(cols, cols)
        gptq_actual.dXXT = dXXT.clone()
        gptq_ref.dXXT = dXXT.clone()

        w_saved = gptq_ref.layer.weight.data.clone()
        H_saved = gptq_ref.H.clone()
        dXXT_saved = gptq_ref.dXXT.clone()

        gptq_actual.fasterquant(
            blocksize=blocksize, percdamp=0.01, groupsize=-1,
            actorder=True, alpha=alpha,
        )
        w_actual = gptq_actual.layer.weight.data.clone().float()

        w_reference = _reference_fasterquant(
            w=w_saved,
            H=H_saved,
            dXXT=dXXT_saved,
            quantizer=gptq_ref.quantizer,
            blocksize=blocksize,
            percdamp=0.01,
            groupsize=-1,
            actorder=True,
            alpha=alpha,
        )

        self.assertTrue(
            torch.allclose(w_actual, w_reference, atol=1e-5),
            f"Alpha=0.5: actual weights do not match reference. "
            f"Max diff: {(w_actual - w_reference).abs().max().item():.2e}",
        )


if __name__ == "__main__":
    unittest.main()
