# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Behavioral contracts for the common model freeze transaction."""

import copy
import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from tico.quantization import convert, prepare
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.wrapq.lifecycle import enable_calibration, freeze_qparams
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.quantizer import PTQQuantizer
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn


class Pair(QuantModuleBase):
    def __init__(self):
        super().__init__()
        self.first = MinMaxObserver(name="first")
        self.last = PercentileObserver(name="last")
        self.preparations = 0

    def _all_observers(self):
        return self.first, self.last

    def _prepare_calibration(self):
        self.preparations += 1

    def forward(self, x):
        return self._fq(self._fq(x, self.first), self.last)


class Conditional(Pair):
    def local_observer_groups(self):
        return {"required": (self.first,), "cache": (self.last,)}

    def forward(self, x, use_cache=False):
        if use_cache:
            self.require_capability("cache")
            x = self._fq(x, self.last)
        return self._fq(x, self.first)


class TestFrozenLifecycle(unittest.TestCase):
    def test_once_only_arbitrary_containers_and_aliases(self):
        leaf = Pair()
        root = nn.Module()
        root.plain = nn.Module()
        root.plain.leaf = leaf
        root.tied = leaf
        leaf.alias = leaf.first
        with patch.object(leaf.first, "reset", wraps=leaf.first.reset) as reset:
            enable_calibration(root)
            self.assertEqual(reset.call_count, 1)
        self.assertEqual(leaf.preparations, 1)
        leaf(torch.tensor([-1.0, 2.0]))
        with patch.object(
            leaf.first, "derive_candidate", wraps=leaf.first.derive_candidate
        ) as derive:
            PTQQuantizer(PTQConfig()).convert(root)
            self.assertEqual(derive.call_count, 1)
        before = leaf.first.require_frozen()
        freeze_qparams(root)
        after = leaf.first.require_frozen()
        assert before.scale is not None and after.scale is not None
        self.assertTrue(torch.equal(before.scale, after.scale))

    def test_planning_does_not_retain_the_model_until_cyclic_gc(self):
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            model = Pair()
            model.enable_calibration()
            model(torch.tensor([-1.0, 2.0]))
            model.freeze_qparams()
            reference = weakref.ref(model)
            del model
            self.assertIsNone(reference())
        finally:
            if was_enabled:
                gc.enable()

    def test_declared_unregistered_observer_rejects(self):
        model = Pair()
        observer = model._modules.pop("first")
        object.__setattr__(model, "first", observer)
        with self.assertRaisesRegex(RuntimeError, "registered local observer"):
            model.enable_calibration()

    def test_required_alias_is_not_hidden_by_an_optional_reference(self):
        class Holder(QuantModuleBase):
            def __init__(self, branch):
                super().__init__()
                self.branch = branch

            def _all_observers(self):
                return ()

            def _conditional_quant_subtrees(self):
                return {"branch": (self.branch,)}

        leaf = Pair()
        root = nn.ModuleList([Holder(leaf), leaf])
        enable_calibration(root)
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            freeze_qparams(root)
        self.assertIs(leaf._mode, Mode.CALIB)

    def test_tied_optional_module_is_ready_through_either_capability(self):
        class Holder(QuantModuleBase):
            def __init__(self, branch):
                super().__init__()
                self.branch = branch
                self.trigger = MinMaxObserver(name="trigger")

            def _all_observers(self):
                return (self.trigger,)

            def local_observer_groups(self):
                return {"required": (), "branch": (self.trigger,)}

            def _conditional_quant_subtrees(self):
                return {"branch": (self.branch,)}

        leaf = Pair()
        first, second = Holder(leaf), Holder(leaf)
        root = nn.ModuleList([first, second])
        enable_calibration(root)
        x = torch.tensor([-1.0, 2.0])
        leaf(x)
        first.trigger.collect(x)
        freeze_qparams(root)
        self.assertTrue(first.has_capability("branch"))
        self.assertFalse(second.has_capability("branch"))
        self.assertIsNotNone(leaf.first.frozen_state)
        self.assertFalse(leaf._conditional_unavailable)
        root.load_state_dict(copy.deepcopy(root.state_dict()))

    def test_conflicting_observer_owners_reject(self):
        first, second = Pair(), Pair()
        second.first = first.first
        root = nn.ModuleList([first, second])
        with self.assertRaisesRegex(RuntimeError, "conflicting wrapper owners"):
            enable_calibration(root)

    def test_partial_optional_subtree_has_no_partial_frozen_grids(self):
        class OptionalTree(Pair):
            def __init__(self):
                super().__init__()
                self.branch = Pair()

            def _conditional_quant_subtrees(self):
                return {"branch": (self.branch,)}

        model = OptionalTree()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model(x)
        model.branch.first.collect(x)
        model.freeze_qparams()
        self.assertFalse(model.has_capability("branch"))
        self.assertIsNone(model.branch.first.frozen_state)
        self.assertIsNone(model.branch.last.frozen_state)
        restored = OptionalTree()
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertFalse(restored.has_capability("branch"))

    def test_real_percentile_failure_preserves_all_state_and_retry_handle(self):
        for inplace in (True, False):
            with self.subTest(inplace=inplace):
                model = Pair()
                enable_calibration(model)
                setattr(model, "tico_quantizer", PTQQuantizer(PTQConfig()))
                model.first.collect(torch.tensor([-1.0, 2.0]))
                before = copy.deepcopy(model.state_dict())
                with self.assertRaisesRegex(RuntimeError, "calibration"):
                    convert(model, inplace=inplace)
                self.assertTrue(hasattr(model, "tico_quantizer"))
                self.assertIs(model._mode, Mode.CALIB)
                self.assertTrue(model.first.enabled)
                self.assertFalse(model.first.has_qparams)
                self.assertEqual(before.keys(), model.state_dict().keys())
                for key, value in before.items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(
                            torch.equal(value, model.state_dict()[key]), key
                        )
                model(torch.tensor([-1.0, 2.0]))
                frozen = convert(model, inplace=inplace)
                self.assertIs(frozen._mode, Mode.QUANT)
                self.assertFalse(hasattr(frozen, "tico_quantizer"))
                self.assertEqual(hasattr(model, "tico_quantizer"), not inplace)
                self.assertIs(model._mode, Mode.QUANT if inplace else Mode.CALIB)

    def test_install_failure_rolls_back_small_state(self):
        model = Pair()
        model.enable_calibration()
        model(torch.tensor([-1.0, 2.0]))
        original = model.last.install_candidate

        def fail(state):
            original(state)
            raise RuntimeError("install failure")

        with patch.object(model.last, "install_candidate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "install failure"):
                model.freeze_qparams()
        self.assertIs(model._mode, Mode.CALIB)
        self.assertTrue(model.first.enabled)
        self.assertTrue(model.last.enabled)
        self.assertIsNone(model.first.frozen_state)
        self.assertIsNone(model.last.frozen_state)
        model.freeze_qparams()
        self.assertIs(model._mode, Mode.QUANT)

    def test_install_failure_restores_existing_clipping_provenance(self):
        model = Pair()
        model.enable_calibration()
        model(torch.tensor([-1.0, 2.0]))
        model.freeze_qparams()
        before = model.last.require_frozen()
        assert before.clipping_bounds is not None

        def fail(state):
            del state
            bounds = model.last._frozen_clipping_bounds
            assert bounds is not None
            bounds[0].fill_(333.0)
            raise RuntimeError("in-place install failure")

        with patch.object(model.last, "install_candidate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "in-place install failure"):
                model.freeze_qparams()
        after = model.last.require_frozen()
        assert after.clipping_bounds is not None
        for actual, expected in zip(after.clipping_bounds, before.clipping_bounds):
            self.assertTrue(torch.equal(actual, expected))
        self.assertIs(model._mode, Mode.QUANT)
        self.assertFalse(model.last.enabled)

    def test_parent_semantic_failure_does_not_commit_children(self):
        class Semantic(Pair):
            reject = True

            def _validate_freeze(self, plan, available):
                if self.reject:
                    raise RuntimeError("invalid semantic binding")

        model = Semantic()
        model.child = Pair()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model(x)
        model.child(x)
        with self.assertRaisesRegex(RuntimeError, "invalid semantic binding"):
            model.freeze_qparams()
        for module in (model, model.child):
            self.assertIs(module._mode, Mode.CALIB)
            self.assertIsNone(module.first.frozen_state)
            self.assertIsNone(module.last.frozen_state)
        model.reject = False
        model.freeze_qparams()
        self.assertIs(model.child._mode, Mode.QUANT)

    def test_invalid_optional_statistics_are_not_silently_unavailable(self):
        model = Conditional()
        model.enable_calibration()
        model.first.collect(torch.tensor([-1.0, 2.0]))
        # Percentile explicitly filters nonfinite samples; use a min/max site to
        # distinguish invalid observed data from the untouched-range sentinel.
        setattr(model, "last", MinMaxObserver(name="last"))
        model.last.collect(torch.tensor([float("nan")]))
        with self.assertRaisesRegex(RuntimeError, "invalid calibration"):
            model.freeze_qparams()
        self.assertIs(model._mode, Mode.CALIB)

    def test_plain_root_postload_rejects_missing_frozen_site_nonstrict(self):
        root = nn.Module()
        root.child = Pair()
        enable_calibration(root)
        root.child(torch.tensor([-1.0, 2.0]))
        freeze_qparams(root)
        state = copy.deepcopy(root.state_dict())
        del state["child.first._extra_state"]
        del state["child.first._cached_scale"]
        del state["child.first._cached_zp"]
        with self.assertRaisesRegex(RuntimeError, "(Missing frozen site|not frozen)"):
            root.load_state_dict(state, strict=False)

    def test_fresh_plain_receiver_checks_promised_local_readiness(self):
        source = Pair()
        source.enable_calibration()
        source(torch.tensor([-1.0, 2.0]))
        source.freeze_qparams()
        root = nn.Module()
        root.child = source
        state = copy.deepcopy(root.state_dict())
        for suffix in ("_extra_state", "_cached_scale", "_cached_zp"):
            del state["child.first." + suffix]
        receiver = nn.Module()
        receiver.child = Pair()
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            receiver.load_state_dict(state, strict=False)

    def test_fresh_plain_receiver_cannot_hide_required_sites_in_capabilities(self):
        source = Pair()
        source.enable_calibration()
        source(torch.tensor([-1.0, 2.0]))
        source.freeze_qparams()
        root = nn.Module()
        root.child = source
        state = copy.deepcopy(root.state_dict())
        state["child._extra_state"]["capabilities"] = []
        for suffix in ("_extra_state", "_cached_scale", "_cached_zp"):
            del state["child.first." + suffix]
        receiver = nn.Module()
        receiver.child = Pair()
        with self.assertRaisesRegex(RuntimeError, "not frozen"):
            receiver.load_state_dict(state, strict=False)

    def test_whole_module_boundary_rejects_corrupt_mode_instead_of_float_execution(
        self,
    ):
        from tico.quantization.wrapq.lifecycle import validate_frozen_model

        model = Pair()
        model.enable_calibration()
        model(torch.tensor([-1.0, 2.0]))
        model.freeze_qparams()
        object.__setattr__(model, "_mode", "QUANT")
        with self.assertRaisesRegex(RuntimeError, "Corrupt WrapQ frozen readiness"):
            validate_frozen_model(model)

    def test_nonstrict_load_cannot_demote_a_child_of_a_frozen_model(self):
        model = Pair()
        model.child = Pair()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model(x)
        model.child(x)
        model.freeze_qparams()
        for mode in ("CALIB", "NO_QUANT"):
            with self.subTest(mode=mode):
                state = copy.deepcopy(model.state_dict())
                state["child._extra_state"]["mode"] = mode
                with self.assertRaisesRegex(
                    RuntimeError, "Inconsistent frozen wrapper modes"
                ):
                    model.load_state_dict(state, strict=False)

    def test_nonstrict_cannot_demote_wrapper_missing_mode_schema(self):
        model = prepare(nn.Linear(2, 2, bias=False).eval(), PTQConfig())
        model(torch.tensor([[-1.0, 2.0]]))
        frozen = convert(model)
        state = copy.deepcopy(frozen.state_dict())
        del state["wrapped._extra_state"]
        receiver = prepare(nn.Linear(2, 2, bias=False).eval(), PTQConfig())
        with self.assertRaisesRegex(RuntimeError, "Incomplete WrapQ wrapper state"):
            receiver.load_state_dict(state, strict=False)

    def test_optional_capability_is_explicit_and_persistent(self):
        model = Conditional()
        model.enable_calibration()
        x = torch.tensor([-1.0, 2.0])
        model(x)
        model.freeze_qparams()
        self.assertIsNone(model.last.frozen_state)
        model(x)
        with self.assertRaisesRegex(RuntimeError, "capability 'cache'"):
            model(x, use_cache=True)
        restored = Conditional()
        restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=False)
        self.assertTrue(torch.equal(model(x), restored(x)))
        with self.assertRaisesRegex(RuntimeError, "capability 'cache'"):
            restored(x, use_cache=True)

    def test_uncalibrated_linear_rejects_before_quant(self):
        model = prepare(nn.Linear(2, 2, bias=False).eval(), PTQConfig())
        with self.assertRaisesRegex(RuntimeError, "calibration"):
            convert(model)
        self.assertTrue(hasattr(model, "tico_quantizer"))
        self.assertIs(model.wrapped._mode, Mode.CALIB)

    def test_legacy_mutating_lifecycle_override_rejected(self):
        class Old(Pair):
            def freeze_qparams(self):
                super().freeze_qparams()

        with self.assertRaisesRegex(RuntimeError, "migrate lifecycle overrides"):
            freeze_qparams(Old())


if __name__ == "__main__":
    unittest.main()
