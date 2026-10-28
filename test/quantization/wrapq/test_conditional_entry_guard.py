# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Conditional entry remains guarded outside the owning model's forward."""

import copy
import io
import unittest
from unittest.mock import patch

import torch

from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.utils.static_export import ExportCapabilityGuard
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn


class _Leaf(QuantModuleBase):
    def __init__(self):
        super().__init__()
        self.obs = MinMaxObserver(name="input")

    def _all_observers(self):
        return (self.obs,)

    def forward(self, value):
        return self._fq(value, self.obs)


class _Owner(QuantModuleBase):
    def __init__(self):
        super().__init__()
        self.branch = _Leaf()
        self.unused_weight = nn.Parameter(torch.ones(3))

    def _all_observers(self):
        return ()

    def _conditional_quant_subtrees(self):
        return {"optional": ("branch",)}

    def forward(self, value):
        self.require_capability("optional")
        return self.branch(value)


class _Stage(nn.Module):
    def __init__(self, owner):
        super().__init__()
        self.guard = ExportCapabilityGuard(owner, "optional")
        self.branch = owner.branch

    def forward(self, value):
        self.guard.require()
        return self.branch(value)


class TestConditionalEntryGuard(unittest.TestCase):
    def test_whole_module_restore_does_not_depend_on_constructor_hooks(self):
        source = _Owner()
        source.enable_calibration()
        source.freeze_qparams()
        buffer = io.BytesIO()
        torch.save(source, buffer)
        buffer.seek(0)
        with patch.object(_Leaf, "__init__", side_effect=AssertionError("constructor")):
            restored = torch.load(buffer, weights_only=False)
        with patch.object(restored.branch, "forward") as forward:
            with self.assertRaisesRegex(
                RuntimeError, "conditional subtree.*unavailable"
            ):
                restored.branch(torch.ones(1))
            forward.assert_not_called()

    def test_unavailable_direct_child_rejects_before_forward_after_restore(self):
        source = _Owner()
        source.enable_calibration()
        source.freeze_qparams()
        restored = _Owner()
        restored.load_state_dict(copy.deepcopy(source.state_dict()))
        for model in (source, restored):
            with self.subTest(restored=model is restored):
                with patch.object(model.branch, "forward") as forward:
                    with self.assertRaisesRegex(
                        RuntimeError, "conditional subtree.*unavailable"
                    ):
                        model.branch(torch.ones(1))
                    forward.assert_not_called()
                self.assertFalse(model.branch.has_capability("required"))
                with self.assertRaisesRegex(
                    RuntimeError, "conditional subtree.*unavailable"
                ):
                    model.branch.require_capability("required")

    def test_available_child_entry_and_stage_preserve_output(self):
        owner = _Owner()
        owner.enable_calibration()
        value = torch.tensor([-1.0, 2.0])
        owner.branch(value)
        owner.freeze_qparams()
        stage = _Stage(owner)
        torch.testing.assert_close(stage(value), owner(value), rtol=0, atol=0)
        self.assertNotIn("unused_weight", stage.state_dict())
        self.assertNotIn(owner, tuple(stage.modules()))

    def test_stage_created_before_freeze_checks_live_named_owner(self):
        owner = _Owner()
        owner.enable_calibration()
        stage = _Stage(owner)
        owner.freeze_qparams()
        with patch.object(owner.branch, "forward") as forward:
            with self.assertRaisesRegex(
                RuntimeError, "capability 'optional'.*unavailable"
            ):
                stage(torch.ones(1))
            forward.assert_not_called()

    def test_unavailable_owner_cannot_persist_vacuous_ready_capability(self):
        source = _Owner()
        source.enable_calibration()
        source.freeze_qparams()
        state = copy.deepcopy(source.state_dict())
        state["branch._extra_state"]["capabilities"] = ["required"]
        with self.assertRaisesRegex(RuntimeError, "unavailable subtree capabilities"):
            _Owner().load_state_dict(state, strict=False)

    def test_plain_float_stages_require_explicit_no_quant(self):
        owner = nn.Identity()
        for mode in (None, Mode.QUANT, Mode.CALIB):
            with (
                self.subTest(mode=mode),
                self.assertRaisesRegex(TypeError, "capability owner"),
            ):
                ExportCapabilityGuard(owner, "optional", mode=mode)
        guard = ExportCapabilityGuard(owner, "optional", mode=Mode.NO_QUANT)
        guard.require()

    def test_explicit_float_mode_cannot_bypass_live_quantized_owner(self):
        owner = _Owner()
        owner.enable_calibration()
        owner.freeze_qparams()
        with self.assertRaisesRegex(RuntimeError, "capability 'optional'.*unavailable"):
            ExportCapabilityGuard(owner, "optional", mode=Mode.NO_QUANT)

    def test_plain_float_owner_cannot_authorize_explicit_quantized_stage(self):
        owner = nn.Identity()
        object.__setattr__(owner, "_mode", Mode.NO_QUANT)
        with self.assertRaisesRegex(TypeError, "capability owner"):
            ExportCapabilityGuard(owner, "optional", mode=Mode.QUANT)
        object.__setattr__(owner, "_mode", Mode.QUANT)
        with self.assertRaisesRegex(TypeError, "capability owner"):
            ExportCapabilityGuard(owner, "optional", mode=Mode.NO_QUANT)

    def test_mode_bound_stage_rejects_stale_policy_after_conversion(self):
        owner = _Owner()
        guard = ExportCapabilityGuard(owner, "optional", mode=Mode.NO_QUANT)
        owner.enable_calibration()
        owner.branch(torch.tensor([-1.0, 2.0]))
        owner.freeze_qparams()
        with self.assertRaisesRegex(RuntimeError, "stage mode.*rebuild"):
            guard.require()
        with self.assertRaisesRegex(RuntimeError, "stage mode.*rebuild"):
            ExportCapabilityGuard(owner, "optional", mode=Mode.NO_QUANT)
        ExportCapabilityGuard(owner, "optional", mode=Mode.QUANT).require()
