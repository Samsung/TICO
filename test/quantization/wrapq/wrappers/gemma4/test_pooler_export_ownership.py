# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
"""Lifecycle contracts after normal Gemma vision profile registration."""

import copy
import io
import unittest
from contextlib import ExitStack
from typing import cast
from unittest import mock

import torch

from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.wrapq.lifecycle import freeze_qparams, validate_frozen_model
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.base import ObserverBase
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_model import (
    QuantGemma4VisionModel,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_vision_pooler import (
    QuantGemma4VisionPooler,
)
from tico.quantization.wrapq.wrappers.quant_module_base import QuantModuleBase
from torch import nn

from test.quantization.quant_spec_helpers import make_affine_ptq_config
from test.quantization.wrapq.wrappers.gemma4.test_quant_vision_model import (
    _make_vision_config,
    _vision_position_ids,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_vision_pooler import (
    _make_pooler,
)


def _frozen_vision_model():
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel

    torch.manual_seed(1097)
    config = _make_vision_config(standardize=False)
    values = torch.randn(1, 16, 3 * config.patch_size**2)
    positions = _vision_position_ids(1, 16)
    model = QuantGemma4VisionModel(
        Gemma4VisionModel(config).eval(), qcfg=make_affine_ptq_config()
    ).eval()
    model.enable_calibration()
    with torch.no_grad():
        model(pixel_values=values, pixel_position_ids=positions, return_dict=True)
    # Include both imported locked state and an explicit execution-policy choice.
    pooler = cast(QuantGemma4VisionPooler, model.pooler.wrapped)
    cast(AffineObserverBase, pooler.obs_pool_in).load_qparams(
        torch.tensor(0.03125), torch.tensor(117), lock=True
    )
    pooler.obs_pool_out.disable_fake_quant()
    model.freeze_qparams()
    return model, values, positions


class _ObserverOwner(QuantModuleBase):
    def __init__(self, observer):
        super().__init__()
        self.observer = observer

    def _all_observers(self):
        return (self.observer,)


class _ConditionalPoolerOwner(QuantModuleBase):
    def __init__(self):
        super().__init__()
        self.pooler = QuantGemma4VisionPooler(
            _make_pooler(), qcfg=make_affine_ptq_config()
        ).eval()

    def _all_observers(self):
        return ()

    def _conditional_quant_subtrees(self):
        return {"vision": ("pooler",)}


class TestPoolerExportOwnership(unittest.TestCase):
    def test_actual_vision_export_preparation_preserves_valid_ownership(self):
        model, values, positions = _frozen_vision_model()
        validate_frozen_model(model)
        source_before = _state_contract(model.pooler.wrapped)
        adapter = model.as_export_module(pixel_position_ids=positions)
        self.assertEqual(source_before, _state_contract(model.pooler.wrapped))
        validate_frozen_model(model)
        with torch.no_grad():
            output = adapter(values).last_hidden_state
        self.assertEqual(tuple(output.shape), (4, 32))
        self.assertTrue(torch.isfinite(output).all())

    def test_refreeze_preserves_source_and_profile_snapshots_once_per_site(self):
        model, _values, positions = _frozen_vision_model()
        model.as_export_module(pixel_position_ids=positions)
        source = cast(QuantGemma4VisionPooler, model.pooler.wrapped)
        source_observers = _state_contract(source)["observers"]
        profile_observers = _state_contract(model.pooler_export.wrapped)["observers"]
        for name in (
            "obs_act_in",
            "obs_pool_in",
            "obs_pool_out",
            "obs_root_hidden_size",
            "obs_pool_matmul_out",
        ):
            with self.subTest(site=name):
                self.assertEqual(
                    source_observers[name]["snapshot"],
                    profile_observers[name]["snapshot"],
                )
        self.assertTrue(source_observers["obs_pool_in"]["snapshot"]["locked"])
        self.assertEqual(
            source_observers["obs_pool_in"]["snapshot"]["provenance"], "imported"
        )
        self.assertFalse(
            source_observers["obs_pool_out"]["snapshot"]["fake_quant_enabled"]
        )
        before = _state_contract(model)
        frozen = [
            observer
            for observer in model.modules()
            if isinstance(observer, ObserverBase) and observer.frozen_state is not None
        ]
        with ExitStack() as stack:
            installs = [
                stack.enter_context(
                    mock.patch.object(
                        observer,
                        "install_candidate",
                        wraps=observer.install_candidate,
                    )
                )
                for observer in frozen
            ]
            model.freeze_qparams()
        self.assertTrue(all(install.call_count == 1 for install in installs))
        self.assertEqual(before, _state_contract(model))
        validate_frozen_model(model)

    def test_strict_state_dict_roundtrip_with_equivalent_profile_topology(self):
        model, values, positions = _frozen_vision_model()
        adapter = model.as_export_module(pixel_position_ids=positions)
        model.second_pooler_profile = model.pooler.as_export_module(
            mode="prefill", output_length=16, pixel_position_ids=positions
        )
        validate_frozen_model(model)
        with torch.no_grad():
            expected = adapter(values).last_hidden_state
        before = _state_contract(model)
        receiver, _values, _positions = _frozen_vision_model()
        restored_adapter = receiver.as_export_module(pixel_position_ids=positions)
        receiver.second_pooler_profile = receiver.pooler.as_export_module(
            mode="prefill", output_length=16, pixel_position_ids=positions
        )
        result = receiver.load_state_dict(
            copy.deepcopy(model.state_dict()), strict=True
        )
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        validate_frozen_model(receiver)
        self.assertEqual(before, _state_contract(receiver))
        with torch.no_grad():
            actual = restored_adapter(values).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        receiver.freeze_qparams()
        self.assertEqual(before, _state_contract(receiver))

    def test_whole_module_roundtrip_after_actual_vision_preparation(self):
        model, values, positions = _frozen_vision_model()
        model.as_export_module(pixel_position_ids=positions)
        before = _state_contract(model)
        with torch.no_grad():
            expected = model.forward_export(values).last_hidden_state
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=False)
        validate_frozen_model(restored)
        self.assertEqual(before, _state_contract(restored))
        with torch.no_grad():
            actual = restored.forward_export(values).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        restored.freeze_qparams()
        self.assertEqual(before, _state_contract(restored))

    def test_two_registered_profiles_survive_failed_profile_construction(self):
        model, _values, positions = _frozen_vision_model()
        source = cast(QuantGemma4VisionPooler, model.pooler.wrapped)
        source_before = _state_contract(source)
        model.as_export_module(pixel_position_ids=positions)
        first = model.pooler_export
        hidden = torch.linspace(-1.0, 1.0, 16 * 32).reshape(1, 16, 32)
        with torch.no_grad():
            first_output = first(hidden)
        first_before = _state_contract(first.wrapped)
        first_matrix = first.wrapped.pool_weights.detach().clone()
        model.second_pooler_profile = source.as_export_module(
            output_length=16, pixel_position_ids=positions
        )
        second = model.second_pooler_profile
        second_before = _state_contract(second.wrapped)
        torch.testing.assert_close(
            second.wrapped.pool_weights, torch.eye(16).unsqueeze(0), rtol=0, atol=0
        )
        first_grid = first.wrapped.obs_pool_weight.require_frozen()
        second_grid = second.wrapped.obs_pool_weight.require_frozen()
        assert first_grid.scale is not None and second_grid.scale is not None
        torch.testing.assert_close(
            second_grid.scale, first_grid.scale * 4, rtol=0, atol=0
        )
        with self.assertRaisesRegex(ValueError, "Cannot pool"):
            source.as_export_module(output_length=3, pixel_position_ids=positions)
        self.assertEqual(source_before, _state_contract(source))
        self.assertEqual(first_before, _state_contract(first.wrapped))
        self.assertEqual(second_before, _state_contract(second.wrapped))
        torch.testing.assert_close(
            first.wrapped.pool_weights, first_matrix, rtol=0, atol=0
        )
        with torch.no_grad():
            torch.testing.assert_close(first(hidden), first_output, rtol=0, atol=0)
        validate_frozen_model(model)
        model.freeze_qparams()
        self.assertEqual(source_before, _state_contract(source))
        self.assertEqual(first_before, _state_contract(first.wrapped))
        self.assertEqual(second_before, _state_contract(second.wrapped))

    def test_unrelated_observer_owners_still_reject(self):
        shared = MinMaxObserver(name="unrelated_shared_site")
        shared.collect(torch.tensor([-1.0, 2.0]))
        first = _ObserverOwner(shared)
        second = _ObserverOwner(shared)
        first.freeze_qparams()
        second.freeze_qparams()
        model = nn.Module()
        model.first = first
        model.second = second
        with self.assertRaisesRegex(RuntimeError, "conflicting wrapper owners"):
            validate_frozen_model(model)
        with self.assertRaisesRegex(RuntimeError, "conflicting wrapper owners"):
            freeze_qparams(model)

    def test_unavailable_source_rejects_before_profile_work_or_child_execution(self):
        owner = _ConditionalPoolerOwner()
        positions = _vision_position_ids(1, 16)
        stage = owner.pooler.as_export_module(
            output_length=4, pixel_position_ids=positions
        )
        owner.enable_calibration()
        owner.freeze_qparams()
        restored = _ConditionalPoolerOwner()
        restored.load_state_dict(copy.deepcopy(owner.state_dict()), strict=True)
        for candidate in (owner, restored):
            with self.subTest(restored=candidate is restored):
                with mock.patch.object(
                    candidate.pooler, "_build_export_pool_weights"
                ) as build:
                    with self.assertRaisesRegex(
                        RuntimeError, "conditional subtree.*unavailable"
                    ):
                        candidate.pooler.as_export_module(
                            output_length=4, pixel_position_ids=positions
                        )
                    build.assert_not_called()
        with mock.patch.object(stage.wrapped, "forward_export") as execute:
            with self.assertRaisesRegex(
                RuntimeError, "conditional subtree.*unavailable"
            ):
                stage(torch.ones(1, 16, 32))
            execute.assert_not_called()

    def test_stale_mode_rejects_before_profile_execution(self):
        owner = _ConditionalPoolerOwner()
        positions = _vision_position_ids(1, 16)
        stage = owner.pooler.as_export_module(
            output_length=4, pixel_position_ids=positions
        )
        owner.enable_calibration()
        owner.pooler(
            torch.linspace(-1.0, 1.0, 16 * 32).reshape(1, 16, 32),
            positions,
            torch.zeros(1, 16, dtype=torch.bool),
            4,
        )
        owner.freeze_qparams()
        self.assertTrue(owner.has_capability("vision"))
        with mock.patch.object(stage.wrapped, "forward_export") as execute:
            with self.assertRaisesRegex(RuntimeError, "mode does not match.*rebuild"):
                stage(torch.ones(1, 16, 32))
            execute.assert_not_called()

    def test_profile_device_movement_does_not_move_source_observers(self):
        model, _values, positions = _frozen_vision_model()
        model.as_export_module(pixel_position_ids=positions)
        source = cast(QuantGemma4VisionPooler, model.pooler.wrapped)
        before = _state_contract(source)
        profile = model.pooler_export
        profile.to(device="meta")
        self.assertTrue(
            all(buffer.device.type == "meta" for buffer in profile.buffers())
        )
        self.assertTrue(all(buffer.device.type == "cpu" for buffer in source.buffers()))
        self.assertEqual(before, _state_contract(source))
        validate_frozen_model(source)
