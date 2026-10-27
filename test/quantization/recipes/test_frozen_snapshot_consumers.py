# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Artifact metadata readers consume owned snapshots without changing source state."""

import unittest
from types import SimpleNamespace
from typing import cast

import torch

from tico.quantization.recipes.debug.export_regression import _state_contract
from tico.quantization.recipes.export.gemma4 import _observer_manifest
from tico.quantization.recipes.export.gemma4_assistant import _quantized_lm_head_payload
from tico.quantization.wrapq.observers.affine_base import AffineObserverBase
from tico.quantization.wrapq.observers.identity import IdentityObserver
from tico.quantization.wrapq.wrappers.gemma4_assistant.quant_for_causal_lm import (
    QuantGemma4AssistantForCausalLM,
    QuantGemma4AssistantLMHeadWeight,
)


class TestFrozenSnapshotConsumers(unittest.TestCase):
    def test_sparse_head_reads_exact_weight_snapshot_without_aliasing(self):
        linear = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[0.25, 0.5], [-0.25, 1.0]]))
        head = QuantGemma4AssistantLMHeadWeight(linear).eval()
        head.enable_calibration()
        cast(AffineObserverBase, head.obs_weight).load_qparams(
            torch.tensor([0.25, 0.25]), torch.tensor([0, 1])
        )
        head.freeze_qparams()
        before = _state_contract(head)
        assistant = cast(QuantGemma4AssistantForCausalLM, SimpleNamespace(lm_head=head))
        payload = _quantized_lm_head_payload(assistant)
        torch.testing.assert_close(
            payload["lm_head_weight_int"],
            torch.tensor([[1, 2], [0, 5]], dtype=torch.uint8),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            payload["lm_head_weight_scale"], torch.tensor([0.25, 0.25]), atol=0, rtol=0
        )
        payload["lm_head_weight_scale"].add_(10)
        self.assertEqual(before, _state_contract(head))

    def test_float_manifest_has_no_affine_sentinel_grid(self):
        observer = IdentityObserver(name="float_boundary")
        observer.install_candidate(observer.derive_candidate())
        contract = _observer_manifest(observer)
        self.assertNotIn("scale", contract)
        self.assertNotIn("zero_point", contract)
        self.assertEqual(contract["name"], "float_boundary")
