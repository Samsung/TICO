# Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

import unittest

import torch

from tico.quantization.wrapq.observers.identity import IdentityObserver


class TestIdentityObserver(unittest.TestCase):
    """Verify that IdentityObserver behaves as a transparent FP-pass."""

    def setUp(self):
        self.obs = IdentityObserver(name="act_fp32")

    def test_initial_state(self):
        self.assertFalse(self.obs.enabled, msg="Observer must start disabled")
        self.assertIsNone(self.obs.frozen_state)
        self.assertEqual(self.obs.derive_candidate().kind, "identity")

    def test_collect_is_noop(self):
        x = torch.randn(8, 16)
        self.obs.collect(x)  # should have *no* effect
        self.assertFalse(self.obs.enabled, msg="`enabled` flag must stay False")
        self.assertIsNone(self.obs.frozen_state)

    def test_explicit_stateless_readiness(self):
        self.assertIsNone(self.obs.compute_qparams())
        state = self.obs.require_frozen()
        self.assertEqual(state.kind, "identity")
        self.assertEqual(state.provenance, "stateless")
        self.assertIsNone(state.scale)
        self.assertIsNone(state.zero_point)

    def test_fake_quant_identity(self):
        x = torch.randn(4, 4)
        y = self.obs.fake_quant(x)
        # IdentityObserver returns the *same* object, not a clone
        self.assertIs(y, x)
        # Values must of course be identical
        self.assertTrue(torch.allclose(y, x))
