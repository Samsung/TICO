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

"""Unit tests for GPTQ qparam handoff to folded linear layers."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.qparams as qparams

import torch
import torch.nn as nn
from tico.quantization.wrapq.utils.linear_folding import fold_input_affine_into_linear


class _FakeObserver:
    """Record qparams loaded by the handoff helper."""

    def __init__(self) -> None:
        self.loaded = None

    def load_qparams(self, scale, zero, lock) -> None:
        """Store one qparam handoff for later assertions."""
        self.loaded = (scale, zero, lock)


class _FakeQuantModule(nn.Module):
    """Expose a folded linear layer through a PTQ-like module interface."""

    def __init__(self, linear: nn.Linear, observer: _FakeObserver) -> None:
        super().__init__()
        self.fp_name = "vision_tower.patch_embedder.input_proj"
        self.module = linear
        self._observer = observer

    def get_observer(self, name: str):
        """Return the fake weight observer for the requested role."""
        return self._observer if name == "weight" else None


class FoldedLinearQParamHandoffTest(unittest.TestCase):
    """Validate scale adjustment and conservative rejection behavior."""

    def _inject(self, linear: nn.Linear):
        """Inject one synthetic GPTQ record into a fake PTQ module."""
        observer = _FakeObserver()
        root = nn.Module()
        root.linear = _FakeQuantModule(linear, observer)
        quantizer = SimpleNamespace(
            scale=torch.tensor([0.25, 0.5]),
            zero=torch.tensor([3, 4]),
        )

        with patch.object(qparams, "QuantModuleBase", _FakeQuantModule), patch.object(
            qparams,
            "AffineObserverBase",
            _FakeObserver,
        ):
            result = qparams.inject_gptq_qparams(
                root,
                {root.linear.fp_name: quantizer},
            )
        return observer, quantizer, result

    def test_positive_uniform_fold_scales_reused_weight_qparams(self) -> None:
        """Multiply GPTQ weight scales by the folded input scale."""
        source = nn.Linear(3, 2, bias=False)
        folded = fold_input_affine_into_linear(
            source,
            scale=2.0,
            shift=-1.0,
        )

        observer, quantizer, result = self._inject(folded)

        loaded_scale, loaded_zero, locked = observer.loaded
        torch.testing.assert_close(loaded_scale, quantizer.scale * 2.0)
        self.assertIs(loaded_zero, quantizer.zero)
        self.assertTrue(locked)
        self.assertEqual(result, {"matched": 1, "missed": 0, "unused": 0})

    def test_unfolded_linear_keeps_reused_weight_qparams(self) -> None:
        """Keep GPTQ qparams unchanged for a linear layer without fold metadata."""
        linear = nn.Linear(3, 2, bias=False)

        observer, quantizer, result = self._inject(linear)

        loaded_scale, loaded_zero, locked = observer.loaded
        self.assertIs(loaded_scale, quantizer.scale)
        self.assertIs(loaded_zero, quantizer.zero)
        self.assertTrue(locked)
        self.assertEqual(result, {"matched": 1, "missed": 0, "unused": 0})

    def test_nonuniform_fold_requires_weight_qparam_recomputation(self) -> None:
        """Reject GPTQ qparam reuse when one scalar multiplier is insufficient."""
        source = nn.Linear(3, 2, bias=False)
        folded = fold_input_affine_into_linear(
            source,
            scale=torch.tensor([1.0, 2.0, 3.0]),
            shift=0.0,
        )

        with self.assertRaisesRegex(RuntimeError, "Recompute"):
            self._inject(folded)


if __name__ == "__main__":
    unittest.main()
