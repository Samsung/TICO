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

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.qparams as qparams

import torch


class FakeObserver:
    """Fake affine observer that records loaded quantization parameters."""

    def __init__(self):
        self.loaded = None

    def load_qparams(self, scale, zero, lock):
        """Record the qparams passed by the injection helper."""
        self.loaded = (scale, zero, lock)


class FakeQuantModule(torch.nn.Module):
    """Fake quant module with a named weight observer."""

    def __init__(self, fp_name, observer=None):
        super().__init__()
        self.fp_name = fp_name
        self._observer = observer if observer is not None else FakeObserver()

    def get_observer(self, name):
        """Return the fake observer when the requested name is 'weight'."""
        if name == "weight":
            return self._observer
        return None


class TestRecipeQParams(unittest.TestCase):
    def test_find_gptq_quantizers_checks_common_owner_locations(self):
        """GPTQ quantizer lookup should search wrapper owner locations."""
        owner = SimpleNamespace(quantizers={"linear": object()})
        root = SimpleNamespace(wrapped=SimpleNamespace(module=owner))

        found_owner, quantizers = qparams.find_gptq_quantizers(root)

        self.assertIs(found_owner, owner)
        self.assertEqual(quantizers, owner.quantizers)

    def test_clear_gptq_quantizers_removes_all_common_owner_attributes(self):
        """Quantizer cleanup should remove attached dictionaries from all candidates."""
        root = SimpleNamespace(quantizers={"root": object()})
        root.wrapped = SimpleNamespace(quantizers={"wrapped": object()})
        root.wrapped.module = SimpleNamespace(quantizers={"module": object()})

        qparams.clear_gptq_quantizers(root)

        self.assertFalse(hasattr(root, "quantizers"))
        self.assertFalse(hasattr(root.wrapped, "quantizers"))
        self.assertFalse(hasattr(root.wrapped.module, "quantizers"))

    def test_inject_gptq_qparams_reports_matched_missed_and_unused(self):
        """GPTQ qparam injection should load matched observers and report coverage."""
        matched_observer = FakeObserver()
        missed_observer = FakeObserver()
        root = torch.nn.Sequential(
            FakeQuantModule("matched", matched_observer),
            FakeQuantModule("missed", missed_observer),
        )
        scale = torch.tensor([0.5])
        zero = torch.tensor([1])
        quantizers = {
            "matched": SimpleNamespace(scale=scale, zero=zero),
            "unused": SimpleNamespace(
                scale=torch.tensor([1.0]), zero=torch.tensor([0])
            ),
        }

        with patch.object(qparams, "QuantModuleBase", FakeQuantModule), patch.object(
            qparams, "AffineObserverBase", FakeObserver
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                stats = qparams.inject_gptq_qparams(root, quantizers, verbose=True)

        self.assertEqual(stats, {"matched": 1, "missed": 1, "unused": 1})
        self.assertEqual(matched_observer.loaded, (scale, zero, True))
        self.assertIsNone(missed_observer.loaded)
