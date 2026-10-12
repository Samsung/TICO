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
from typing import Optional, Tuple
from unittest.mock import patch

import tico.quantization.recipes.qparams as qparams

import torch

from tico.quantization.wrapq.dtypes import DType, UINT8


class FakeObserver:
    """Fake affine observer that records loaded quantization parameters."""

    def __init__(self, dtype=UINT8):
        self.loaded: Optional[Tuple[torch.Tensor, torch.Tensor, bool]] = None
        self.dtype = dtype

    def load_qparams(self, scale, zero, lock):
        """Record the qparams passed by the injection helper."""
        self.loaded = (scale, zero, lock)


def _fake_quantizer(scale, zero, bits=8):
    """Return a GPTQ-quantizer-like record on the unsigned ``[0, 2**bits - 1]`` grid."""
    return SimpleNamespace(scale=scale, zero=zero, maxq=torch.tensor(2**bits - 1))


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

        found_owner, quantizers = qparams.find_gptq_quantizers(root)  # type: ignore[arg-type]

        self.assertIs(found_owner, owner)
        self.assertEqual(quantizers, owner.quantizers)

    def test_clear_gptq_quantizers_removes_all_common_owner_attributes(self):
        """Quantizer cleanup should remove attached dictionaries from all candidates."""
        root = SimpleNamespace(quantizers={"root": object()})
        root.wrapped = SimpleNamespace(quantizers={"wrapped": object()})
        root.wrapped.module = SimpleNamespace(quantizers={"module": object()})

        qparams.clear_gptq_quantizers(root)  # type: ignore[arg-type]

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
            "matched": _fake_quantizer(scale, zero),
            "unused": _fake_quantizer(torch.tensor([1.0]), torch.tensor([0])),
        }

        with patch.object(qparams, "QuantModuleBase", FakeQuantModule), patch.object(
            qparams, "AffineObserverBase", FakeObserver
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                stats = qparams.inject_gptq_qparams(root, quantizers, verbose=True)

        self.assertEqual(stats, {"matched": 1, "missed": 1, "unused": 1})
        self.assertEqual(matched_observer.loaded, (scale, zero, True))
        self.assertIsNone(missed_observer.loaded)

    def test_inject_gptq_qparams_maps_zero_point_into_signed_observer_domain(self):
        """A signed observer dtype receives the GPTQ zero point shifted by qmin."""
        observer = FakeObserver(dtype=DType.int(4))
        root = torch.nn.Sequential(FakeQuantModule("linear", observer))
        # Symmetric 4-bit GPTQ keeps zero = (maxq + 1) / 2 = 8 on [0, 15];
        # the same grid on int4 [-8, 7] has zero point 0.
        quantizers = {
            "linear": _fake_quantizer(
                torch.tensor([0.25, 0.5]), torch.tensor([8.0, 8.0]), bits=4
            )
        }

        with patch.object(qparams, "QuantModuleBase", FakeQuantModule), patch.object(
            qparams, "AffineObserverBase", FakeObserver
        ):
            stats = qparams.inject_gptq_qparams(root, quantizers)

        self.assertEqual(stats, {"matched": 1, "missed": 0, "unused": 0})
        assert observer.loaded is not None
        loaded_scale, loaded_zero, locked = observer.loaded
        torch.testing.assert_close(loaded_scale, torch.tensor([0.25, 0.5]))
        torch.testing.assert_close(loaded_zero, torch.tensor([0.0, 0.0]))
        self.assertTrue(locked)

    def test_inject_gptq_qparams_rejects_level_count_mismatch(self):
        """A GPTQ grid with a different number of levels than the observer fails."""
        observer = FakeObserver(dtype=DType.uint(8))
        root = torch.nn.Sequential(FakeQuantModule("model.layers.0.q", observer))
        quantizers = {
            "model.layers.0.q": _fake_quantizer(
                torch.tensor([0.5]), torch.tensor([3.0]), bits=4
            )
        }

        with patch.object(qparams, "QuantModuleBase", FakeQuantModule), patch.object(
            qparams, "AffineObserverBase", FakeObserver
        ):
            with self.assertRaisesRegex(RuntimeError, "model.layers.0.q.*16 integer"):
                qparams.inject_gptq_qparams(root, quantizers)
        self.assertIsNone(observer.loaded)

    def test_gptq_zero_point_for_observer_is_identity_for_unsigned_dtypes(self):
        """Unsigned observer dtypes keep the GPTQ zero point unchanged."""
        zero = torch.tensor([[3.0], [15.0]])
        mapped = qparams.gptq_zero_point_for_observer(
            _fake_quantizer(torch.tensor([[1.0], [1.0]]), zero, bits=4),
            FakeObserver(dtype=DType.uint(4)),
            module_name="linear",
        )
        self.assertIs(mapped, zero)
