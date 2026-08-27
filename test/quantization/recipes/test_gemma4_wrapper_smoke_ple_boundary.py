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

"""Tests for the Gemma4 wrapper-smoke PLE producer boundary."""

import unittest
from unittest.mock import patch

import torch

from tico.quantization.recipes.debug.wrapper_smoke.case import ForwardInput
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    Gemma4BaseCase,
    Gemma4TextDecoderLayerBaseCase,
)
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.qscheme import QScheme


class _FakeQuantDecoder(torch.nn.Module):
    """Expose the subset of QuantGemma4TextDecoderLayer used by the smoke case."""

    hidden_size_per_layer_input = 2

    def __init__(self) -> None:
        super().__init__()
        self._mode = Mode.CALIB
        self.export_observer = None

    def _make_obs(self, name: str) -> MinMaxObserver:
        """Create the signed per-tensor observer used by the synthetic PLE producer."""
        return MinMaxObserver(
            name=name,
            dtype=DType.int(16),
            qscheme=QScheme.PER_TENSOR_SYMM,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        per_layer_input: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Return PLE so the test can observe calibration and fake quantization."""
        del hidden_states, kwargs
        if per_layer_input is None:
            raise RuntimeError("per_layer_input is required by this test.")
        return per_layer_input

    def as_export_module(
        self,
        mode: str,
        *,
        return_kv: bool,
        per_layer_input_observer=None,
    ) -> torch.nn.Module:
        """Record the observer routed into the real decoder export adapter."""
        del mode, return_kv
        self.export_observer = per_layer_input_observer
        return torch.nn.Identity()


class _FakePTQWrapper(torch.nn.Module):
    """Mimic the PTQWrapper boundary used by wrapper-smoke."""

    def __init__(self, wrapped: torch.nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped

    def forward(self, *args, **kwargs):
        """Delegate calls to the wrapped quant module."""
        return self.wrapped(*args, **kwargs)


class TestGemma4WrapperSmokePLEBoundary(unittest.TestCase):
    """Verify calibration, quant simulation, and export share one PLE observer."""

    def test_case_calibrates_and_routes_external_ple_observer(self) -> None:
        """The standalone decoder case should model the missing text-model producer."""
        case = Gemma4TextDecoderLayerBaseCase()
        wrapped = _FakeQuantDecoder()
        prepared = _FakePTQWrapper(wrapped)

        case.after_prepare(prepared, {})
        observer = case._per_layer_input_observer
        self.assertIsNotNone(observer)
        assert observer is not None

        hidden = torch.zeros(1, 2, 2)
        per_layer_input = torch.tensor(
            [[[-0.91, -0.13], [0.27, 1.03]]],
            dtype=torch.float32,
        )
        sample = ForwardInput(
            (),
            {
                "hidden_states": hidden,
                "per_layer_input": per_layer_input,
            },
        )

        calibration_output = case.forward(prepared, sample)
        torch.testing.assert_close(calibration_output, per_layer_input)
        self.assertAlmostEqual(float(observer.min_val), -0.91, places=5)
        self.assertAlmostEqual(float(observer.max_val), 1.03, places=5)

        # Avoid invoking the public quantizer in this focused lifecycle test;
        # the case override must still freeze its separately owned observer.
        with patch.object(
            Gemma4BaseCase,
            "convert_model",
            return_value=prepared,
        ):
            quantized = case.convert_model(prepared, {})

        self.assertTrue(observer.has_qparams)
        wrapped._mode = Mode.QUANT
        quantized_output = case.forward(quantized, sample)
        torch.testing.assert_close(
            quantized_output,
            observer.fake_quant(per_layer_input),
        )

        case.export_module(quantized, {})
        self.assertIs(wrapped.export_observer, observer)

    def test_export_rejects_ple_without_prepared_boundary_observer(self) -> None:
        """PLE export must not silently fall back to a floating-point placeholder."""
        case = Gemma4TextDecoderLayerBaseCase()
        wrapped = _FakeQuantDecoder()
        wrapped._mode = Mode.QUANT
        quantized = _FakePTQWrapper(wrapped)

        with self.assertRaisesRegex(RuntimeError, "PLE boundary observer"):
            case.export_module(quantized, {})


if __name__ == "__main__":
    unittest.main()
