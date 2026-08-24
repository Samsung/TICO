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

"""Tests for the canonical Gemma4 static vision-prefill profile."""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import unittest
from types import SimpleNamespace
from typing import cast

import torch
import torch.nn as nn

from tico.quantization.recipes.debug.static_gemma4_runtime import StaticGemma4Runtime
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    build_gemma4_vision_prefill_export_module,
    Gemma4VisionPrefillExportAdapter,
)
from tico.quantization.wrapq.wrappers.gemma4.static_vision_profile import (
    Gemma4StaticVisionProfile,
)


class _FakePTQWrapper(nn.Module):
    """Expose one module through the PTQWrapper-compatible attribute."""

    def __init__(self, wrapped: nn.Module) -> None:
        super().__init__()
        self.wrapped = wrapped


class _FakeStaticVisionModel(nn.Module):
    """Return a model-output-like object from a static vision graph."""

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        """Return deterministic hidden states from the pixel-values-only ABI."""
        return SimpleNamespace(last_hidden_state=pixel_values + 1.0)


class _ScaleProjection(nn.Module):
    """Apply a visible projection after the static vision model."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Scale hidden states so the test observes projection execution."""
        return hidden_states * 2.0


class _FakeVisionModel(nn.Module):
    """Record static specialization requests."""

    def __init__(self) -> None:
        super().__init__()
        self.export_calls = 0
        self.last_position_ids = None

    def as_export_module(
        self,
        mode: str,
        *,
        pixel_position_ids: torch.Tensor,
    ) -> nn.Module:
        """Return one deterministic static model."""
        if mode != "prefill":
            raise ValueError(mode)
        self.export_calls += 1
        self.last_position_ids = pixel_position_ids.detach().clone()
        return _FakeStaticVisionModel()


class _FakeWrappedModel(nn.Module):
    """Provide the hierarchy consumed by the canonical builder."""

    def __init__(self) -> None:
        super().__init__()
        self.vision_model = _FakeVisionModel()
        self.vision_tower = _FakePTQWrapper(self.vision_model)
        self.embed_vision = _ScaleProjection()


class TestGemma4StaticVisionExportProfile(unittest.TestCase):
    """Validate adapter construction and single-profile runtime caching."""

    def setUp(self) -> None:
        """Create one valid fixed patch-position profile."""
        self.position_ids = torch.tensor(
            [[[0, 0], [1, 0], [0, 1], [1, 1]]],
            dtype=torch.long,
        )
        self.pixel_values = torch.randn(1, 4, 3)
        self.profile = Gemma4StaticVisionProfile(
            name="tiny",
            visual_start_idx=0,
            num_visual_tokens=4,
            max_soft_tokens=4,
            patch_grid_height=2,
            patch_grid_width=2,
            patch_size=1,
            pooling_kernel_size=1,
        )

    def test_builder_returns_complete_static_vision_stage(self) -> None:
        """The shared builder should specialize vision and retain projection."""
        wrapped_model = _FakeWrappedModel()

        module = build_gemma4_vision_prefill_export_module(
            wrapped_model,
            pixel_position_ids=self.position_ids,
        )

        self.assertIsInstance(module, Gemma4VisionPrefillExportAdapter)
        self.assertEqual(wrapped_model.vision_model.export_calls, 1)
        torch.testing.assert_close(
            wrapped_model.vision_model.last_position_ids,
            self.position_ids,
        )
        torch.testing.assert_close(
            module(self.pixel_values),
            (self.pixel_values + 1.0) * 2.0,
        )

    def test_runtime_builds_once_and_reuses_identical_profile(self) -> None:
        """Repeated images with the same position layout should reuse one graph."""
        wrapped_model = _FakeWrappedModel()
        runtime = cast(
            StaticGemma4Runtime,
            SimpleNamespace(
                _wrapped_model=wrapped_model,
                device=torch.device("cpu"),
                vision_prefill=None,
                vision_profile=self.profile,
            ),
        )

        first = StaticGemma4Runtime._get_or_create_vision_prefill(
            runtime,
            self.position_ids,
        )
        second = StaticGemma4Runtime._get_or_create_vision_prefill(
            runtime,
            self.position_ids.clone(),
        )

        self.assertIs(first, second)
        self.assertEqual(wrapped_model.vision_model.export_calls, 1)

    def test_runtime_rejects_a_different_position_profile(self) -> None:
        """A new patch layout must use a new static runtime instance."""
        wrapped_model = _FakeWrappedModel()
        runtime = cast(
            StaticGemma4Runtime,
            SimpleNamespace(
                _wrapped_model=wrapped_model,
                device=torch.device("cpu"),
                vision_prefill=None,
                vision_profile=self.profile,
            ),
        )
        StaticGemma4Runtime._get_or_create_vision_prefill(
            runtime,
            self.position_ids,
        )
        changed = self.position_ids.clone()
        changed[0, 0, 0] = 7

        with self.assertRaisesRegex(ValueError, "does not match"):
            StaticGemma4Runtime._get_or_create_vision_prefill(runtime, changed)

    def test_runtime_requires_processor_position_ids(self) -> None:
        """Static vision cannot be specialized without processor coordinates."""
        runtime = cast(
            StaticGemma4Runtime,
            SimpleNamespace(
                _wrapped_model=_FakeWrappedModel(),
                device=torch.device("cpu"),
                vision_prefill=None,
                vision_profile=self.profile,
            ),
        )

        with self.assertRaisesRegex(ValueError, "requires image_position_ids"):
            StaticGemma4Runtime._get_or_create_vision_prefill(runtime, None)


if __name__ == "__main__":
    unittest.main()
