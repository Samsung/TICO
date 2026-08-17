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

"""Tests for Qwen3-VL static vision profile selection in the debug runtime."""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from tico.quantization.recipes.debug.static_qwen3_vl_runtime import (
    StaticQwen3VLTextLayerRuntime,
)
from tico.quantization.wrapq.wrappers.qwen_vl.vision_profile import Qwen3VLVisionProfile


class _FakeStaticVision(nn.Module):
    """Expose a pixel-only static vision ABI for one profile."""

    def __init__(self, profile: Qwen3VLVisionProfile) -> None:
        super().__init__()
        self.profile = profile
        self.calls = 0

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Return deterministic features and record pixel-only invocation."""
        self.calls += 1
        return pixel_values.mean(dim=-1), (pixel_values.sum(dim=-1),)


class _FakeVisionWrapper(nn.Module):
    """Record profile specialization requests from the runtime."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[Qwen3VLVisionProfile] = []

    def as_export_module(
        self,
        mode: str,
        *,
        grid_thw,
    ) -> nn.Module:
        """Return a new static adapter for the requested grid."""
        if mode != "prefill":
            raise ValueError(mode)
        profile = Qwen3VLVisionProfile.from_grid_thw(grid_thw)
        self.requests.append(profile)
        return _FakeStaticVision(profile)


class TestStaticQwen3VLVisionProfile(unittest.TestCase):
    """Validate runtime-side static vision profile caching and dispatch."""

    @staticmethod
    def _make_runtime() -> StaticQwen3VLTextLayerRuntime:
        """Create an uninitialized runtime with only vision-profile state."""
        runtime = object.__new__(StaticQwen3VLTextLayerRuntime)
        runtime.device = torch.device("cpu")
        runtime.qwen_model = SimpleNamespace(
            visual=SimpleNamespace(spatial_merge_size=2)
        )
        runtime._vision_model = _FakeVisionWrapper()
        runtime.vision_prefill_adapters = {}
        return runtime

    def test_same_profile_reuses_one_static_adapter(self) -> None:
        """Equivalent processor grids should select the same cached module."""
        runtime = self._make_runtime()
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)

        first = runtime._get_or_create_vision_prefill_adapter(grid)
        second = runtime._get_or_create_vision_prefill_adapter(grid.clone())

        self.assertIs(first, second)
        expected = Qwen3VLVisionProfile(1, 4, 4)
        self.assertEqual(runtime._vision_model.requests, [expected])
        self.assertEqual(list(runtime.vision_prefill_adapters), [expected])

    def test_different_profiles_create_distinct_static_adapters(self) -> None:
        """Different grids should select independently specialized modules."""
        runtime = self._make_runtime()

        first = runtime._get_or_create_vision_prefill_adapter(
            torch.tensor([[1, 4, 4]], dtype=torch.long)
        )
        second = runtime._get_or_create_vision_prefill_adapter(
            torch.tensor([[2, 4, 4]], dtype=torch.long)
        )

        self.assertIsNot(first, second)
        self.assertEqual(
            runtime._vision_model.requests,
            [
                Qwen3VLVisionProfile(1, 4, 4),
                Qwen3VLVisionProfile(2, 4, 4),
            ],
        )
        self.assertEqual(len(runtime.vision_prefill_adapters), 2)

    def test_runtime_calls_static_adapter_with_pixels_only(self) -> None:
        """The simulated NPU stage should not receive ``image_grid_thw``."""
        runtime = self._make_runtime()
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        pixels = torch.randn(16, 8)

        image_embeds, deepstack = runtime._run_vision_prefill(pixels, grid)

        adapter = runtime._get_or_create_vision_prefill_adapter(grid)
        self.assertEqual(adapter.calls, 1)
        torch.testing.assert_close(image_embeds, pixels.mean(dim=-1))
        self.assertEqual(len(deepstack), 1)
        torch.testing.assert_close(deepstack[0], pixels.sum(dim=-1))

    def test_runtime_rejects_an_invalid_grid(self) -> None:
        """Profile validation should run before adapter construction."""
        runtime = self._make_runtime()

        with self.assertRaisesRegex(ValueError, "divisible"):
            runtime._get_or_create_vision_prefill_adapter(
                torch.tensor([[1, 5, 4]], dtype=torch.long)
            )
        self.assertEqual(runtime._vision_model.requests, [])


if __name__ == "__main__":
    unittest.main()
