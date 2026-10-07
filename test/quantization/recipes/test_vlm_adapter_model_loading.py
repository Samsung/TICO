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

"""
Model loading contract of the Qwen3-VL and Gemma4 recipe adapters.

Both adapters load their model through ``AutoModelForImageTextToText``. These
tests inject failures at the processor/loader boundary and assert that the
first error reaches the caller unchanged, that no legacy auto class is retried,
and that the normal loading path forwards its arguments exactly once.

No Hub access, model download, or GPU is used; every external boundary is
replaced with a small recorder.
"""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import contextlib
import unittest
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, Iterator, TYPE_CHECKING
from unittest.mock import patch

import tico.quantization.recipes.adapters.gemma4 as gemma4_mod
import tico.quantization.recipes.adapters.qwen3_vl as qwen_mod

import torch
import transformers
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.qwen3_vl import Qwen3VLAdapter
from tico.quantization.recipes.context import RecipeContext

PRIMARY_LOADER = "AutoModelForImageTextToText"
LEGACY_LOADER = "AutoModelForVision2Seq"

MODEL_NAME = "org/fake-vlm"
HF_TOKEN = "hf_fake_token"
CACHE_DIR = "fake-cache-dir"


class _FakeAutoClass:
    """Record ``from_pretrained`` calls and return a result or raise an error."""

    def __init__(self, events: list[str], tag: str, result=None, error=None):
        self.events = events
        self.tag = tag
        self.result = result
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def from_pretrained(self, name, **kwargs):
        self.calls.append((name, kwargs))
        self.events.append(self.tag)
        if self.error is not None:
            raise self.error
        return self.result


class _FakeModel:
    """Minimal HF-like model exposing the config fields the adapters touch."""

    def __init__(
        self, events: list[str], eval_error=None, max_position_embeddings=4096
    ):
        self.events = events
        self.eval_error = eval_error
        text_config = SimpleNamespace(
            use_cache=True, max_position_embeddings=max_position_embeddings
        )
        self.config = SimpleNamespace(
            use_cache=True,
            text_config=text_config,
            vision_config=SimpleNamespace(),
            get_text_config=lambda: text_config,
        )

    def eval(self):
        self.events.append("eval")
        if self.eval_error is not None:
            raise self.eval_error
        return self


def _recipe_cfg(**runtime_overrides) -> dict[str, Any]:
    """Return a small recipe cfg; runtime fields may be overridden or removed."""
    runtime: dict[str, Any] = {"device": "cpu", "dtype": "bfloat16"}
    for key, value in runtime_overrides.items():
        if value is None:
            runtime.pop(key, None)
        else:
            runtime[key] = value
    return {
        "model": {
            "name_or_path": MODEL_NAME,
            "trust_remote_code": False,
            "hf_token": HF_TOKEN,
            "cache_dir": CACHE_DIR,
        },
        "runtime": runtime,
        "calibration": {"seq_len": 128},
        "model_args": {},
    }


@contextlib.contextmanager
def _without_attribute(module: ModuleType, name: str) -> Iterator[None]:
    """Temporarily remove ``module.name`` if the installed package exposes it."""
    try:
        original = getattr(module, name)
    except AttributeError:
        yield
        return
    delattr(module, name)
    try:
        yield
    finally:
        setattr(module, name, original)


if TYPE_CHECKING:
    # Give the mixin the TestCase API for type checking without letting
    # unittest collect the mixin itself as a test class.
    _MixinBase = unittest.TestCase
else:
    _MixinBase = object


class _ModelLoadingContractMixin(_MixinBase):
    """Shared ``load_model`` contract tests; subclasses bind one adapter."""

    adapter_cls: type
    adapter_module: ModuleType
    expected_post_load_events: list[str]

    # Hooks implemented by the concrete test classes -------------------------
    def expected_default_device_map(self, device: torch.device) -> str:
        raise NotImplementedError

    def post_load_patches(self, events: list[str]) -> list[Any]:
        """Return patchers replacing adapter-specific post-load boundaries."""
        return []

    # Helpers ---------------------------------------------------------------
    def _assert_raises_same(self, exc: BaseException):
        return _SameExceptionContext(self, exc)

    @contextlib.contextmanager
    def _loading_boundaries(
        self,
        events: list[str],
        *,
        primary: _FakeAutoClass,
        legacy: _FakeAutoClass | None,
        processor_error: BaseException | None = None,
    ) -> Iterator[_FakeAutoClass]:
        """Replace processor/loader classes and adapter post-load boundaries."""
        processor = _FakeAutoClass(
            events, "processor", result=SimpleNamespace(), error=processor_error
        )
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch.object(self.adapter_module, "AutoProcessor", processor)
            )
            stack.enter_context(patch.object(transformers, PRIMARY_LOADER, primary))
            if legacy is None:
                stack.enter_context(_without_attribute(transformers, LEGACY_LOADER))
            else:
                stack.enter_context(
                    patch.object(transformers, LEGACY_LOADER, legacy, create=True)
                )
            for patcher in self.post_load_patches(events):
                stack.enter_context(patcher)
            yield processor

    @staticmethod
    def _legacy_recorder(events: list[str]) -> _FakeAutoClass:
        """Return a legacy loader that would succeed if it were (wrongly) retried."""
        return _FakeAutoClass(events, "legacy", result=_FakeModel(events))

    # A. first loader error must reach the caller unchanged -----------------
    def test_primary_loader_failure_propagates_without_legacy_retry(self):
        """The injected from_pretrained error is raised as-is; no retry happens."""
        errors: list[BaseException] = [
            RuntimeError("synthetic loader failure"),
            MemoryError("synthetic host OOM"),
            torch.OutOfMemoryError("synthetic CUDA OOM without allocation"),
            OSError("synthetic missing repository"),
            ValueError("synthetic unrecognized config"),
            ImportError("synthetic missing optional dependency inside loader"),
        ]
        for error in errors:
            for install_legacy in (True, False):
                with self.subTest(
                    error=type(error).__name__, legacy_loader_installed=install_legacy
                ):
                    events: list[str] = []
                    primary = _FakeAutoClass(events, "primary", error=error)
                    legacy = self._legacy_recorder(events) if install_legacy else None
                    ctx = RecipeContext(cfg=_recipe_cfg(), adapter=self.adapter_cls())

                    with self._loading_boundaries(
                        events, primary=primary, legacy=legacy
                    ), self._assert_raises_same(error):
                        self.adapter_cls().load_model(ctx)

                    self.assertEqual(len(primary.calls), 1)
                    if legacy is not None:
                        self.assertEqual(legacy.calls, [])
                    self.assertIsNone(ctx.model)
                    self.assertEqual(events, ["processor", "primary"])

    # B. normal loading contract ---------------------------------------------
    def test_successful_load_forwards_arguments_once(self):
        """Processor then loader run once with the configured arguments."""
        for seq_len in (128, None):
            with self.subTest(calibration_seq_len=seq_len):
                events: list[str] = []
                model = _FakeModel(events)
                primary = _FakeAutoClass(events, "primary", result=model)
                legacy = self._legacy_recorder(events)
                cfg = _recipe_cfg()
                if seq_len is None:
                    del cfg["calibration"]["seq_len"]
                ctx = RecipeContext(cfg=cfg, adapter=self.adapter_cls())

                with self._loading_boundaries(
                    events, primary=primary, legacy=legacy
                ) as processor:
                    result = self.adapter_cls().load_model(ctx)

                self.assertIs(result, ctx)
                self.assertIs(ctx.model, model)
                self.assertIs(ctx.processor, processor.result)
                self.assertEqual(ctx.device, torch.device("cpu"))
                self.assertEqual(ctx.dtype, torch.bfloat16)
                self.assertEqual(
                    processor.calls,
                    [
                        (
                            MODEL_NAME,
                            {
                                "trust_remote_code": False,
                                "token": HF_TOKEN,
                                "cache_dir": CACHE_DIR,
                            },
                        )
                    ],
                )
                self.assertEqual(
                    primary.calls,
                    [
                        (
                            MODEL_NAME,
                            {
                                "dtype": torch.bfloat16,
                                "trust_remote_code": False,
                                "token": HF_TOKEN,
                                "cache_dir": CACHE_DIR,
                                "device_map": "cpu",
                            },
                        )
                    ],
                )
                self.assertEqual(legacy.calls, [])
                self.assertEqual(
                    events, ["processor", "primary", *self.expected_post_load_events]
                )
                self.assertFalse(model.config.use_cache)
                self.assertFalse(model.config.text_config.use_cache)
                expected_positions = 4096 if seq_len is None else seq_len
                self.assertEqual(
                    model.config.text_config.max_position_embeddings,
                    expected_positions,
                )

    def test_device_map_defaults_and_explicit_override(self):
        """Default device_map follows the adapter policy; overrides pass through."""
        explicit_map = {"": 0}
        cases: list[tuple[str, dict[str, Any], bool, Callable[[torch.device], Any]]] = [
            ("cpu_default", {}, False, lambda device: "cpu"),
            (
                "explicit_cuda_device_default_map",
                {"device": "cuda:1"},
                False,
                self.expected_default_device_map,
            ),
            (
                "auto_detected_cuda_default_map",
                {"device": None},
                True,
                self.expected_default_device_map,
            ),
            (
                "explicit_device_map_override",
                {"device": "cuda:1", "device_map": explicit_map},
                False,
                lambda device: explicit_map,
            ),
        ]
        for name, runtime, cuda_available, expected_map in cases:
            with self.subTest(case=name):
                events: list[str] = []
                model = _FakeModel(events)
                primary = _FakeAutoClass(events, "primary", result=model)
                ctx = RecipeContext(
                    cfg=_recipe_cfg(**runtime), adapter=self.adapter_cls()
                )

                with self._loading_boundaries(
                    events, primary=primary, legacy=None
                ), patch.object(torch.cuda, "is_available", lambda: cuda_available):
                    self.adapter_cls().load_model(ctx)

                expected_device = torch.device(
                    runtime.get("device") or ("cuda" if cuda_available else "cpu")
                )
                self.assertEqual(ctx.device, expected_device)
                self.assertEqual(len(primary.calls), 1)
                passed_map = primary.calls[0][1]["device_map"]
                expected = expected_map(expected_device)
                self.assertEqual(passed_map, expected)
                if name == "explicit_device_map_override":
                    self.assertIs(passed_map, explicit_map)

    # C. other failure boundaries ------------------------------------------
    def test_processor_failure_skips_model_loader(self):
        """A processor error propagates before any model loader is called."""
        error = OSError("synthetic processor failure")
        events: list[str] = []
        primary = _FakeAutoClass(events, "primary", result=_FakeModel(events))
        legacy = self._legacy_recorder(events)
        ctx = RecipeContext(cfg=_recipe_cfg(), adapter=self.adapter_cls())

        with self._loading_boundaries(
            events, primary=primary, legacy=legacy, processor_error=error
        ), self._assert_raises_same(error):
            self.adapter_cls().load_model(ctx)

        self.assertEqual(primary.calls, [])
        self.assertEqual(legacy.calls, [])
        self.assertIsNone(ctx.model)
        self.assertEqual(events, ["processor"])

    def test_post_load_failure_does_not_retry_legacy_loader(self):
        """An error after a successful load is not hidden by another loader."""
        error = RuntimeError("synthetic post-load failure")
        events: list[str] = []
        model = _FakeModel(events, eval_error=error)
        primary = _FakeAutoClass(events, "primary", result=model)
        legacy = self._legacy_recorder(events)
        ctx = RecipeContext(cfg=_recipe_cfg(), adapter=self.adapter_cls())

        with self._loading_boundaries(
            events, primary=primary, legacy=legacy
        ), self._assert_raises_same(error):
            self.adapter_cls().load_model(ctx)

        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(legacy.calls, [])
        self.assertEqual(events, ["processor", "primary", "eval"])


class _SameExceptionContext:
    """Assert that a block raises exactly the given exception object."""

    def __init__(self, case: unittest.TestCase, expected: BaseException):
        self.case = case
        self.expected = expected

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is None:
            self.case.fail(
                f"{type(self.expected).__name__} was not raised; "
                "load_model returned instead of propagating the loader error"
            )
        self.case.assertIs(exc, self.expected)
        self.case.assertIsNotNone(exc.__traceback__)
        # A re-raise through another loader or a wrapper would attach
        # context/cause; the first error must stay unchained.
        self.case.assertIsNone(exc.__context__)
        self.case.assertIsNone(exc.__cause__)
        return True


class TestQwen3VLAdapterModelLoading(_ModelLoadingContractMixin, unittest.TestCase):
    """Qwen3VLAdapter.load_model loading and failure contract."""

    adapter_cls = Qwen3VLAdapter
    adapter_module = qwen_mod
    expected_post_load_events = ["eval"]

    def expected_default_device_map(self, device: torch.device) -> str:
        return "auto" if device.type != "cpu" else "cpu"


class TestGemma4AdapterModelLoading(_ModelLoadingContractMixin, unittest.TestCase):
    """Gemma4Adapter.load_model loading and failure contract."""

    adapter_cls = Gemma4Adapter
    adapter_module = gemma4_mod
    expected_post_load_events = ["eval", "no_moe", "ple_fusion"]

    def expected_default_device_map(self, device: torch.device) -> str:
        return "cpu" if device.type == "cpu" else str(device)

    def post_load_patches(self, events: list[str]) -> list[Any]:
        def fake_assert_no_moe(model):
            events.append("no_moe")

        def fake_fuse(model):
            events.append("ple_fusion")
            return ["fused"]

        return [
            patch.object(gemma4_mod, "assert_gemma4_e2b_no_moe", fake_assert_no_moe),
            patch.object(gemma4_mod, "fuse_gemma4_ple_embedding_scale", fake_fuse),
        ]

    def test_no_moe_failure_does_not_retry_legacy_loader(self):
        """A failing architecture check propagates without a second loader."""
        error = ValueError("synthetic MoE architecture rejected")
        events: list[str] = []
        model = _FakeModel(events)
        primary = _FakeAutoClass(events, "primary", result=model)
        legacy = self._legacy_recorder(events)
        ctx = RecipeContext(cfg=_recipe_cfg(), adapter=Gemma4Adapter())

        def failing_assert_no_moe(model):
            events.append("no_moe")
            raise error

        with self._loading_boundaries(
            events, primary=primary, legacy=legacy
        ), patch.object(
            gemma4_mod, "assert_gemma4_e2b_no_moe", failing_assert_no_moe
        ), self._assert_raises_same(
            error
        ):
            Gemma4Adapter().load_model(ctx)

        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(legacy.calls, [])
        self.assertEqual(events, ["processor", "primary", "eval", "no_moe"])
        self.assertNotIn("gemma4_ple_scale_fused_modules", ctx.artifacts)

    def test_ple_fusion_setting_and_static_profile_order_are_preserved(self):
        """PLE fusion honors its setting and precedes static profile handling."""
        for fuse_enabled in (True, False):
            with self.subTest(ple_embedding_scale_fusion=fuse_enabled):
                events: list[str] = []
                model = _FakeModel(events)
                primary = _FakeAutoClass(events, "primary", result=model)
                cfg = _recipe_cfg()
                cfg["model_args"] = {
                    "text": {"ple_embedding_scale_fusion": fuse_enabled},
                    "vision": {"profile": "fake_profile"},
                }
                original_model_args = cfg["model_args"]
                normalized_model_args = {"normalized": True}
                captured: dict[str, Any] = {}

                def fake_validate_processor(processor):
                    events.append("validate_processor")
                    captured["validated_processor"] = processor

                profile = SimpleNamespace(validate_processor=fake_validate_processor)

                def fake_canonicalize(model_args):
                    events.append("canonicalize")
                    captured["canonicalize_input"] = model_args
                    return normalized_model_args

                def fake_build_profile(model_args, *, vision_config, max_seq_len):
                    events.append("build_profile")
                    captured["profile_args"] = (model_args, vision_config, max_seq_len)
                    return profile

                ctx = RecipeContext(cfg=cfg, adapter=Gemma4Adapter())
                with self._loading_boundaries(
                    events, primary=primary, legacy=None
                ), patch.object(
                    gemma4_mod,
                    "canonicalize_gemma4_static_vision_model_args",
                    fake_canonicalize,
                ), patch.object(
                    gemma4_mod,
                    "build_gemma4_static_vision_profile",
                    fake_build_profile,
                ):
                    Gemma4Adapter().load_model(ctx)

                expected_events = ["processor", "primary", "eval", "no_moe"]
                if fuse_enabled:
                    expected_events.append("ple_fusion")
                expected_events += [
                    "canonicalize",
                    "build_profile",
                    "validate_processor",
                ]
                self.assertEqual(events, expected_events)
                self.assertEqual(len(primary.calls), 1)

                if fuse_enabled:
                    self.assertEqual(
                        ctx.artifacts["gemma4_ple_scale_fused_modules"], ["fused"]
                    )
                else:
                    self.assertNotIn("gemma4_ple_scale_fused_modules", ctx.artifacts)

                self.assertIs(captured["canonicalize_input"], original_model_args)
                self.assertEqual(
                    captured["profile_args"],
                    (normalized_model_args, model.config.vision_config, 128),
                )
                self.assertIs(captured["validated_processor"], ctx.processor)
                self.assertIs(ctx.cfg["model_args"], normalized_model_args)
                self.assertIs(ctx.artifacts["gemma4_static_vision_profile"], profile)


if __name__ == "__main__":
    unittest.main()
