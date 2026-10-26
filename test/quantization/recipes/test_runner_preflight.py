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

"""Exercise recipe preflight with the real stage registry and resolver."""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import contextlib
import io
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.stages as stages_mod
from tico.quantization.recipes.adapters import get_adapter
from tico.quantization.recipes.runner import (
    _requires_calibration_inputs,
    QuantizationRunner,
)
from tico.quantization.recipes.stages.base import Stage


class _MetadataGuard:
    @property
    def requires_calibration_inputs(self):
        raise AssertionError("Calibration requirement is already known")


class _RecordingStage(Stage):
    requires_calibration_inputs = False

    def __init__(self, name, events):
        self.name = name
        self.events = events

    def run(self, ctx, stage_cfg):
        self.events.append(self.name)
        return ctx


class TestRunnerPreflight(unittest.TestCase):
    @staticmethod
    def _config(pipeline, *, adapter_requires_inputs=False):
        return {
            "model": {"family": "llama", "name_or_path": "unused"},
            "runtime": {"print_config": False, "seed": None},
            "pipeline": pipeline,
            "export": {
                "enabled": adapter_requires_inputs,
                "artifacts": ["circle_full"] if adapter_requires_inputs else [],
            },
        }

    def test_invalid_stage_after_calibration_consumer(self):
        for invalid, error, message in (
            ({"name": "unknown_preflight"}, KeyError, "Unknown quantization stage"),
            (42, TypeError, "Each pipeline stage must be a mapping"),
            ({}, KeyError, "name"),
        ):
            with self.subTest(invalid=invalid):
                cfg = self._config([{"name": "gptq"}, invalid])
                with self.assertRaisesRegex(error, message):
                    _requires_calibration_inputs(cfg, get_adapter("llama"))

    def test_adapter_requirement_does_not_skip_stage_validation(self):
        cfg = self._config(
            [{"name": "unknown_preflight"}], adapter_requires_inputs=True
        )
        with self.assertRaisesRegex(KeyError, "Unknown quantization stage"):
            _requires_calibration_inputs(
                cfg, SimpleNamespace(requires_calibration_inputs=lambda _: True)
            )

    def test_requirements_accumulate_without_resolving_disabled_stages(self):
        for pipeline, adapter_requires_inputs, expected in (
            ([], False, False),
            ([], True, True),
            ([{"name": "cle"}, {"name": "spinquant"}], False, False),
            ([{"name": "gptq"}, {"name": "cle"}], False, True),
            ([{"name": "cle"}, {"name": "PTQ"}], False, True),
            ([{"name": "unknown_preflight", "enabled": False}], False, False),
            ([{"name": "unknown_preflight", "enabled": False}], True, True),
            ([{"name": "gptq", "enabled": False}], False, False),
        ):
            with self.subTest(pipeline=pipeline, adapter=adapter_requires_inputs):
                cfg = self._config(
                    pipeline, adapter_requires_inputs=adapter_requires_inputs
                )
                self.assertIs(
                    _requires_calibration_inputs(
                        cfg,
                        SimpleNamespace(
                            requires_calibration_inputs=lambda _: adapter_requires_inputs
                        ),
                    ),
                    expected,
                )

    def test_legacy_stage_without_metadata_still_requires_calibration(self):
        with patch.dict(
            stages_mod._STAGE_REGISTRY, {"legacy_preflight": SimpleNamespace()}
        ):
            for enabled in (True, False):
                with self.subTest(enabled=enabled):
                    cfg = self._config(
                        [
                            {"name": "legacy_preflight", "enabled": enabled},
                            {"name": "cle"},
                        ]
                    )
                    # A legacy duck-typed adapter may omit its requirement hook too.
                    self.assertIs(
                        _requires_calibration_inputs(cfg, SimpleNamespace()), enabled
                    )

    def test_known_requirement_does_not_read_later_stage_metadata(self):
        with patch.dict(
            stages_mod._STAGE_REGISTRY, {"guard_preflight": _MetadataGuard()}
        ):
            for adapter_requires_inputs in (False, True):
                with self.subTest(adapter=adapter_requires_inputs):
                    pipeline = [] if adapter_requires_inputs else [{"name": "gptq"}]
                    pipeline.append({"name": "guard_preflight"})
                    cfg = self._config(
                        pipeline, adapter_requires_inputs=adapter_requires_inputs
                    )
                    self.assertTrue(
                        _requires_calibration_inputs(
                            cfg,
                            SimpleNamespace(
                                requires_calibration_inputs=lambda _: adapter_requires_inputs
                            ),
                        )
                    )

    def test_adapter_requirement_is_called_once(self):
        calls = []

        def requirement(cfg):
            calls.append(cfg)
            return True

        cfg = self._config([{"name": "cle"}, {"name": "gptq"}])
        self.assertTrue(
            _requires_calibration_inputs(
                cfg, SimpleNamespace(requires_calibration_inputs=requirement)
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], cfg)

    def test_runner_rejects_invalid_pipeline_before_model_loading(self):
        adapter = get_adapter("llama")
        for pipeline, adapter_requires_inputs, error, message in (
            (
                [{"name": "gptq"}, {"name": "unknown_preflight"}],
                False,
                KeyError,
                "Unknown quantization stage",
            ),
            (
                [{"name": "unknown_preflight"}],
                True,
                KeyError,
                "Unknown quantization stage",
            ),
            (
                [{"name": "gptq"}, 42],
                False,
                TypeError,
                "Each pipeline stage must be a mapping",
            ),
        ):
            with self.subTest(pipeline=pipeline, adapter=adapter_requires_inputs):
                cfg = self._config(
                    pipeline, adapter_requires_inputs=adapter_requires_inputs
                )
                with (
                    patch.object(
                        adapter,
                        "load_model",
                        side_effect=AssertionError("Model loading reached"),
                    ) as load_model,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    with self.assertRaisesRegex(error, message):
                        QuantizationRunner().run(cfg)
                load_model.assert_not_called()

    def test_preflight_does_not_run_or_reorder_stages(self):
        events: list[str] = []

        def load_model(ctx):
            events.append("load")
            return ctx

        adapter = get_adapter("llama")
        cfg = self._config(
            [
                {"name": "first_preflight"},
                {"name": "unknown_preflight", "enabled": False},
                {"name": "second_preflight"},
                {"name": "first_preflight"},
            ]
        )
        with (
            patch.dict(
                stages_mod._STAGE_REGISTRY,
                {
                    name: _RecordingStage(name, events)
                    for name in ("first_preflight", "second_preflight")
                },
            ),
            patch.object(adapter, "load_model", side_effect=load_model),
            patch.object(adapter, "build_calibration_inputs") as build_inputs,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            QuantizationRunner().run(cfg)
        self.assertEqual(
            events, ["load", "first_preflight", "second_preflight", "first_preflight"]
        )
        build_inputs.assert_not_called()

    def test_extensions_load_before_stage_preflight(self):
        events: list[str] = []
        module_name = "tico_test_preflight_extension"
        extension = ModuleType(module_name)

        def activate():
            events.append("activate")
            stages_mod._STAGE_REGISTRY["extension_preflight"] = _RecordingStage(
                "run", events
            )

        def load_model(ctx):
            events.append("load")
            return ctx

        setattr(extension, "activate", activate)
        cfg = self._config([{"name": "cle"}, {"name": "extension_preflight"}])
        cfg["extensions"] = [f"{module_name}:activate"]
        with (
            patch.dict(sys.modules, {module_name: extension}),
            patch.dict(stages_mod._STAGE_REGISTRY),
            patch.object(
                get_adapter("llama"),
                "load_model",
                side_effect=load_model,
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            QuantizationRunner().run(cfg)
        self.assertEqual(events, ["activate", "load", "run"])
