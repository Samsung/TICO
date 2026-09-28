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

"""Synthetic tests for config-driven recipe extension loading."""

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
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import tico.quantization.recipes.adapters as adapters_mod
import tico.quantization.recipes.runner as runner_mod
from tico.quantization.recipes.adapters import get_adapter, register_adapter
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.extensions import (
    load_extension,
    load_recipe_extensions,
    parse_extension_entry,
)
from tico.quantization.recipes.runner import QuantizationRunner

_MODULE_NAME = "tico_test_synthetic_extension"


class _ExtensionAdapter(ModelAdapter):
    """Adapter registered by the synthetic extension module."""

    family = "synthetic_ext"

    def __init__(self, events):
        self.events = events

    def load_model(self, ctx):
        self.events.append("load")
        ctx.model = SimpleNamespace(name="model")
        return ctx

    def build_calibration_inputs(self, ctx):
        return []

    def forward_calibration(self, ctx, model, calibration_inputs, *, desc):
        return None

    def calibrate_prepared_model(self, ctx, prepared_model, stage_cfg):
        return None

    def build_ptq_config(self, ctx, stage_cfg):
        return None

    def evaluate(self, ctx):
        self.events.append("evaluate")

    def export(self, ctx):
        self.events.append("export")


def _install_synthetic_module(events):
    """Install an in-memory extension module exposing ``activate()``."""
    module = types.ModuleType(_MODULE_NAME)
    adapter = _ExtensionAdapter(events)

    def activate():
        events.append("activate")
        register_adapter("synthetic_ext", adapter)
        return adapter

    module.activate = activate  # type: ignore[attr-defined]
    module.adapter = adapter  # type: ignore[attr-defined]
    module.not_callable = 1  # type: ignore[attr-defined]
    sys.modules[_MODULE_NAME] = module
    return module


class TestExtensionEntries(unittest.TestCase):
    def test_parse_entry_forms(self):
        """Both module-only and module:callable forms are accepted."""
        self.assertEqual(parse_extension_entry("pkg.mod"), ("pkg.mod", None))
        self.assertEqual(parse_extension_entry(" pkg.mod : run "), ("pkg.mod", "run"))

    def test_parse_entry_rejects_malformed_values(self):
        """Malformed entries fail with explicit messages."""
        with self.assertRaisesRegex(TypeError, "must be strings"):
            parse_extension_entry(123)
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            parse_extension_entry("   ")
        with self.assertRaisesRegex(ValueError, "missing module"):
            parse_extension_entry(":run")
        with self.assertRaisesRegex(ValueError, "missing callable"):
            parse_extension_entry("pkg.mod:")
        with self.assertRaisesRegex(ValueError, "at most one ':'"):
            parse_extension_entry("pkg.mod:a:b")


class TestLoadRecipeExtensions(unittest.TestCase):
    def setUp(self):
        self.events: list = []
        self._registry_patch = patch.dict(adapters_mod._ADAPTERS, clear=False)
        self._registry_patch.start()
        _install_synthetic_module(self.events)

    def tearDown(self):
        sys.modules.pop(_MODULE_NAME, None)
        self._registry_patch.stop()

    def test_absent_key_loads_nothing(self):
        """Configs without ``extensions`` never import or activate anything."""
        self.assertEqual(load_recipe_extensions({"model": {"family": "llama"}}), [])
        self.assertEqual(self.events, [])

    def test_module_only_entry_imports_without_calling(self):
        """A module entry imports the module and returns it."""
        module = load_extension(_MODULE_NAME)
        self.assertIs(module, sys.modules[_MODULE_NAME])
        self.assertEqual(self.events, [])

    def test_callable_entry_registers_before_lookup(self):
        """The callable runs, and its registration is visible afterwards."""
        loaded = load_recipe_extensions({"extensions": [f"{_MODULE_NAME}:activate"]})
        self.assertEqual(loaded, [f"{_MODULE_NAME}:activate"])
        self.assertEqual(self.events, ["activate"])
        self.assertIs(get_adapter("synthetic_ext"), sys.modules[_MODULE_NAME].adapter)

    def test_repeated_loading_is_safe(self):
        """Loading the same extension twice must not raise or duplicate state."""
        cfg = {"extensions": [f"{_MODULE_NAME}:activate"]}
        load_recipe_extensions(cfg)
        load_recipe_extensions(cfg)
        self.assertEqual(self.events, ["activate", "activate"])
        self.assertIs(get_adapter("synthetic_ext"), sys.modules[_MODULE_NAME].adapter)

    def test_missing_module_fails_clearly(self):
        """Unavailable extensions raise instead of being skipped."""
        with self.assertRaisesRegex(RuntimeError, "Failed to import extensions entry"):
            load_recipe_extensions({"extensions": ["tico_test_missing_extension"]})

    def test_missing_callable_and_non_callable_fail_clearly(self):
        """Wrong attribute names and non-callable targets are reported."""
        with self.assertRaisesRegex(RuntimeError, "does not exist in module"):
            load_extension(f"{_MODULE_NAME}:nope")
        with self.assertRaisesRegex(TypeError, "must name a callable"):
            load_extension(f"{_MODULE_NAME}:not_callable")

    def test_invalid_container_types_are_rejected(self):
        """``extensions`` must be a list of strings."""
        with self.assertRaisesRegex(TypeError, "must be a list"):
            load_recipe_extensions({"extensions": f"{_MODULE_NAME}:activate"})
        with self.assertRaisesRegex(TypeError, "must be strings"):
            load_recipe_extensions({"extensions": [42]})

    def test_extension_errors_propagate(self):
        """Exceptions raised inside the extension callable are not swallowed."""

        def failing():
            raise ImportError("optional dependency missing")

        sys.modules[_MODULE_NAME].failing = failing  # type: ignore[attr-defined]
        with self.assertRaisesRegex(ImportError, "optional dependency missing"):
            load_extension(f"{_MODULE_NAME}:failing")

    def test_runner_loads_extensions_before_adapter_resolution(self):
        """The runner activates listed extensions, then selects the adapter."""
        cfg = {
            "extensions": [f"{_MODULE_NAME}:activate"],
            "model": {"family": "synthetic_ext", "name_or_path": "synthetic"},
            "runtime": {"seed": 1, "print_config": False},
            "pipeline": [],
        }
        with patch.object(runner_mod, "set_seed", lambda seed: None), patch.object(
            runner_mod, "validate_recipe_dataset_usage", lambda cfg, **kw: None
        ), contextlib.redirect_stdout(io.StringIO()):
            ctx = QuantizationRunner().run(cfg)
        self.assertEqual(self.events, ["activate", "load", "evaluate", "export"])
        self.assertIs(ctx.adapter, sys.modules[_MODULE_NAME].adapter)

    def test_runner_without_extension_cannot_find_external_adapter(self):
        """A previous process's import is not activation: nothing is registered."""
        cfg = {
            "model": {"family": "synthetic_ext", "name_or_path": "synthetic"},
            "runtime": {"print_config": False},
            "pipeline": [],
        }
        with patch.object(
            runner_mod, "set_seed", lambda seed: None
        ), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            KeyError, "Unknown model family"
        ):
            QuantizationRunner().run(cfg)


if __name__ == "__main__":
    unittest.main()
