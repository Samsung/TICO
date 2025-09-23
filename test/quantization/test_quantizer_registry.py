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

import sys
import types
import unittest
from unittest.mock import patch

from tico.experimental.quantization.config.base import BaseConfig
from tico.experimental.quantization.quantizer import BaseQuantizer

from tico.experimental.quantization.quantizer_registry import (
    get_quantizer,
    register_quantizer,
)


# ---------- Helper classes used only in tests ----------
class _ExactConfig(BaseConfig):
    @property
    def name(self) -> str:
        return "exact_algo"


class _LazyConfig(BaseConfig):
    def __init__(self, algo_name: str):
        self._name = algo_name

    @property
    def name(self) -> str:
        return self._name


class _ImportFailConfig(BaseConfig):
    def __init__(self, algo_name: str):
        self._name = algo_name

    @property
    def name(self) -> str:
        return self._name


class _UnregisteredImportConfig(BaseConfig):
    def __init__(self, algo_name: str):
        self._name = algo_name

    @property
    def name(self) -> str:
        return self._name


class _DummyQuantizer(BaseQuantizer):
    def prepare(self, model, args=None, kwargs=None):
        return model

    def convert(self, model):
        return model


class _AnotherDummyQuantizer(BaseQuantizer):
    def prepare(self, model, args=None, kwargs=None):
        return model

    def convert(self, model):
        return model


def _install_fake_quantizer_module(
    module_name: str, config_cls: type, quantizer_cls: type
):
    """
    Create an in-memory module whose import triggers registry registration.

    The module will call:
        register_quantizer(config_cls)(quantizer_cls)
    so that get_quantizer() can find the mapping after the lazy import.
    """
    mod = types.ModuleType(module_name)
    # pylint: disable=no-member
    # Inject the decorator and classes it needs into the module namespace
    mod.register_quantizer = register_quantizer  # type: ignore[attr-defined]
    mod.config_cls = config_cls  # type: ignore[attr-defined]
    mod.quantizer_cls = quantizer_cls  # type: ignore[attr-defined]

    # When this module is imported, it registers the quantizer for the config
    def _auto_register():
        mod.register_quantizer(mod.config_cls)(mod.quantizer_cls)

    mod._auto_register = _auto_register  # type: ignore[attr-defined]
    # Execute the registration immediately upon import
    mod._auto_register()
    # pylint: enable=no-member
    # Install into sys.modules so importlib.import_module can find it
    sys.modules[module_name] = mod
    return mod


class QuantizerRegistryTest(unittest.TestCase):
    def setUp(self):
        """
        Ensure a clean slate for sys.modules entries we might inject.
        Tests that create fake modules will add them and should remove in tearDown.
        """
        self._added_modules = []

    def tearDown(self):
        """Remove any fake modules we injected during a test."""
        for name in self._added_modules:
            sys.modules.pop(name, None)

    def test_register_and_lookup_exact_type(self):
        """Decorator registration should resolve by exact config type."""

        @register_quantizer(_ExactConfig)
        class _ExactQuant(_DummyQuantizer):
            pass

        cfg = _ExactConfig()
        q = get_quantizer(cfg)
        self.assertIsInstance(q, _ExactQuant)  # exact mapping should be used

    def test_lazy_import_by_naming_convention_success(self):
        """
        get_quantizer should import:
          tico.experimental.quantization.algorithm.{name}.quantizer
        and find the registered quantizer afterward.
        """
        algo_name = "fakealgo"
        cfg = _LazyConfig(algo_name)

        # Install a fake module that registers upon import
        module_name = f"tico.experimental.quantization.algorithm.{algo_name}.quantizer"

        _install_fake_quantizer_module(
            module_name, config_cls=_LazyConfig, quantizer_cls=_AnotherDummyQuantizer
        )
        self._added_modules.append(module_name)

        q = get_quantizer(cfg)
        self.assertIsInstance(q, _AnotherDummyQuantizer)

    def test_lazy_import_raises_meaningful_error_on_failure(self):
        """
        If the naming-convention import fails outright, get_quantizer should raise
        a RuntimeError with a helpful message.
        """
        cfg = _ImportFailConfig("does_not_exist")

        with self.assertRaises(RuntimeError) as ctx:
            _ = get_quantizer(cfg)

        msg = str(ctx.exception)
        self.assertIn("Failed to import quantizer module", msg)
        self.assertIn("does_not_exist", msg)

    def test_not_found_after_successful_import_raises(self):
        """
        Even if the module is importable, if it does NOT call register_quantizer,
        get_quantizer should error with a clear message.
        """
        algo_name = "importable_but_unregistered"
        cfg = _UnregisteredImportConfig(algo_name)
        module_name = f"tico.experimental.quantization.algorithm.{algo_name}.quantizer"

        # Create a module that does NOT register anything
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod
        self._added_modules.append(module_name)

        with self.assertRaises(RuntimeError) as ctx:
            _ = get_quantizer(cfg)

        msg = str(ctx.exception)
        self.assertIn("No quantizer registered", msg)
        self.assertIn(type(cfg).__name__, msg)

    def test_get_quantizer_returns_instance_not_class(self):
        """Factory must instantiate the quantizer with the provided config."""

        @register_quantizer(_ExactConfig)
        class _ExactQuant2(_DummyQuantizer):
            pass

        cfg = _ExactConfig()
        q = get_quantizer(cfg)
        self.assertIsInstance(q, _ExactQuant2)
        # Ensure the instance carries the same config reference
        self.assertIs(q.config, cfg)
