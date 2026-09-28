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

"""Synthetic tests for out-of-tree adapter registration and selection."""

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import unittest
from unittest.mock import patch

import tico.quantization.recipes.adapters as adapters_mod
from tico.quantization.recipes.adapters import (
    available_adapters,
    get_adapter,
    register_adapter,
    resolve_adapter,
)
from tico.quantization.recipes.adapters.base import ModelAdapter
from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter
from tico.quantization.recipes.adapters.llama import LlamaAdapter


class _SyntheticAdapter(ModelAdapter):
    """Minimal concrete adapter for registry tests."""

    family = "synthetic"

    def load_model(self, ctx):
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
        return None

    def export(self, ctx):
        return None


class _Gemma4Variant(Gemma4Adapter):
    """Adapter variant that keeps the built-in family identifier."""


class TestAdapterRegistry(unittest.TestCase):
    def setUp(self):
        # Isolate registry mutations per test.
        self._registry_patch = patch.dict(adapters_mod._ADAPTERS, clear=False)
        self._registry_patch.start()

    def tearDown(self):
        self._registry_patch.stop()

    def test_register_and_lookup_normalizes_key(self):
        """A registered adapter should be found through normalized lookups."""
        adapter = _SyntheticAdapter()
        self.assertIs(register_adapter("  Synthetic_Ext ", adapter), adapter)
        self.assertIs(get_adapter("synthetic_ext"), adapter)
        self.assertIs(get_adapter("SYNTHETIC_EXT"), adapter)
        self.assertIn("synthetic_ext", available_adapters())

    def test_repeated_registration_of_same_adapter_is_noop(self):
        """Re-activating an extension must not fail or change the mapping."""
        adapter = _SyntheticAdapter()
        register_adapter("synthetic_ext", adapter)
        register_adapter("synthetic_ext", adapter)
        self.assertIs(get_adapter("synthetic_ext"), adapter)

    def test_conflicting_registration_is_rejected(self):
        """A different adapter under an occupied key must raise, not replace."""
        first = _SyntheticAdapter()
        register_adapter("synthetic_ext", first)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_adapter("synthetic_ext", _SyntheticAdapter())
        self.assertIs(get_adapter("synthetic_ext"), first)

    def test_builtin_keys_are_protected(self):
        """Built-in adapters cannot be overwritten by external registration."""
        builtin = get_adapter("gemma4")
        with self.assertRaisesRegex(ValueError, "'gemma4' is already registered"):
            register_adapter("gemma4", _Gemma4Variant())
        self.assertIs(get_adapter("gemma4"), builtin)
        self.assertIs(type(get_adapter("gemma4")), Gemma4Adapter)

    def test_register_rejects_invalid_inputs(self):
        """Non-adapter objects, empty keys, and missing families are rejected."""
        with self.assertRaisesRegex(TypeError, "ModelAdapter instance"):
            register_adapter("bad", object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            register_adapter("   ", _SyntheticAdapter())

        class _NoFamily(_SyntheticAdapter):
            family = ""

        with self.assertRaisesRegex(ValueError, "non-empty family"):
            register_adapter("nofamily", _NoFamily())

    def test_resolve_adapter_defaults_to_family(self):
        """Without model.adapter the built-in family adapter is selected."""
        adapter = resolve_adapter({"model": {"family": "llama"}})
        self.assertIsInstance(adapter, LlamaAdapter)
        self.assertIs(adapter, get_adapter("llama"))

    def test_resolve_adapter_selects_variant_with_matching_family(self):
        """model.adapter picks a registered variant while model.family stays."""
        variant = _Gemma4Variant()
        register_adapter("gemma4_variant", variant)
        cfg = {"model": {"family": "gemma4", "adapter": "gemma4_variant"}}
        self.assertIs(resolve_adapter(cfg), variant)
        # The built-in entry is untouched for unrelated workflows.
        self.assertIs(
            type(resolve_adapter({"model": {"family": "gemma4"}})), Gemma4Adapter
        )

    def test_resolve_adapter_accepts_family_alias(self):
        """Family aliases resolve to the same canonical family before comparison."""
        variant = _Gemma4Variant()
        variant.family = "qwen3_vl"  # type: ignore[misc]
        register_adapter("qwen_variant", variant)
        cfg = {"model": {"family": "qwen3-vl", "adapter": "qwen_variant"}}
        self.assertIs(resolve_adapter(cfg), variant)

    def test_resolve_adapter_rejects_family_mismatch(self):
        """A variant serving a different family must not be selected silently."""
        register_adapter("synthetic_ext", _SyntheticAdapter())
        cfg = {"model": {"family": "gemma4", "adapter": "synthetic_ext"}}
        with self.assertRaisesRegex(ValueError, "serves family 'synthetic'"):
            resolve_adapter(cfg)

    def test_resolve_adapter_unknown_key_and_missing_family(self):
        """Unknown adapter keys and missing model.family fail clearly."""
        with self.assertRaisesRegex(KeyError, "Unknown model family"):
            resolve_adapter({"model": {"family": "gemma4", "adapter": "missing"}})
        with self.assertRaisesRegex(KeyError, "model.family"):
            resolve_adapter({"model": {"adapter": "gemma4"}})
        with self.assertRaisesRegex(TypeError, "model must be a mapping"):
            resolve_adapter({"model": "gemma4"})


if __name__ == "__main__":
    unittest.main()
