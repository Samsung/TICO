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

import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from tico.circle import _schema
from tico.circle.passes.optimization.compatibility import custom_ops
from tico.circle.passes.optimization.fold.evaluators import heavy
from tico.circle.passes.optimization.fuse import spatial
from tico.circle.passes.optimization.legalize import dynamic_fully_connected


class SchemaEnumValuesTest(unittest.TestCase):
    """Characterize schema layout compatibility and explicit legacy fallbacks."""

    MODULES = (heavy, spatial, dynamic_fully_connected, custom_ops)
    FALLBACK_MODULES = (heavy, spatial, dynamic_fully_connected)

    def setUp(self):
        """Keep synthetic schema packages from leaking into other tests."""
        _schema.circle_schema.cache_clear()
        self.addCleanup(_schema.circle_schema.cache_clear)

    def test_nested_and_flat_enum_layouts_keep_integer_conversion(self):
        for nested in (False, True):
            enum_type = SimpleNamespace(SAME="0", VALID=1)
            enum_module = SimpleNamespace(Padding=enum_type) if nested else enum_type
            package = SimpleNamespace(circle=SimpleNamespace(Padding=enum_module))
            with patch.object(_schema, "import_module", return_value=package):
                for module in self.MODULES:
                    for name, expected in (("SAME", 0), ("VALID", 1)):
                        with self.subTest(nested=nested, module=module, name=name):
                            value = module._schema_enum_value("Padding", name)
                            self.assertEqual(value, expected)
                            self.assertIs(type(value), int)
            _schema.circle_schema.cache_clear()

    def test_nested_enum_takes_precedence_over_module_members(self):
        package = SimpleNamespace(
            circle=SimpleNamespace(
                Padding=SimpleNamespace(Padding=SimpleNamespace(SAME=0), VALID=1)
            )
        )
        with patch.object(_schema, "import_module", return_value=package):
            for module in self.MODULES:
                with (
                    self.subTest(module=module),
                    self.assertRaisesRegex(
                        RuntimeError,
                        r"^Circle schema does not provide Padding\.VALID\.$",
                    ),
                ):
                    module._schema_enum_value("Padding", "VALID")

    def test_missing_enum_and_member_keep_exact_diagnostics(self):
        for schema in (
            SimpleNamespace(),
            SimpleNamespace(Padding=None),
            SimpleNamespace(Padding=SimpleNamespace()),
        ):
            with patch.object(
                _schema, "import_module", return_value=SimpleNamespace(circle=schema)
            ):
                for module in self.MODULES:
                    with self.subTest(schema=schema, module=module):
                        with self.assertRaises(RuntimeError) as raised:
                            module._schema_enum_value("Padding", "SAME")
                        self.assertEqual(
                            str(raised.exception),
                            "Circle schema does not provide Padding.SAME.",
                        )
            _schema.circle_schema.cache_clear()

    def test_strict_lookup_preserves_schema_loading_errors(self):
        error_types: tuple[type[Exception], ...] = (
            AttributeError,
            ImportError,
            RuntimeError,
        )
        for error_type in error_types:
            failure = error_type("schema unavailable")
            with patch.object(_schema, "import_module", side_effect=failure):
                for module in self.MODULES:
                    with self.subTest(module=module, error_type=error_type):
                        with self.assertRaises(error_type) as raised:
                            module._schema_enum_value("Padding", "SAME")
                        self.assertIs(raised.exception, failure)

    def test_fallback_uses_available_enum_instead_of_legacy_value(self):
        package = SimpleNamespace(
            circle=SimpleNamespace(Padding=SimpleNamespace(SAME=0))
        )
        with patch.object(_schema, "import_module", return_value=package):
            for module in self.FALLBACK_MODULES:
                with self.subTest(module=module):
                    self.assertEqual(
                        module._maybe_schema_enum_value("Padding", "SAME", 7), 0
                    )

    def test_missing_enum_and_member_use_integer_legacy_fallback(self):
        fallback: Any = "7"
        for schema in (SimpleNamespace(), SimpleNamespace(Padding=SimpleNamespace())):
            with patch.object(
                _schema, "import_module", return_value=SimpleNamespace(circle=schema)
            ):
                for module in self.FALLBACK_MODULES:
                    with self.subTest(schema=schema, module=module):
                        value = module._maybe_schema_enum_value(
                            "Padding", "SAME", fallback
                        )
                        self.assertEqual(value, 7)
                        self.assertIs(type(value), int)
            _schema.circle_schema.cache_clear()

    def test_only_original_schema_errors_select_fallback(self):
        error_types: tuple[type[Exception], ...] = (
            AttributeError,
            ImportError,
            RuntimeError,
            ValueError,
            TypeError,
        )
        for error_type in error_types:
            failure = error_type("schema unavailable")
            with patch.object(_schema, "import_module", side_effect=failure):
                for module in self.FALLBACK_MODULES:
                    with self.subTest(module=module, error_type=error_type):
                        if error_type in (AttributeError, ImportError, RuntimeError):
                            self.assertEqual(
                                module._maybe_schema_enum_value("Padding", "SAME", 7), 7
                            )
                        else:
                            with self.assertRaises(error_type) as raised:
                                module._maybe_schema_enum_value("Padding", "SAME", 7)
                            self.assertIs(raised.exception, failure)

    def test_invalid_enum_value_is_not_hidden_by_fallback(self):
        for value, error_type in (("invalid", ValueError), (None, TypeError)):
            package = SimpleNamespace(
                circle=SimpleNamespace(Padding=SimpleNamespace(SAME=value))
            )
            with patch.object(_schema, "import_module", return_value=package):
                for module in self.MODULES:
                    with self.subTest(module=module, value=value):
                        with self.assertRaises(error_type):
                            module._schema_enum_value("Padding", "SAME")
                        if module in self.FALLBACK_MODULES:
                            with self.assertRaises(error_type):
                                module._maybe_schema_enum_value("Padding", "SAME", 7)
            _schema.circle_schema.cache_clear()

    def test_invalid_fallback_value_is_not_hidden(self):
        fallback: Any = "invalid"
        package = SimpleNamespace(circle=SimpleNamespace())
        with patch.object(_schema, "import_module", return_value=package):
            for module in self.FALLBACK_MODULES:
                with self.subTest(module=module), self.assertRaises(ValueError):
                    module._maybe_schema_enum_value("Padding", "SAME", fallback)

    def test_installed_schema_values_are_preserved(self):
        from circle_schema import circle

        for module in self.MODULES:
            with self.subTest(module=module):
                self.assertEqual(
                    module._schema_enum_value("Padding", "SAME"),
                    circle.Padding.Padding.SAME,
                )
                self.assertEqual(
                    module._schema_enum_value("BuiltinOperator", "ADD"),
                    circle.BuiltinOperator.BuiltinOperator.ADD,
                )


if __name__ == "__main__":
    unittest.main()
