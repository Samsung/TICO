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

from __future__ import annotations

import unittest
from types import SimpleNamespace

from tico.circle.passes.optimization.simplify import transpose_region_rules as rules


class TransposeBoundedLayoutRegionRuleRegistryTest(unittest.TestCase):
    """Verify rule lookup without changing bounded-region functionality."""

    def test_add_rule_metadata(self) -> None:
        """Check that ADD resolves to the same-shape binary rule."""

        rule = rules._rule_for_builtin_code(rules._ADD_BUILTIN_CODE)
        self.assertIsInstance(rule, rules._SameShapeBinaryElementwiseRule)
        self.assertEqual(rule.data_input_positions(None), (0, 1))
        self.assertEqual(rule.data_output_positions(None), (0,))

    def test_pad_rule_metadata(self) -> None:
        """Check that PAD resolves to the constant-remapping rule."""

        rule = rules._rule_for_builtin_code(rules._PAD_BUILTIN_CODE)
        self.assertIsInstance(rule, rules._PadRule)
        self.assertEqual(rule.data_input_positions(None), (0,))
        self.assertEqual(rule.data_output_positions(None), (0,))

    def test_operator_lookup_uses_opcode_table(self) -> None:
        """Resolve a registered rule through an operator opcode index."""

        operator = SimpleNamespace(opcodeIndex=1)
        operator_codes = [
            SimpleNamespace(builtinCode=-1),
            SimpleNamespace(builtinCode=rules._PAD_BUILTIN_CODE),
        ]
        self.assertIsInstance(
            rules._rule_for_operator(operator, operator_codes),
            rules._PadRule,
        )

    def test_unsupported_builtin_has_no_rule(self) -> None:
        """Keep unregistered Circle builtins outside the region pass."""

        self.assertIsNone(rules._rule_for_builtin_code(1_000_000))

    def test_invalid_opcode_index_has_no_rule(self) -> None:
        """Reject an operator whose opcode index is outside the table."""

        operator = SimpleNamespace(opcodeIndex=3)
        operator_codes = [SimpleNamespace(builtinCode=rules._ADD_BUILTIN_CODE)]
        self.assertIsNone(rules._rule_for_operator(operator, operator_codes))


if __name__ == "__main__":
    unittest.main()
