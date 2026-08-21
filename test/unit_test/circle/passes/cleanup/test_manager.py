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

from tico.circle.passes import (
    CirclePass,
    CirclePassContext,
    CirclePassManager,
    CirclePassResult,
    CirclePassStrategy,
)

from test.unit_test.circle.fixture import make_test_document


class ChangeDescriptionOnce(CirclePass):
    """Change a fixture description exactly once for scheduler tests."""

    def run(self, document, context):
        if document.model.description == "fixture":
            document.model.description = "changed"
            return CirclePassResult(modified=True, changes=1)
        return CirclePassResult(modified=False)


class SetDescriptionTokenOnce(CirclePass):
    """Set one independent character in a synthetic scheduler state."""

    def __init__(self, token_index: int) -> None:
        self.token_index = int(token_index)

    def run(self, document, context):
        del context
        tokens = list(document.model.description)
        if tokens[self.token_index] == "1":
            return CirclePassResult(modified=False)
        tokens[self.token_index] = "1"
        document.model.description = "".join(tokens)
        return CirclePassResult(modified=True, changes=1)


class CirclePassManagerTest(unittest.TestCase):
    def test_until_no_change_reaches_a_fixed_point(self):
        document = make_test_document()
        manager = CirclePassManager(
            [ChangeDescriptionOnce()],
            strategy=CirclePassStrategy.UNTIL_NO_CHANGE,
        )

        result = manager.run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(result.executions), 2)
        self.assertEqual(document.model.description, "changed")

    def test_round_scheduling_matches_restart_with_fewer_pass_invocations(self):
        """Reach the same fixed point without triangular restart rescanning."""

        pass_count = 8
        restart_document = make_test_document()
        restart_document.model.description = "0" * pass_count
        round_document = restart_document.clone()
        restart_passes = [SetDescriptionTokenOnce(index) for index in range(pass_count)]
        round_passes = [SetDescriptionTokenOnce(index) for index in range(pass_count)]
        context = CirclePassContext(verify_after_each_pass=False)

        restart_result = CirclePassManager(
            restart_passes,
            strategy=CirclePassStrategy.RESTART,
        ).run(restart_document, context)
        round_result = CirclePassManager(
            round_passes,
            strategy=CirclePassStrategy.UNTIL_NO_CHANGE,
        ).run(round_document, CirclePassContext(verify_after_each_pass=False))

        self.assertEqual(restart_document.model.description, "1" * pass_count)
        self.assertEqual(
            round_document.model.description,
            restart_document.model.description,
        )
        self.assertEqual(round_result.changes, restart_result.changes)
        self.assertEqual(len(round_result.executions), pass_count * 2)
        self.assertLess(
            len(round_result.executions),
            len(restart_result.executions),
        )


if __name__ == "__main__":
    unittest.main()
