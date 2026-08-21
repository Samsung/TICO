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

from tico.circle.passes.worklist import CircleRuleWorkItem, CircleRuleWorklist


class CircleRuleWorklistTest(unittest.TestCase):
    """Check deterministic local scheduling and final validation refills."""

    def test_initial_scan_uses_subgraph_order(self) -> None:
        worklist = CircleRuleWorklist(3)

        self.assertEqual(
            [worklist.pop(), worklist.pop(), worklist.pop()],
            [
                CircleRuleWorkItem(0, 0),
                CircleRuleWorkItem(1, 0),
                CircleRuleWorkItem(2, 0),
            ],
        )
        self.assertIsNone(worklist.pop())

    def test_duplicate_schedule_keeps_earliest_start_and_can_move_front(self) -> None:
        worklist = CircleRuleWorklist(3)
        self.assertEqual(worklist.pop(), CircleRuleWorkItem(0, 0))
        self.assertEqual(worklist.pop(), CircleRuleWorkItem(1, 0))

        worklist.schedule(1, 8)
        worklist.schedule(1, 3, front=True)

        self.assertEqual(worklist.pop(), CircleRuleWorkItem(1, 3))
        self.assertEqual(worklist.pop(), CircleRuleWorkItem(2, 0))

    def test_modification_requests_exactly_one_global_validation_sweep(self) -> None:
        worklist = CircleRuleWorklist(2)
        self.assertIsNotNone(worklist.pop())
        self.assertIsNotNone(worklist.pop())
        self.assertIsNone(worklist.pop())

        worklist.mark_modified()
        self.assertTrue(worklist.refill_for_validation(2))
        self.assertEqual(worklist.pop(), CircleRuleWorkItem(0, 0))
        self.assertEqual(worklist.pop(), CircleRuleWorkItem(1, 0))
        self.assertFalse(worklist.refill_for_validation(2))


if __name__ == "__main__":
    unittest.main()
