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
