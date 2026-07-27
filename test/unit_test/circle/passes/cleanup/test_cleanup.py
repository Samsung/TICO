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

from tico.circle.passes import CirclePassContext
from tico.circle.passes.cleanup import CompactIndicesPass, DeadCodeEliminationPass

from test.unit_test.circle.fixture import make_test_document


class CircleCleanupPassTest(unittest.TestCase):
    def test_dead_code_elimination_removes_unreachable_operator(self):
        document = make_test_document()

        result = DeadCodeEliminationPass(subgraph_indices=(0,)).run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph(0).operators), 2)
        self.assertEqual(len(document.subgraph(1).operators), 1)

    def test_compact_indices_removes_dead_objects_after_dce(self):
        document = make_test_document()
        context = CirclePassContext(verify_after_each_pass=False)
        DeadCodeEliminationPass(subgraph_indices=(0,)).run(document, context)

        result = CompactIndicesPass().run(document, context)

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph(0).tensors), 4)
        self.assertEqual(len(document.model.buffers), 2)
        self.assertEqual(len(document.model.operatorCodes), 2)
        self.assertTrue(document.verify(raise_on_error=False).ok)
