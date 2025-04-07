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

from tico.passes import ops
from tico.passes.convert_layout_op_to_reshape import ConvertLayoutOpToReshape

from test.modules.op.squeeze import SimpleSqueezeWithDims
from test.modules.op.unsqueeze import SimpleUnsqueeze

from test.modules.op.view import SimpleView
from test.utils.helper import num_of_ops
from test.utils.pass_value_test import SinglePassValueTest


class ConvertLayoutOpToReshapeTest(SinglePassValueTest):
    def test_view(self):
        self.setup(SimpleView())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.view), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)

        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.view), 0)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)

    def test_unsqueeze(self):
        self.setup(SimpleUnsqueeze())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.unsqueeze), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)

        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.unsqueeze), 0)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)

    def test_squeeze(self):
        self.setup(SimpleSqueezeWithDims())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.squeeze), 1)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 0)

        self.run_value_test(ConvertLayoutOpToReshape())
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.squeeze), 0)
        self.assertEqual(num_of_ops(self.exported_program(), ops.aten.reshape), 1)
