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

import unittest

from tico.serialize.circle_mapping import validate_circle_shape


class CircleSerializeTest(unittest.TestCase):
    def test_validate_circle_shape(self):
        # static shape
        validate_circle_shape(shape=[1, 2, 3], shape_signature=None)
        # dynamic shape
        validate_circle_shape(shape=[1, 1, 3], shape_signature=[1, -1, 3])
        validate_circle_shape(shape=[1, 1, 3], shape_signature=[-1, -1, 3])

        # Invalid dynamic shape
        with self.assertRaises(ValueError):
            validate_circle_shape(shape=[1, 2, 3], shape_signature=[1, -1, 2])
        with self.assertRaises(ValueError):
            validate_circle_shape(shape=[1, 2, 3], shape_signature=[1, -2, 3])
        with self.assertRaises(ValueError):
            validate_circle_shape(shape=[1], shape_signature=[-1, -1])
        with self.assertRaises(ValueError):
            validate_circle_shape(shape=[1, 2, 3], shape_signature=[])
