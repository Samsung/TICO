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

from tico.experimental.quantization.custom.qscheme import QScheme


class TestQScheme(unittest.TestCase):
    def test_enum_members(self):
        self.assertTrue(QScheme.PER_TENSOR_AFFINE in QScheme)
        self.assertTrue(QScheme.PER_CHANNEL_AFFINE in QScheme)

    def test_is_per_channel(self):
        self.assertFalse(QScheme.PER_TENSOR_AFFINE.is_per_channel())
        self.assertTrue(QScheme.PER_CHANNEL_AFFINE.is_per_channel())

    def test_str(self):
        self.assertEqual(str(QScheme.PER_TENSOR_AFFINE), "per_tensor_affine")
