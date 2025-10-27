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

from tico.quantization.wrapq.qscheme import QScheme


class TestQScheme(unittest.TestCase):
    def test_enum_members(self):
        self.assertIn(QScheme.PER_TENSOR_ASYMM, QScheme)
        self.assertIn(QScheme.PER_TENSOR_SYMM, QScheme)
        self.assertIn(QScheme.PER_CHANNEL_ASYMM, QScheme)
        self.assertIn(QScheme.PER_CHANNEL_SYMM, QScheme)

    def test_is_per_channel(self):
        self.assertFalse(QScheme.PER_TENSOR_ASYMM.is_per_channel())
        self.assertTrue(QScheme.PER_CHANNEL_ASYMM.is_per_channel())

    def test_is_symmetric(self):
        self.assertTrue(QScheme.PER_TENSOR_SYMM.is_symmetric())
        self.assertFalse(QScheme.PER_CHANNEL_ASYMM.is_symmetric())

    def test_str(self):
        self.assertEqual(str(QScheme.PER_TENSOR_ASYMM), "per_tensor_asymm")
        self.assertEqual(str(QScheme.PER_CHANNEL_SYMM), "per_channel_symm")
