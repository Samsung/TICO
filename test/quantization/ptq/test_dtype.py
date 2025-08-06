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

from tico.experimental.quantization.ptq.dtypes import DType, INT8, UINT4


class TestDType(unittest.TestCase):
    def test_presets(self):
        self.assertEqual(INT8.bits, 8)
        self.assertTrue(INT8.signed)
        self.assertEqual(UINT4.bits, 4)
        self.assertFalse(UINT4.signed)

    def test_range_signed(self):
        dt = DType.int(6)  # 6-bit signed
        self.assertEqual(dt.qmin, -32)
        self.assertEqual(dt.qmax, 31)

    def test_range_unsigned(self):
        dt = DType.uint(5)  # 5-bit unsigned
        self.assertEqual(dt.qmin, 0)
        self.assertEqual(dt.qmax, 31)

    def test_str(self):
        self.assertEqual(str(DType.int(3)), "int3")
        self.assertEqual(str(DType.uint(7)), "uint7")
