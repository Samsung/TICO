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

from tico.serialize.operators.utils import get_integer_dtype_min


class TestGetIntegerDtypeMin(unittest.TestCase):
    def test_signed_types(self):
        self.assertEqual(get_integer_dtype_min("int2"), -2)
        self.assertEqual(get_integer_dtype_min("int3"), -4)
        self.assertEqual(get_integer_dtype_min("int4"), -8)
        self.assertEqual(get_integer_dtype_min("int8"), -128)
        self.assertEqual(get_integer_dtype_min("int16"), -32768)

    def test_unsigned_types(self):
        self.assertEqual(get_integer_dtype_min("uint2"), 0)
        self.assertEqual(get_integer_dtype_min("uint4"), 0)
        self.assertEqual(get_integer_dtype_min("uint8"), 0)
        self.assertEqual(get_integer_dtype_min("uint16"), 0)

    def test_invalid_format(self):
        with self.assertRaises(ValueError):
            get_integer_dtype_min("float32")
        with self.assertRaises(ValueError):
            get_integer_dtype_min("int")
        with self.assertRaises(ValueError):
            get_integer_dtype_min("intXYZ")
        with self.assertRaises(ValueError):
            get_integer_dtype_min("")

    def test_too_small_bitwidth(self):
        with self.assertRaises(ValueError):
            get_integer_dtype_min("int0")
        with self.assertRaises(ValueError):
            get_integer_dtype_min("uint1")
        with self.assertRaises(ValueError):
            get_integer_dtype_min("int1")
