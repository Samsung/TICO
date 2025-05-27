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

from tico.utils.utils import get_quant_dtype


class TestGetQuantDtype(unittest.TestCase):
    def test_supported_ranges(self):
        self.assertEqual(get_quant_dtype(-32768, 32767), "int16")
        self.assertEqual(get_quant_dtype(0, 65535), "uint16")
        self.assertEqual(get_quant_dtype(0, 255), "uint8")
        self.assertEqual(get_quant_dtype(-128, 127), "int8")
        self.assertEqual(get_quant_dtype(-8, 7), "int4")
        self.assertEqual(get_quant_dtype(0, 15), "uint4")

    def test_unsupported_ranges(self):
        with self.assertRaises(ValueError):
            get_quant_dtype(0, 10)
        with self.assertRaises(ValueError):
            get_quant_dtype(-100, 100)
        with self.assertRaises(ValueError):
            get_quant_dtype(256, 512)
        with self.assertRaises(ValueError):
            get_quant_dtype(-32768, 32768)
