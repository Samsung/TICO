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

import numpy as np

from tico.serialize.pack import pack_buffer


class PackTest(unittest.TestCase):
    def test_pack_uint4(self):
        input_ = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint8)

        output_ = pack_buffer(input_, "uint4")

        self.assertEqual((3,), output_.shape)
        self.assertEqual(1 + (2 << 4), output_[0])
        self.assertEqual(3 + (4 << 4), output_[1])
        self.assertEqual(5 + (6 << 4), output_[2])

    def test_pack_uint4_odd(self):
        input_ = np.array([1, 2, 3, 4, 5], dtype=np.uint8)

        output_ = pack_buffer(input_, "uint4")

        self.assertEqual((3,), output_.shape)
        self.assertEqual(1 + (2 << 4), output_[0])
        self.assertEqual(3 + (4 << 4), output_[1])
        self.assertEqual(5, output_[2])

    def test_pack_dtype_mismatch_NEG(self):
        input_ = np.array([1, 2, 3, 4, 5, 6], dtype=np.int16)

        # uint4 data has to be saved in uint8
        with self.assertRaises(RuntimeError):
            pack_buffer(input_, "uint4")
