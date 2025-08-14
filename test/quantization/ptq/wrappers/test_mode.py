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

from tico.experimental.quantization.ptq.mode import Mode


class TestModeEnum(unittest.TestCase):
    def test_unique_values(self):
        self.assertEqual(len(list(Mode)), 3)
        self.assertNotEqual(Mode.NO_QUANT, Mode.CALIB)
        self.assertNotEqual(Mode.NO_QUANT, Mode.QUANT)
        self.assertNotEqual(Mode.CALIB, Mode.QUANT)

    def test_str(self):
        self.assertEqual(str(Mode.NO_QUANT), "no_quant")
        self.assertEqual(str(Mode.CALIB), "calib")
