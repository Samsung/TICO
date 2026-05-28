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

import tico
import torch
from tico.utils.record_input import RecordingInput

from test.modules.op.add import SimpleAdd


class RecordInputTest(unittest.TestCase):
    def test_args(self):
        m = SimpleAdd()
        args, kwargs = m.get_example_inputs()
        with RecordingInput(m) as rec:
            m.eval()
            m(*args, **kwargs)
            captured_input = rec.captured_input

        self.assertIsNotNone(captured_input)
        self.assertEqual(captured_input, args)
        tico.convert(m, captured_input)

    def test_args_kwargs(self):
        m = SimpleAdd()
        args, kwargs = (torch.ones(1),), {
            "y": torch.ones(1),
        }
        with RecordingInput(m) as rec:
            m.eval()
            m(*args, **kwargs)
            captured_input = rec.captured_input

        self.assertIsNotNone(captured_input)
        self.assertEqual(
            captured_input,
            (
                torch.ones(1),
                torch.ones(1),
            ),
        )
        tico.convert(m, captured_input)

    def test_kwargs(self):
        m = SimpleAdd()
        args, kwargs = (), {
            "x": torch.ones(1),
            "y": torch.ones(1),
        }

        with RecordingInput(m) as rec:
            m.eval()
            m(*args, **kwargs)
            captured_input = rec.captured_input

        self.assertIsNotNone(captured_input)
        self.assertEqual(
            captured_input,
            (
                torch.ones(1),
                torch.ones(1),
            ),
        )
        tico.convert(m, captured_input)

    def test_input_to_remove(self):
        m = SimpleAdd()
        args, kwargs = m.get_example_inputs()
        with RecordingInput(m, input_to_remove=["x"]) as rec:
            m.eval()
            m(*args, **kwargs)
            captured_input = rec.captured_input

        self.assertIsNotNone(captured_input)
        self.assertIsNone(captured_input[0])  # arg[0] = 'x'

    def test_condition(self):
        m = SimpleAdd()
        args, kwargs = m.get_example_inputs()
        condition = lambda arg_dict: False
        with RecordingInput(m, condition) as rec:
            m.eval()
            m(*args, **kwargs)
            captured_input = rec.captured_input

        self.assertEqual(captured_input, None)
