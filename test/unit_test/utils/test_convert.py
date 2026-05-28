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

import tempfile
import unittest
from pathlib import Path

import tico
import torch
from torch.export import export, save

from test.modules.op.add import SimpleAdd


class ConvertTest(unittest.TestCase):
    def test_args(self):
        m = SimpleAdd()
        args, kwargs = m.get_example_inputs()
        tico.convert(m.eval(), args, kwargs)

    def test_kwargs(self):
        m = SimpleAdd()
        args, _ = m.get_example_inputs()
        kwargs = {"x": args[0], "y": args[1]}
        tico.convert(m.eval(), tuple(), kwargs)

    def test_args_kwargs(self):
        m = SimpleAdd()
        args, _ = m.get_example_inputs()
        x, y = args
        tico.convert(m.eval(), (x,), {"y": y})


class ConvertFromExportedProgramTest(unittest.TestCase):
    def test_args(self):
        m = SimpleAdd()
        args, kwargs = m.get_example_inputs()
        with torch.no_grad():
            ep = export(m.eval(), args, kwargs)

        tico.convert_from_exported_program(ep)


class ConvertFromPt2Test(unittest.TestCase):
    def test_args(self):
        m = SimpleAdd()
        args, kwargs = m.get_example_inputs()
        with torch.no_grad():
            ep = export(m.eval(), args, kwargs)

        file_name: str = "ConvertFromPt2Test.test_args.pt2"
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / file_name
            save(ep, file_path)
            tico.convert_from_pt2(file_path)
