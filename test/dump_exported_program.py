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

import argparse
import importlib
import inspect

import torch

from test.modules.base import TestModuleBase


def dump(model, example_inputs):
    exported_program = torch.export.export(model.eval(), example_inputs)

    print(exported_program)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--module_name",
        required=True,
        help="To dump all models in a module, provide a module name.\
                Or you can dump a single class in the module.\
                (e.g. test.modules.op.add, test.modules.op.mean.SimpleMeanKeepDim)",
    )

    args = parser.parse_args()

    try:
        module = importlib.import_module(args.module_name)
        models = inspect.getmembers(module, inspect.isclass)

        for name, cls in models:
            model = cls()
            assert isinstance(model, TestModuleBase)
            example_inputs = model.get_example_inputs()

            dump(model, example_inputs)

    except ModuleNotFoundError:
        path = args.module_name.split(".")
        module_name = (".").join(path[:-1])
        model_name = path[-1]

        module = importlib.import_module(module_name)
        models = inspect.getmembers(module, inspect.isclass)

        if all(name != model_name for name, _ in models):
            raise RuntimeError("Invalid module name")
        for name, cls in models:
            if name == model_name:
                model = cls()
                assert isinstance(model, TestModuleBase)
                example_inputs = model.get_example_inputs()

                dump(model, example_inputs)


if __name__ == "__main__":
    main()
