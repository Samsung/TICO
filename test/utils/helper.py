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

"""
This module provides utilites that help test.
"""

import unittest
from typing import Callable, List

import torch
from test.utils.base_builders import TestDictBuilderBase
from torch.export import ExportedProgram


def num_of_ops(ep: ExportedProgram, ops: List[torch._ops.OpOverload]):
    count = 0
    for node in ep.graph.nodes:
        if not node.op == "call_function":
            continue
        if node.target in ops:
            count += 1

    return count


def get_args_kwargs(example_inputs: tuple):
    if len(example_inputs) == 0:
        return (), {}

    if isinstance(example_inputs[-1], dict):
        return example_inputs[:-1], example_inputs[-1]
    else:
        return example_inputs, {}


def declare_unittests(global_obj: dict, namespace: str, builder: Callable):
    """
    Build unittests from the namespace and declare them in global_obj
    """
    assert issubclass(builder, TestDictBuilderBase)  # type: ignore[arg-type]

    # Collect tests from the namespace
    b = builder(namespace=namespace)

    for submodule in b.submodules:
        testdict = b.build(submodule)
        if len(testdict) > 0:
            # Create a new class using type(name, bases, dict)
            new_class = type(submodule, (unittest.TestCase,), testdict)
            # Set up a proper name where global_obj belongs
            new_class.__module__ = global_obj["__name__"]

            # Class is declared in the global_obj
            # NOTE Replacing '.' with '_' to avoid potential namespace corruption
            global_obj[submodule.replace(".", "_")] = new_class
