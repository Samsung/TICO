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

# type: ignore
import unittest
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

from tico.utils.utils import ArgTypeError, enforce_type


class _Node:
    pass


@enforce_type
@dataclass
class _ArgsAny:
    x: Any


@enforce_type
@dataclass
class _ArgsOptional:
    x: Optional[int]


class EnforceTypeTest_Trivial(unittest.TestCase):
    def test_args0(self):
        _ArgsAny(None)
        _ArgsAny(1)
        _ArgsAny([1])
        _ArgsAny("str")
        _ArgsAny(["str"])
        _ArgsAny([0, 1, 2])
        _ArgsAny([0, 1, 2, "str"])

    def test_args1(self):
        _ArgsOptional(None)
        _ArgsOptional(1)

    def test_args1_neg(self):
        with self.assertRaises(ArgTypeError):
            _ArgsOptional("str")
        with self.assertRaises(ArgTypeError):
            _ArgsOptional(["str"])
        with self.assertRaises(ArgTypeError):
            _ArgsOptional([0, 1, 2])
        with self.assertRaises(ArgTypeError):
            _ArgsOptional([0, 1, 2, "str"])


@enforce_type
@dataclass
class _ArgsOptionalList:
    x: Optional[List[int]]


class EnforceTypeTest_OptionalList(unittest.TestCase):
    def test_optional_list(self):
        _ArgsOptionalList(None)
        _ArgsOptionalList([1])
        _ArgsOptionalList([0, 1, 2])

    def test_optional_list_neg(self):
        with self.assertRaises(ArgTypeError):
            _ArgsOptionalList("str")
        with self.assertRaises(ArgTypeError):
            _ArgsOptionalList(["str"])
        with self.assertRaises(ArgTypeError):
            _ArgsOptionalList(1)
        with self.assertRaises(ArgTypeError):
            _ArgsOptionalList([0, 1, 2, "str"])


@enforce_type
@dataclass
class _ArgsListUnion:
    x: Optional[List[Union[int, str]]]


class EnforceTypeTest_ListUnion(unittest.TestCase):
    def test_optional_list(self):
        _ArgsListUnion(None)
        _ArgsListUnion([1])
        _ArgsListUnion(["str"])
        _ArgsListUnion([0, 1, 2, "str"])

    def test_optional_list_neg(self):
        with self.assertRaises(ArgTypeError):
            _ArgsListUnion("str")
        with self.assertRaises(ArgTypeError):
            _ArgsListUnion(1)


@enforce_type
@dataclass
class _ArgsUnionOptionalList:
    x: Union[Optional[List[int]], str]


class EnforceTypeTest_UnionOptionalList(unittest.TestCase):
    def test_union(self):
        _ArgsUnionOptionalList(None)
        _ArgsUnionOptionalList([1])
        _ArgsUnionOptionalList([0, 1, 2])
        _ArgsUnionOptionalList("str")

    def test_union_neg(self):
        with self.assertRaises(ArgTypeError):
            _ArgsUnionOptionalList(["str"])
        with self.assertRaises(ArgTypeError):
            _ArgsUnionOptionalList(1)
        with self.assertRaises(ArgTypeError):
            _ArgsUnionOptionalList([0, 1, 2, "str"])


@enforce_type
@dataclass
class _ArgsSimpleDatatype:
    x: int
    y: str


class EnforceTypeTest_SimpleDatatype(unittest.TestCase):
    def test_simple(self):
        _ArgsSimpleDatatype(1, "2")

    def test_simple_neg(self):
        with self.assertRaises(ArgTypeError):
            _ArgsSimpleDatatype("2", 1)


@enforce_type
@dataclass
class _ArgsDict:
    x: dict[str, int]


class EnforceTypeTest_SimpleDict(unittest.TestCase):
    def test_simple(self):
        _ArgsDict({"str": 1})

    def test_simple_neg(self):
        with self.assertRaises(ArgTypeError):
            _ArgsDict({"str": "str"})
        with self.assertRaises(ArgTypeError):
            _ArgsDict({"str": [1, 2]})
        with self.assertRaises(ArgTypeError):
            _ArgsDict({1: 1})


@enforce_type
@dataclass
class _ArgsCombined:
    bias: Optional[_Node] = None
    stride: Union[Optional[List[int]], bool] = field(default_factory=lambda: [1, 1])
    padding: Union[List[int], str, Any] = field(default_factory=lambda: [0, 0])
    dilation: Union[Union[List[int], List[int]]] = field(default_factory=lambda: [1, 1])
    groups: int = 1

    def __post_init__(self):
        assert self.groups == 1, "Only support group 1 conv2d"


class EnforceTypeTest_Combined(unittest.TestCase):
    def test_all_neg(self):
        with self.assertRaises(ArgTypeError):
            # stride is wrong
            node_args = (_Node(), "stride", [1], [1])
            node_kwargs = {"groups": 1}

            _ArgsCombined(*node_args, **node_kwargs)

        with self.assertRaises(ArgTypeError):
            # stride is wrong
            node_args = (None, 1, [1], [1])
            node_kwargs = {"groups": 1}

            _ArgsCombined(*node_args, **node_kwargs)

        with self.assertRaises(ArgTypeError):
            # stride is wrong
            node_args = (_Node(), ["str"], [1], [1])
            node_kwargs = {"groups": 1}

            _ArgsCombined(*node_args, **node_kwargs)

        with self.assertRaises(AssertionError):
            node_args = (_Node(), [1], [1], [1])
            # groups is wrong
            node_kwargs = {"groups": 10}

            _ArgsCombined(*node_args, **node_kwargs)

        with self.assertRaises(ArgTypeError):
            # stride is wrong
            node_args = (_Node(), [1, 1, "str"], [1], [1])
            node_kwargs = {"groups": 10}

            _ArgsCombined(*node_args, **node_kwargs)

    def test_all(self):
        node_args = (_Node(), [1, 1, 1], [1], [1])
        node_kwargs = {"groups": 1}

        _ArgsCombined(*node_args, **node_kwargs)
