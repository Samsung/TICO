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

from typing import Sequence

import numpy as np
import torch
from circle_schema import circle

from tico.serialize.circle_mapping import to_circle_shape
from tico.utils.dtype import circle_dtype_to_torch_dtype
from tico.utils.installed_packages import is_dynamic_cache_available


def is_dynamic_cache_instance(value):
    if is_dynamic_cache_available():
        from transformers.cache_utils import DynamicCache

        return isinstance(value, DynamicCache)
    else:
        return False


def flatten_and_convert_kwargs(kwargs: dict) -> dict[str, torch.Tensor]:
    result = {}  # type: ignore[var-annotated]
    for k, v in kwargs.items():
        if v is None:
            continue
        elif isinstance(v, (list, tuple)):
            # 1. handle list
            def unpack_recursive(name, value, store=None):
                if store is None:
                    store = {}

                if isinstance(value, (tuple, list)):
                    for i, v in enumerate(value):
                        # recursive call. Append index to name and explore lower level
                        unpack_recursive(f"{name}_{i}", v, store)
                else:
                    # base type (scalar etc.) directly stored
                    store[name] = value

                return store

            unpack_recursive(k, v, result)
        elif is_dynamic_cache_instance(v):
            # 2. handle DynamicCache
            for idx, cache_val in enumerate(v.key_cache):
                result[f"{k}_key_cache_{idx}"] = cache_val

            for idx, cache_val in enumerate(v.value_cache):
                result[f"{k}_value_cache_{idx}"] = cache_val
        else:
            result[k] = v

    # 3. Convert to tensors
    for k, v in result.items():
        result[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v)

    return result


def flatten_and_convert_args(args: Sequence) -> tuple:
    result = []  # type: ignore[var-annotated]
    for item in args:
        if item is None:
            continue

        # 1. recursion on list and tuple
        if isinstance(item, (list, tuple)):
            result.extend(flatten_and_convert_args(item))
            continue

        # 2. handle DynamicCache
        if is_dynamic_cache_available():
            from transformers.cache_utils import DynamicCache

            if isinstance(item, DynamicCache):
                # NOTE The tensor order is: key_in → key_out → value_in → value_out
                #
                # Refer to https://github.com/huggingface/transformers/blob/3457e8e73e4f5532cc69059682b1ba4484d7e7e8/src/transformers/cache_utils.py#L557
                # ```
                # self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                # self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                # ```
                result.extend(item.key_cache)
                result.extend(item.value_cache)
                continue

        # 3. Convert to tensors
        result.append(item if isinstance(item, torch.Tensor) else torch.tensor(item))

    return tuple(result)


class ModelInputSpec:
    @classmethod
    def load(cls, circle_path):
        def load(circle_path: str) -> bytes:
            with open(circle_path, "rb") as f:
                buf = bytes(f.read())
            return buf

        circle_binary = load(circle_path)
        return cls(circle_binary)

    def __init__(self, circle_binary):
        model = circle.Model.Model.GetRootAsModel(circle_binary, 0)
        assert model.SubgraphsLength() == 1, "Only one subgraph is supported"

        graph = model.Subgraphs(0)
        tensors = [graph.Tensors(graph.Inputs(o)) for o in range(graph.InputsLength())]

        self.names = [t.Name().decode("utf-8").split("::")[-1] for t in tensors]
        self.shapes = [t.ShapeAsNumpy() for t in tensors]
        self.shape_signatures = list(
            map(
                lambda x: None if (isinstance(x, int) and x == 0) else x,
                (t.ShapeSignatureAsNumpy() for t in tensors),
            )
        )
        self.types: list[torch.dtype] = [
            circle_dtype_to_torch_dtype(t.Type()) for t in tensors
        ]
        self.name_to_idx = {name: idx for idx, name in enumerate(self.names)}

    def bind(self, args, kwargs, check=True):
        """Convert args and kwargs into an ordered list according to model input order"""
        inputs = []
        args = flatten_and_convert_args(args)
        kwargs = flatten_and_convert_kwargs(kwargs)

        arg_num = len(args) + len(kwargs)
        m_input_num = len(self.names)
        if arg_num != m_input_num:
            raise ValueError(
                f"Mismatch: number of model inputs and number of passed arguments are not the same: inputs({m_input_num}) != passed({arg_num}), input spec: {self.names}"
            )

        # 1. positional arguments
        for i, val in enumerate(args):
            name = self.names[i]
            inputs.append(val)

        # 2. keyword arguments
        for idx in range(len(args), len(self.names)):
            name = self.names[idx]
            inputs.append(kwargs[name])

        if check:
            self.check_types(inputs)
            self.check_shapes(inputs)

        return inputs

    def check_types(self, inputs):
        """Check the types of input values"""
        for i, (inp, ref_type) in enumerate(zip(inputs, self.types)):
            # TODO: Support more data types (np array)
            assert isinstance(
                inp, (torch.Tensor | int | float)
            ), f"Input '{self.names[i]}' type must be a torch tensor or scalar."

            if isinstance(inp, torch.Tensor):
                if inp.dtype != ref_type:
                    raise TypeError(
                        f"Input '{self.names[i]}' type {inp.dtype} != expected {ref_type}"
                    )
            else:
                # Scalars (int, float)
                if ref_type == torch.float32:
                    if not isinstance(inp, (float)):
                        raise TypeError(
                            f"Input '{self.names[i]}' type {type(inp)} != expected {ref_type}"
                        )
                elif ref_type == torch.int64:
                    if not isinstance(inp, (int)):
                        raise TypeError(
                            f"Input '{self.names[i]}' type {type(inp)} != expected {ref_type}"
                        )
                else:
                    print(f"Unexpected ref_type: {ref_type}")

    def check_shapes(self, inputs):
        """Check the shapes of input values"""

        def merge(shape, shape_sig):
            """
            Merge shape signature with shape
            """
            from copy import deepcopy

            shape_merged = deepcopy(shape)
            if shape_sig is not None:
                for idx, ss in enumerate(shape_sig):
                    if ss == -1:
                        shape_merged[idx] = -1

            return shape_merged

        for i, (inp, ref_shape, ref_shape_sig) in enumerate(
            zip(inputs, self.shapes, self.shape_signatures)
        ):
            # TODO: Support more data types (np array)
            assert isinstance(
                inp, (torch.Tensor | int | float)
            ), f"Input '{self.names[i]}' type must be a torch tensor or scalar."

            if isinstance(inp, torch.Tensor):  # Tensor
                in_shape, in_shape_sig = to_circle_shape(inp.size())

                if len(in_shape) != len(ref_shape):
                    raise ValueError(
                        f"Input '{self.names[i]}' has invalid rank {len(in_shape)}!= expected {len(ref_shape)}"
                    )

                in_merged_shape = merge(in_shape, in_shape_sig)
                ref_merged_shape = merge(ref_shape, ref_shape_sig)
                for in_shp, ref_shp in zip(in_merged_shape, ref_merged_shape):
                    if ref_shp == -1:
                        continue
                    if in_shp == -1:
                        raise ValueError(
                            f"Input '{self.names[i]}' has unknown dimension {inp.size()} != expected shape({ref_shape}) / shape signature({ref_shape_sig}) "
                        )
                    if in_shp != ref_shp:
                        raise ValueError(
                            f"Input '{self.names[i]}' has wrong dimension {inp.size()} != expected shape({ref_shape}) / shape signature({ref_shape_sig}) "
                        )
            elif isinstance(inp, (int, float)):  # Scalar
                if len(ref_shape) > 0:
                    raise ValueError(
                        f"Input '{self.names[i]}' has invalid rank {len(ref_shape)}"
                    )
            else:
                print(f"Unexpected input type: {type(inp)}")
