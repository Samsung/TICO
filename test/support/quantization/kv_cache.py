# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
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

"""Shared scalar cache domains, deterministic hidden inputs and exact ABI parity.

Factories live in family modules so importing this module loads no optional models.
"""

from dataclasses import dataclass

import numpy as np
import torch
from tico.circle.runtime import ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.wrapq.mode import Mode


@dataclass
class CacheDomains:
    numpy_dtype: np.dtype
    compute_numpy_dtype: np.dtype
    storage_qparam: tuple[float, int]
    compute_qparam: tuple[float, int]
    producer_qparam: tuple[float, int]

    @classmethod
    def affine(cls, dtype, compute_dtype=None):
        compute_dtype = compute_dtype or dtype
        storage = np.dtype("int16" if dtype.signed else "uint8")
        compute = np.dtype("int16" if compute_dtype.signed else "uint8")
        return cls(
            storage,
            compute,
            (0.09375, 0 if dtype.signed else 113),
            (0.21875, 0 if compute_dtype.signed else 91),
            (0.125, 0 if dtype.signed else 107),
        )


def load_qparam(observer, scale: float, zero_point: int) -> None:
    observer.load_qparams(
        torch.tensor(scale),
        torch.tensor(zero_point, dtype=torch.int32),
        lock=True,
    )


def hidden(length: int) -> torch.Tensor:
    return torch.linspace(-0.91, 1.07, length * 8).reshape(1, length, 8)


def tensor_qparam(tensor) -> tuple[float, int] | None:
    quantization = tensor.quantization
    if quantization is None or not quantization.scale:
        return None
    return float(quantization.scale[0]), int(quantization.zero_point[0])


def producer_map(program):
    producers = {}
    for operator_index, operator in enumerate(program.operators):
        for tensor_index in operator.outputs:
            producers[tensor_index] = (operator_index, operator)
    return producers


def find_named_tensor(program, indices, suffix: str):
    for index in indices:
        tensor = program.tensor(index)
        if tensor.name and tensor.name.endswith(suffix):
            return tensor
    raise AssertionError(f"Tensor ending with {suffix!r} was not found")


def assert_value_preserving_qparams(case, program) -> None:
    for operator in program.operators:
        if operator.name not in {"RESHAPE", "TRANSPOSE", "CONCATENATION"}:
            continue
        # Shape and permutation operands are integer metadata, not values
        # transformed in the activation quantization domain.
        data_inputs = (
            operator.inputs if operator.name == "CONCATENATION" else operator.inputs[:1]
        )
        tensors = [program.tensor(index) for index in data_inputs]
        output = program.tensor(operator.outputs[0])
        qparams = [tensor_qparam(tensor) for tensor in (*tensors, output)]
        quantized = [qparam for qparam in qparams if qparam is not None]
        if not quantized:
            continue
        case.assertEqual(len(quantized), len(qparams), operator.name)
        case.assertEqual(len(set(quantized)), 1, operator.name)
        case.assertEqual(
            len({tensor.dtype for tensor in (*tensors, output)}), 1, operator.name
        )


def assert_cache_parity(case, attention, module, args, runtime, *, label: str):
    # FAKE_QUANTIZE uses dequantized FLOAT32 at its integer ABI. Explicitly
    # encode each input first, and give eager exactly those same values.
    case.assertIs(attention._mode, Mode.QUANT)
    leaves, spec = torch.utils._pytree.tree_flatten(args)
    case.assertEqual(len(leaves), len(runtime.program.inputs))
    encoded = []
    for value, index in zip(leaves, runtime.program.inputs):
        tensor = runtime.program.tensor(index)
        array = value.detach().numpy()
        if tensor.is_quantized_integer:
            array = fake_quantize_array(array, tensor)
        encoded.append(np.array(array, copy=True))
    eager_args = torch.utils._pytree.tree_unflatten(
        [torch.from_numpy(value.copy()) for value in encoded], spec
    )
    with torch.no_grad():
        eager = module(*eager_args)
    result = runtime.run(encoded, mode=ExecutionMode.FAKE_QUANTIZE)
    case.assertEqual(len(eager), 3, label)
    case.assertEqual(len(result.outputs), len(eager), label)
    for name, expected, actual, index in zip(
        ("hidden", "K_delta", "V_delta"),
        eager,
        result.outputs,
        runtime.program.outputs,
    ):
        expected_array = expected.detach().numpy()
        error = np.abs(expected_array - actual)
        metrics = {
            "max_abs": float(error.max()),
            "mean_abs": float(error.mean()),
            "differing": int(np.count_nonzero(expected_array != actual)),
            "qparam": tensor_qparam(runtime.program.tensor(index)),
        }
        print(f"{label} {name}: {metrics}")
        np.testing.assert_array_equal(
            actual, expected_array, err_msg=f"{label} {name}: {metrics}"
        )
    return result.outputs
