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

import unittest

import numpy as np

from tico.circle.analysis import TensorContract
from tico.circle.errors import CircleValueError
from tico.circle.value import TensorQuantization, TensorValue
from tico.circle.value.dtype import circle_tensor_type_value

from test.unit_test.circle.infrastructure_fixture import (
    fake_object_factory,
    FakeDetails,
    FakeTensor,
    FLOAT32,
    INT8,
)


class TensorContractTest(unittest.TestCase):
    """Check tensor metadata capture, validation, and reconstruction."""

    def test_contract_round_trip_preserves_nested_metadata(self):
        """Copy quantization, sparsity, rank, and variant metadata independently."""

        quantization = TensorQuantization(
            scale=(0.25, 0.5),
            zero_point=(0, 0),
            quantized_dimension=1,
            details_type=3,
            details=FakeDetails(),
        )
        source = FakeTensor(
            name="source",
            buffer=4,
            shape=[1, 2],
            shapeSignature=[-1, 2],
            type=INT8,
            quantization=quantization.to_object(fake_object_factory),
            sparsity=FakeDetails(blockSize=16),
            hasRank=True,
            variantTensors=[FakeDetails(blockSize=8)],
        )

        contract = TensorContract.from_tensor(source)
        target = contract.make_tensor(
            name="target",
            buffer_index=2,
            factory=fake_object_factory,
        )
        source.sparsity.axes.append(9)

        self.assertEqual(target.name, "target")
        self.assertEqual(target.buffer, 2)
        self.assertTrue(contract.matches_tensor(target))
        self.assertEqual(target.sparsity.axes, [0, 1])
        self.assertEqual(target.variantTensors[0].blockSize, 8)

    def test_contract_from_value_uses_concrete_value_metadata(self):
        """Build a contract directly from an immutable tensor value."""

        value = TensorValue(
            FLOAT32,
            (1, 3),
            np.arange(3, dtype=np.float32),
        )
        contract = TensorContract.from_value(value, shape_signature=(-1, 3))

        self.assertEqual(contract.tensor_type, FLOAT32)
        self.assertEqual(contract.shape, (1, 3))
        self.assertEqual(contract.element_count, 3)

    def test_dynamic_signature_requires_concrete_placeholder_one(self):
        """Reject symbolic dimensions whose concrete placeholder is not one."""

        with self.assertRaisesRegex(CircleValueError, "placeholder value 1"):
            TensorContract(
                tensor_type=FLOAT32,
                shape=(2, 3),
                shape_signature=(-1, 3),
            )

    def test_per_axis_quantization_axis_must_fit_rank(self):
        """Reject per-axis quantization outside the tensor rank."""

        quantization = TensorQuantization(
            scale=(0.25, 0.5),
            zero_point=(0, 0),
            quantized_dimension=2,
        )
        with self.assertRaisesRegex(CircleValueError, "within the tensor rank"):
            TensorContract(
                tensor_type=INT8,
                shape=(1, 2),
                quantization=quantization,
            )

    def test_mx_axis_without_affine_vectors_must_fit_rank(self):
        """Validate MX block axes even when no affine qparams are present."""

        for name in ("MXINT8", "MXFP4"):
            for shape, axis in (((), 0), ((2, 32), 2), ((2, 32), 5)):
                with self.subTest(dtype=name, shape=shape, axis=axis):
                    with self.assertRaisesRegex(
                        CircleValueError, "MX quantized_dimension.*tensor rank"
                    ):
                        TensorContract(
                            tensor_type=circle_tensor_type_value(name),
                            shape=shape,
                            quantization=TensorQuantization(quantized_dimension=axis),
                        )

    def test_mx_axis_round_trip_uses_normalized_dimension(self):
        """Keep valid MX axes through generated schema tensor contracts."""

        for name in ("MXINT8", "MXFP4"):
            with self.subTest(dtype=name):
                contract = TensorContract(
                    tensor_type=circle_tensor_type_value(name),
                    shape=(2, 32),
                    quantization=TensorQuantization(quantized_dimension=1),
                )
                tensor = contract.make_tensor(name="mx")
                self.assertEqual(TensorContract.from_tensor(tensor), contract)
                tensor.quantization.quantizedDimension = -1
                with self.assertRaisesRegex(
                    CircleValueError, "quantized_dimension must not be negative"
                ):
                    TensorContract.from_tensor(tensor)

    def test_default_rank_metadata_round_trips_through_generated_defaults(self):
        """Treat a false generated hasRank field as the canonical absent value."""

        contract = TensorContract(tensor_type=FLOAT32, shape=(1,))
        tensor = contract.make_tensor(
            name="value",
            factory=fake_object_factory,
        )

        self.assertTrue(contract.matches_tensor(tensor))

    def test_hash_uses_structural_nested_metadata(self):
        """Make equivalent independently allocated metadata hash-compatible."""

        first = TensorContract(
            tensor_type=FLOAT32,
            shape=(1,),
            sparsity=FakeDetails(),
        )
        second = TensorContract(
            tensor_type=FLOAT32,
            shape=(1,),
            sparsity=FakeDetails(),
        )

        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))


if __name__ == "__main__":
    unittest.main()
