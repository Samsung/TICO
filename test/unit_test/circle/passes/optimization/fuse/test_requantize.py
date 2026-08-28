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

from tico.circle.analysis import TensorContract
from tico.circle.graph import as_indices
from tico.circle.passes import CirclePassContext, FuseOutputRequantizePass
from tico.circle.passes.optimization._utils import operator_builtin_code
from tico.circle.value import TensorQuantization

from test.unit_test.circle.infrastructure_fixture import make_empty_document, UINT8
from test.unit_test.circle.passes.optimization._fixture import (
    add_f32,
    make_builder,
    make_codec,
    optimization_object_factory,
)

INT16 = 7
QUANTIZE = 30
CONV_2D = 31
DEPTHWISE_CONV_2D = 32
FULLY_CONNECTED = 33
TRANSPOSE_CONV = 34
ADD = 1

BUILTIN_CODES = {
    "QUANTIZE": QUANTIZE,
    "CONV_2D": CONV_2D,
    "DEPTHWISE_CONV_2D": DEPTHWISE_CONV_2D,
    "FULLY_CONNECTED": FULLY_CONNECTED,
    "TRANSPOSE_CONV": TRANSPOSE_CONV,
    "ADD": ADD,
}


def quantized_contract(
    shape: tuple[int, ...],
    tensor_type: int,
    *,
    scale: tuple[float, ...] = (1.0,),
    zero_point: tuple[int, ...] = (0,),
) -> TensorContract:
    """Create one static per-tensor quantized tensor contract."""

    return TensorContract(
        tensor_type=tensor_type,
        shape=shape,
        shape_signature=shape,
        quantization=TensorQuantization(scale=scale, zero_point=zero_point),
    )


class FuseOutputRequantizeTest(unittest.TestCase):
    """Check QUANTIZE folding into requantizing producers and its guards."""

    def setUp(self) -> None:
        """Create a schema-independent codec for each test."""

        self.codec = make_codec()

    def _pass(self) -> FuseOutputRequantizePass:
        """Create the fold pass with fake schema identities."""

        return FuseOutputRequantizePass(
            builtin_codes=BUILTIN_CODES,
            object_factory=optimization_object_factory,
        )

    def _run(self, document):
        """Run the pass without per-pass verification."""

        return self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

    def _build_conv_quantize(
        self,
        *,
        producer_code: int = CONV_2D,
        target_scale: tuple[float, ...] = (0.1,),
        target_zero_point: tuple[int, ...] = (128,),
    ):
        """Create input -> producer -> QUANTIZE -> output with one consumer."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = builder.add_tensor(
            "input",
            quantized_contract((1, 4, 4, 2), INT16, scale=(0.001,)),
        )
        document.subgraph().inputs = [source]
        weight = add_f32(builder, "weight", 1.0)
        mid = builder.add_operator(
            producer_code,
            inputs=(source, weight),
            output_contracts=(quantized_contract((1, 4, 4, 2), INT16, scale=(0.002,)),),
            output_names=("mid",),
        )[0]
        target = builder.add_operator(
            QUANTIZE,
            inputs=(mid,),
            output_contracts=(
                quantized_contract(
                    (1, 4, 4, 2),
                    UINT8,
                    scale=target_scale,
                    zero_point=target_zero_point,
                ),
            ),
            output_names=("target",),
        )[0]
        document.subgraph().outputs = [target]
        return document, mid, target

    def test_folds_quantize_into_conv_output(self) -> None:
        """Retarget the conv output to the QUANTIZE target and drop QUANTIZE."""

        document, mid, target = self._build_conv_quantize()

        result = self._run(document)

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(len(operators), 1)
        self.assertEqual(
            operator_builtin_code(document.model, operators[0]),
            CONV_2D,
        )
        self.assertEqual(as_indices(operators[0].outputs), [target])
        self.assertNotIn(mid, as_indices(operators[0].outputs))
        self.assertEqual(document.subgraph().outputs, [target])

    def test_folds_quantize_into_depthwise_output(self) -> None:
        """Accept every declared requantizing producer opcode."""

        document, _, target = self._build_conv_quantize(
            producer_code=DEPTHWISE_CONV_2D,
        )

        result = self._run(document)

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(len(operators), 1)
        self.assertEqual(as_indices(operators[0].outputs), [target])

    def test_keeps_quantize_after_non_requantizing_producer(self) -> None:
        """Keep QUANTIZE when the producer has no free output requantization."""

        document, _, _ = self._build_conv_quantize(producer_code=ADD)

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_keeps_quantize_when_source_has_second_consumer(self) -> None:
        """Keep QUANTIZE when the producer output feeds another operator."""

        document, mid, _ = self._build_conv_quantize()
        builder = make_builder(document, self.codec)
        builder.add_operator(
            QUANTIZE,
            inputs=(mid,),
            output_contracts=(quantized_contract((1, 4, 4, 2), UINT8, scale=(0.2,)),),
            output_names=("second",),
        )

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 3)

    def test_keeps_quantize_when_source_is_graph_output(self) -> None:
        """Keep QUANTIZE when the producer output is itself a graph output."""

        document, mid, target = self._build_conv_quantize()
        document.subgraph().outputs = [mid, target]

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_keeps_quantize_with_per_channel_target(self) -> None:
        """Keep QUANTIZE when the target is not per-tensor quantized."""

        document, _, _ = self._build_conv_quantize(
            target_scale=(0.1, 0.2),
            target_zero_point=(128, 128),
        )

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_keeps_quantize_reading_graph_input(self) -> None:
        """Keep QUANTIZE when its source tensor has no producer."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = builder.add_tensor(
            "input",
            quantized_contract((1, 4, 4, 2), INT16, scale=(0.001,)),
        )
        document.subgraph().inputs = [source]
        target = builder.add_operator(
            QUANTIZE,
            inputs=(source,),
            output_contracts=(quantized_contract((1, 4, 4, 2), UINT8, scale=(0.1,)),),
            output_names=("target",),
        )[0]
        document.subgraph().outputs = [target]

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)


if __name__ == "__main__":
    unittest.main()
