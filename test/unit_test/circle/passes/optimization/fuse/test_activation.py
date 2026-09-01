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
from dataclasses import dataclass

from tico.circle.analysis import TensorContract
from tico.circle.graph import as_indices
from tico.circle.passes import CirclePassContext, FuseActivationFunctionPass
from tico.circle.passes.optimization._utils import operator_builtin_code
from tico.circle.value import TensorQuantization

from test.unit_test.circle.infrastructure_fixture import make_empty_document
from test.unit_test.circle.passes.optimization._fixture import (
    add_f32,
    make_builder,
    make_codec,
    optimization_object_factory,
)

INT16 = 7
CONV_2D = 31
DEPTHWISE_CONV_2D = 32
RELU = 35
RELU6 = 36
ADD = 1
FUSED_NONE = 0
FUSED_RELU = 1
FUSED_RELU6 = 3

BUILTIN_CODES = {
    "CONV_2D": CONV_2D,
    "DEPTHWISE_CONV_2D": DEPTHWISE_CONV_2D,
    "RELU": RELU,
    "RELU6": RELU6,
    "ADD": ADD,
}


@dataclass
class ConvOptions:
    """Provide the fused-activation slot used by convolution producers."""

    fusedActivationFunction: int = FUSED_NONE


def quantized_contract(
    shape: tuple[int, ...],
    *,
    scale: tuple[float, ...] = (1.0,),
) -> TensorContract:
    """Create one static per-tensor quantized tensor contract."""

    return TensorContract(
        tensor_type=INT16,
        shape=shape,
        shape_signature=shape,
        quantization=TensorQuantization(
            scale=scale,
            zero_point=(0,) * len(scale),
        ),
    )


class FuseActivationFunctionTest(unittest.TestCase):
    """Check RELU/RELU6 folding into producer fused slots and its guards."""

    def setUp(self) -> None:
        """Create a schema-independent codec for each test."""

        self.codec = make_codec()

    def _pass(self) -> FuseActivationFunctionPass:
        """Create the fuse pass with fake schema identities."""

        return FuseActivationFunctionPass(
            builtin_codes=BUILTIN_CODES,
            object_factory=optimization_object_factory,
        )

    def _run(self, document):
        """Run the pass without per-pass verification."""

        return self._pass().run(
            document,
            CirclePassContext(verify_after_each_pass=False),
        )

    def _build_conv_activation(
        self,
        *,
        producer_code: int = CONV_2D,
        activation_code: int = RELU6,
        producer_fused: int = FUSED_NONE,
        target_scale: tuple[float, ...] = (0.1,),
    ):
        """Create input -> producer -> activation -> output with one consumer."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = builder.add_tensor(
            "input",
            quantized_contract((1, 4, 4, 2), scale=(0.001,)),
        )
        document.subgraph().inputs = [source]
        weight = add_f32(builder, "weight", 1.0)
        mid = builder.add_operator(
            producer_code,
            inputs=(source, weight),
            output_contracts=(quantized_contract((1, 4, 4, 2), scale=(0.002,)),),
            output_names=("mid",),
            builtin_options=ConvOptions(fusedActivationFunction=producer_fused),
        )[0]
        target = builder.add_operator(
            activation_code,
            inputs=(mid,),
            output_contracts=(
                quantized_contract((1, 4, 4, 2), scale=target_scale),
            ),
            output_names=("target",),
        )[0]
        document.subgraph().outputs = [target]
        return document, mid, target

    def test_fuses_relu6_into_conv(self) -> None:
        """Set the fused slot, retarget the output, and drop RELU6."""

        document, mid, target = self._build_conv_activation()

        result = self._run(document)

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(len(operators), 1)
        self.assertEqual(
            operator_builtin_code(document.model, operators[0]),
            CONV_2D,
        )
        self.assertEqual(
            operators[0].builtinOptions.fusedActivationFunction,
            FUSED_RELU6,
        )
        self.assertEqual(as_indices(operators[0].outputs), [target])
        self.assertNotIn(mid, as_indices(operators[0].outputs))
        self.assertEqual(document.subgraph().outputs, [target])

    def test_fuses_relu_into_depthwise(self) -> None:
        """Map RELU to its fused code on a depthwise producer."""

        document, _, target = self._build_conv_activation(
            producer_code=DEPTHWISE_CONV_2D,
            activation_code=RELU,
        )

        result = self._run(document)

        self.assertTrue(result.modified)
        operators = document.subgraph().operators
        self.assertEqual(len(operators), 1)
        self.assertEqual(
            operators[0].builtinOptions.fusedActivationFunction,
            FUSED_RELU,
        )
        self.assertEqual(as_indices(operators[0].outputs), [target])

    def test_keeps_activation_after_non_fusable_producer(self) -> None:
        """Keep RELU6 when the producer has no fused activation slot."""

        document, _, _ = self._build_conv_activation(producer_code=ADD)

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_keeps_activation_when_producer_already_fused(self) -> None:
        """Keep RELU6 when the producer already applies a fused activation."""

        document, _, _ = self._build_conv_activation(
            producer_fused=FUSED_RELU,
        )

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_keeps_activation_when_source_has_second_consumer(self) -> None:
        """Keep RELU6 when the producer output feeds another operator."""

        document, mid, _ = self._build_conv_activation()
        builder = make_builder(document, self.codec)
        builder.add_operator(
            RELU6,
            inputs=(mid,),
            output_contracts=(quantized_contract((1, 4, 4, 2), scale=(0.2,)),),
            output_names=("second",),
        )

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 3)

    def test_keeps_activation_when_source_is_graph_output(self) -> None:
        """Keep RELU6 when the pre-activation tensor is a graph output."""

        document, mid, target = self._build_conv_activation()
        document.subgraph().outputs = [mid, target]

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_keeps_activation_with_per_channel_target(self) -> None:
        """Keep RELU6 when the activation output is not per-tensor quantized."""

        document, _, _ = self._build_conv_activation(
            target_scale=(0.1, 0.2),
        )

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_keeps_activation_reading_graph_input(self) -> None:
        """Keep RELU6 when its source tensor has no producer."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = builder.add_tensor(
            "input",
            quantized_contract((1, 4, 4, 2), scale=(0.001,)),
        )
        document.subgraph().inputs = [source]
        target = builder.add_operator(
            RELU6,
            inputs=(source,),
            output_contracts=(quantized_contract((1, 4, 4, 2), scale=(0.1,)),),
            output_names=("target",),
        )[0]
        document.subgraph().outputs = [target]

        result = self._run(document)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 1)


if __name__ == "__main__":
    unittest.main()
