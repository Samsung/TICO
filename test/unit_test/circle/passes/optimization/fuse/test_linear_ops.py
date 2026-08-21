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

from __future__ import annotations

import unittest

from dataclasses import dataclass

import numpy as np

from tico.circle.passes.base import CirclePassContext
from tico.circle.passes.cleanup import DeadCodeEliminationPass
from tico.circle.passes.optimization._utils import operator_builtin_code
from tico.circle.passes.optimization.fuse.linear import (
    FuseLinearOpsPass,
    LinearFusionPolicy,
)

from test.unit_test.circle.infrastructure_fixture import (
    add_runtime_tensor,
    fake_object_factory,
    FLOAT32,
    make_empty_document,
)
from test.unit_test.circle.passes.optimization._fixture import (
    ADD,
    add_f32,
    add_i32,
    BinaryOptions,
    make_builder,
    make_codec,
    MUL,
    static_contract,
)

SUB = 30
CONV_2D = 31
DEPTHWISE_CONV_2D = 32
FULLY_CONNECTED = 33
TRANSPOSE_CONV = 34

BUILTIN_CODES = {
    "ADD": ADD,
    "MUL": MUL,
    "SUB": SUB,
    "CONV_2D": CONV_2D,
    "DEPTHWISE_CONV_2D": DEPTHWISE_CONV_2D,
    "FULLY_CONNECTED": FULLY_CONNECTED,
    "TRANSPOSE_CONV": TRANSPOSE_CONV,
}
TENSOR_TYPES = {"FLOAT32": FLOAT32}

ADD_OPTIONS = 40
MUL_OPTIONS = 41
SUB_OPTIONS = 42
CONV_2D_OPTIONS = 43
DEPTHWISE_CONV_2D_OPTIONS = 44
FULLY_CONNECTED_OPTIONS = 45
TRANSPOSE_CONV_OPTIONS = 46

BUILTIN_OPTIONS_TYPES = {
    "AddOptions": ADD_OPTIONS,
    "MulOptions": MUL_OPTIONS,
    "SubOptions": SUB_OPTIONS,
    "Conv2DOptions": CONV_2D_OPTIONS,
    "DepthwiseConv2DOptions": DEPTHWISE_CONV_2D_OPTIONS,
    "FullyConnectedOptions": FULLY_CONNECTED_OPTIONS,
    "TransposeConvOptions": TRANSPOSE_CONV_OPTIONS,
}


@dataclass
class LinearOptions:
    """Provide fused activation and depth multiplier fields for linear tests."""

    fusedActivationFunction: int = 0
    depthMultiplier: int = 1
    keepNumDims: bool = False


class FuseLinearOpsPassTest(unittest.TestCase):
    """Check FLOAT32 linear parameter fusion and conservative rejection paths."""

    def setUp(self) -> None:
        """Create one schema-independent constant codec."""

        self.codec = make_codec()
        self.context = CirclePassContext(verify_after_each_pass=False)

    def _pass(
        self,
        *,
        policy: LinearFusionPolicy | None = None,
    ) -> FuseLinearOpsPass:
        """Create the linear-fusion pass with fake schema identities."""

        return FuseLinearOpsPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            activation_none=0,
            codec=self.codec,
            object_factory=fake_object_factory,
            policy=policy,
        )

    def _decode(self, document, tensor_index: int) -> np.ndarray:
        """Decode one FLOAT32 fixture constant."""

        return self.codec.decode_tensor(
            document.model,
            subgraph_index=0,
            tensor_index=tensor_index,
        ).data

    def _fc(
        self,
        builder,
        source: int,
        weight: np.ndarray,
        bias: np.ndarray | None,
        *,
        output_name: str = "fc_output",
        activation: int = 0,
    ) -> int:
        """Append one fixture FULLY_CONNECTED operator."""

        weight_tensor = add_f32(builder, f"{output_name}_weight", weight)
        inputs = [source, weight_tensor]
        if bias is not None:
            inputs.append(add_f32(builder, f"{output_name}_bias", bias))
        else:
            inputs.append(-1)
        return builder.add_operator(
            FULLY_CONNECTED,
            inputs=inputs,
            output_contracts=(static_contract((1, int(weight.shape[0]))),),
            output_names=(output_name,),
            builtin_options_type=FULLY_CONNECTED_OPTIONS,
            builtin_options=LinearOptions(
                fusedActivationFunction=activation,
            ),
        )[0]

    def _conv(
        self,
        builder,
        source: int,
        weight: np.ndarray,
        bias: np.ndarray | None,
        *,
        builtin_code: int = CONV_2D,
        output_name: str = "conv_output",
        depth_multiplier: int = 1,
    ) -> int:
        """Append Conv2D or DepthwiseConv2D with static NHWC contracts."""

        weight_tensor = add_f32(builder, f"{output_name}_weight", weight)
        inputs = [source, weight_tensor]
        inputs.append(
            -1 if bias is None else add_f32(builder, f"{output_name}_bias", bias)
        )
        output_channels = int(
            weight.shape[3] if builtin_code == DEPTHWISE_CONV_2D else weight.shape[0]
        )
        options_type = (
            DEPTHWISE_CONV_2D_OPTIONS
            if builtin_code == DEPTHWISE_CONV_2D
            else CONV_2D_OPTIONS
        )
        return builder.add_operator(
            builtin_code,
            inputs=inputs,
            output_contracts=(static_contract((1, 1, 1, output_channels)),),
            output_names=(output_name,),
            builtin_options_type=options_type,
            builtin_options=LinearOptions(depthMultiplier=depth_multiplier),
        )[0]

    def _transpose_conv(
        self,
        builder,
        source: int,
        weight: np.ndarray,
        bias: np.ndarray | None,
    ) -> int:
        """Append one TransposeConv with its output-shape input."""

        output_shape = add_i32(
            builder,
            "output_shape",
            [1, 1, 1, int(weight.shape[0])],
        )
        weight_tensor = add_f32(builder, "tconv_weight", weight)
        inputs = [output_shape, weight_tensor, source]
        inputs.append(-1 if bias is None else add_f32(builder, "tconv_bias", bias))
        return builder.add_operator(
            TRANSPOSE_CONV,
            inputs=inputs,
            output_contracts=(static_contract((1, 1, 1, int(weight.shape[0]))),),
            output_names=("tconv_output",),
            builtin_options_type=TRANSPOSE_CONV_OPTIONS,
            builtin_options=LinearOptions(),
        )[0]

    def _binary(
        self,
        builder,
        builtin_code: int,
        left: int,
        right: int,
        shape: tuple[int, ...],
        *,
        name: str,
        activation: int = 0,
    ) -> int:
        """Append one binary affine operator."""

        options_type = {
            ADD: ADD_OPTIONS,
            MUL: MUL_OPTIONS,
            SUB: SUB_OPTIONS,
        }[builtin_code]
        return builder.add_operator(
            builtin_code,
            inputs=(left, right),
            output_contracts=(static_contract(shape),),
            output_names=(name,),
            builtin_options_type=options_type,
            builtin_options=BinaryOptions(fusedActivationFunction=activation),
        )[0]

    def test_post_add_into_fc_replaces_only_anchor_until_external_dce(self) -> None:
        """Fold post-FC ADD into bias while leaving the old FC for external DCE."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2],
        )
        document.subgraph().inputs = [source]
        fc = self._fc(
            builder,
            source,
            np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
            np.array([0.5, -0.5], np.float32),
        )
        offset = add_f32(builder, "offset", [1.0, 2.0])
        output = self._binary(
            builder,
            ADD,
            fc,
            offset,
            (1, 2),
            name="output",
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)
        fused = document.subgraph().operators[1]
        self.assertEqual(operator_builtin_code(document.model, fused), FULLY_CONNECTED)
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[2]),
            np.array([1.5, 1.5], np.float32),
        )
        self.assertEqual(fused.outputs, [output])

        DeadCodeEliminationPass().run(document, self.context)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_post_mul_into_fc_scales_weight_and_bias(self) -> None:
        """Scale FC output channels by rewriting both weight rows and bias."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(document, subgraph_index=0, name="x", shape=[1, 2])
        fc = self._fc(
            builder,
            source,
            np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
            np.array([0.5, -0.5], np.float32),
        )
        scale = add_f32(builder, "scale", [2.0, 3.0])
        output = self._binary(builder, MUL, scale, fc, (1, 2), name="output")
        document.subgraph().outputs = [output]

        self._pass().run(document, self.context)

        fused = document.subgraph().operators[1]
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[1]),
            np.array([[2.0, 4.0], [9.0, 12.0]], np.float32),
        )
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[2]),
            np.array([1.0, -1.5], np.float32),
        )

    def test_post_add_supports_conv_depthwise_and_transpose_conv(self) -> None:
        """Fold channel offsets into all supported convolution bias layouts."""

        cases = (
            (
                CONV_2D,
                np.array([[[[1.0]]], [[[2.0]]]], np.float32),
                1,
            ),
            (
                DEPTHWISE_CONV_2D,
                np.array([[[[1.0, 2.0]]]], np.float32),
                2,
            ),
            (
                TRANSPOSE_CONV,
                np.array([[[[1.0]]], [[[2.0]]]], np.float32),
                1,
            ),
        )
        for builtin_code, weight, multiplier in cases:
            with self.subTest(builtin_code=builtin_code):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="x",
                    shape=[1, 1, 1, 1],
                )
                if builtin_code == TRANSPOSE_CONV:
                    linear = self._transpose_conv(
                        builder,
                        source,
                        weight,
                        np.array([0.5, -0.5], np.float32),
                    )
                else:
                    linear = self._conv(
                        builder,
                        source,
                        weight,
                        np.array([0.5, -0.5], np.float32),
                        builtin_code=builtin_code,
                        depth_multiplier=multiplier,
                    )
                offset = add_f32(builder, "offset", [1.0, 2.0])
                output = self._binary(
                    builder,
                    ADD,
                    linear,
                    offset,
                    (1, 1, 1, 2),
                    name="output",
                )
                document.subgraph().outputs = [output]

                self._pass().run(document, self.context)

                fused = document.subgraph().operators[1]
                bias_position = 3 if builtin_code == TRANSPOSE_CONV else 2
                np.testing.assert_allclose(
                    self._decode(document, fused.inputs[bias_position]),
                    np.array([1.5, 1.5], np.float32),
                )

    def test_post_mul_scales_convolution_output_channel_axes(self) -> None:
        """Scale Conv, DepthwiseConv, and TransposeConv output-channel weights."""

        cases = (
            (
                CONV_2D,
                np.array([[[[1.0]]], [[[2.0]]]], np.float32),
                np.array([[[[2.0]]], [[[6.0]]]], np.float32),
                1,
            ),
            (
                DEPTHWISE_CONV_2D,
                np.array([[[[1.0, 2.0]]]], np.float32),
                np.array([[[[2.0, 6.0]]]], np.float32),
                2,
            ),
            (
                TRANSPOSE_CONV,
                np.array([[[[1.0]]], [[[2.0]]]], np.float32),
                np.array([[[[2.0]]], [[[6.0]]]], np.float32),
                1,
            ),
        )
        for builtin_code, weight, expected_weight, multiplier in cases:
            with self.subTest(builtin_code=builtin_code):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="x",
                    shape=[1, 1, 1, 1],
                )
                if builtin_code == TRANSPOSE_CONV:
                    linear = self._transpose_conv(
                        builder,
                        source,
                        weight,
                        np.array([0.5, -0.5], np.float32),
                    )
                else:
                    linear = self._conv(
                        builder,
                        source,
                        weight,
                        np.array([0.5, -0.5], np.float32),
                        builtin_code=builtin_code,
                        depth_multiplier=multiplier,
                    )
                scale = add_f32(builder, "scale", [2.0, 3.0])
                output = self._binary(
                    builder,
                    MUL,
                    linear,
                    scale,
                    (1, 1, 1, 2),
                    name="output",
                )
                document.subgraph().outputs = [output]

                result = self._pass().run(document, self.context)

                self.assertTrue(result.modified)
                fused = document.subgraph().operators[1]
                weight_position = 1
                bias_position = 3 if builtin_code == TRANSPOSE_CONV else 2
                np.testing.assert_allclose(
                    self._decode(document, fused.inputs[weight_position]),
                    expected_weight,
                )
                np.testing.assert_allclose(
                    self._decode(document, fused.inputs[bias_position]),
                    np.array([1.0, -1.5], np.float32),
                )

    def test_decomposed_batch_norm_folds_to_one_conv_after_external_dce(self) -> None:
        """Absorb SUB, MUL, and ADD in restart order as decomposed BatchNorm."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 1, 1, 1],
        )
        conv = self._conv(
            builder,
            source,
            np.array([[[[1.0]]], [[[2.0]]]], np.float32),
            np.array([0.5, -0.5], np.float32),
        )
        mean = add_f32(builder, "mean", [0.25, 0.5])
        centered = self._binary(
            builder,
            SUB,
            conv,
            mean,
            (1, 1, 1, 2),
            name="centered",
        )
        scale = add_f32(builder, "scale", [2.0, 3.0])
        scaled = self._binary(
            builder,
            MUL,
            centered,
            scale,
            (1, 1, 1, 2),
            name="scaled",
        )
        beta = add_f32(builder, "beta", [1.0, -1.0])
        output = self._binary(
            builder,
            ADD,
            scaled,
            beta,
            (1, 1, 1, 2),
            name="output",
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertEqual(result.changes, 7)
        fused = document.subgraph().operators[-1]
        self.assertEqual(operator_builtin_code(document.model, fused), CONV_2D)
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[1]),
            np.array([[[[2.0]]], [[[6.0]]]], np.float32),
        )
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[2]),
            np.array([1.5, -4.0], np.float32),
        )
        self.assertEqual(len(document.subgraph().operators), 4)
        DeadCodeEliminationPass().run(document, self.context)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_pre_add_into_fc_updates_bias_and_bypasses_affine_input(self) -> None:
        """Fold FC(x + c) by adding Wc to the bias."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(document, subgraph_index=0, name="x", shape=[1, 2])
        offset = add_f32(builder, "offset", [1.0, 2.0])
        shifted = self._binary(builder, ADD, source, offset, (1, 2), name="shifted")
        output = self._fc(
            builder,
            shifted,
            np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
            np.array([0.5, -0.5], np.float32),
            output_name="output",
        )
        document.subgraph().outputs = [output]

        self._pass().run(document, self.context)

        fused = document.subgraph().operators[1]
        self.assertEqual(fused.inputs[0], source)
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[2]),
            np.array([5.5, 10.5], np.float32),
        )
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_pre_mul_into_fc_scales_input_columns(self) -> None:
        """Fold FC(x * c) by scaling FC weight columns."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(document, subgraph_index=0, name="x", shape=[1, 2])
        scale = add_f32(builder, "scale", [2.0, 3.0])
        scaled = self._binary(builder, MUL, scale, source, (1, 2), name="scaled")
        output = self._fc(
            builder,
            scaled,
            np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
            np.array([0.5, -0.5], np.float32),
            output_name="output",
        )
        document.subgraph().outputs = [output]

        self._pass().run(document, self.context)

        fused = document.subgraph().operators[1]
        self.assertEqual(fused.inputs[0], source)
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[1]),
            np.array([[2.0, 6.0], [6.0, 12.0]], np.float32),
        )
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[2]),
            np.array([0.5, -0.5], np.float32),
        )

    def test_horizontal_fc_sum_combines_weights_and_biases(self) -> None:
        """Replace ADD(FC1(x), FC2(x)) with one FC using summed parameters."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(document, subgraph_index=0, name="x", shape=[1, 2])
        left = self._fc(
            builder,
            source,
            np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
            np.array([0.5, -0.5], np.float32),
            output_name="left",
        )
        right = self._fc(
            builder,
            source,
            np.array([[5.0, 6.0], [7.0, 8.0]], np.float32),
            np.array([1.0, 2.0], np.float32),
            output_name="right",
        )
        output = self._binary(builder, ADD, left, right, (1, 2), name="output")
        document.subgraph().outputs = [output]

        self._pass().run(document, self.context)

        fused = document.subgraph().operators[2]
        self.assertEqual(operator_builtin_code(document.model, fused), FULLY_CONNECTED)
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[1]),
            np.array([[6.0, 8.0], [10.0, 12.0]], np.float32),
        )
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[2]),
            np.array([1.5, 1.5], np.float32),
        )
        self.assertEqual(len(document.subgraph().operators), 3)
        DeadCodeEliminationPass().run(document, self.context)
        self.assertEqual(len(document.subgraph().operators), 1)

    def test_linear_or_affine_activation_blocks_post_fusion(self) -> None:
        """Keep post-affine graphs when either activation is observable."""

        for linear_activation, affine_activation in ((1, 0), (0, 1)):
            with self.subTest(
                linear_activation=linear_activation,
                affine_activation=affine_activation,
            ):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="x",
                    shape=[1, 2],
                )
                fc = self._fc(
                    builder,
                    source,
                    np.eye(2, dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                    activation=linear_activation,
                )
                offset = add_f32(builder, "offset", [1.0, 2.0])
                output = self._binary(
                    builder,
                    ADD,
                    fc,
                    offset,
                    (1, 2),
                    name="output",
                    activation=affine_activation,
                )
                document.subgraph().outputs = [output]

                result = self._pass().run(document, self.context)

                self.assertFalse(result.modified)

    def test_non_channel_broadcast_and_dead_branch_are_preserved(self) -> None:
        """Reject spatial constants and avoid rewriting unreachable affine branches."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 1, 1, 1],
        )
        conv = self._conv(
            builder,
            source,
            np.array([[[[1.0]]]], np.float32),
            np.array([0.0], np.float32),
        )
        spatial = add_f32(builder, "spatial", np.ones((1, 2, 1, 1), np.float32))
        dead = self._binary(
            builder,
            ADD,
            conv,
            spatial,
            (1, 2, 1, 1),
            name="dead",
        )
        document.subgraph().outputs = [conv]

        result = self._pass().run(document, self.context)

        self.assertFalse(result.modified)
        self.assertEqual(document.subgraph().operators[1].outputs, [dead])

    def test_post_add_synthesizes_fc_bias_when_bias_is_absent(self) -> None:
        """Create a new FC bias for channel addition when the bias is omitted."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 2],
        )
        fc = self._fc(
            builder,
            source,
            np.eye(2, dtype=np.float32),
            None,
        )
        offset = add_f32(builder, "offset", [1.0, 2.0])
        output = self._binary(
            builder,
            ADD,
            fc,
            offset,
            (1, 2),
            name="output",
        )
        document.subgraph().outputs = [output]

        self._pass().run(document, self.context)

        fused = document.subgraph().operators[1]
        self.assertGreaterEqual(fused.inputs[2], 0)
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[2]),
            np.array([1.0, 2.0], np.float32),
        )

    def test_post_mul_without_bias_keeps_optional_bias_omitted(self) -> None:
        """Scale weights without manufacturing a zero bias for post multiplication."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 2],
        )
        fc = self._fc(
            builder,
            source,
            np.eye(2, dtype=np.float32),
            None,
        )
        scale = add_f32(builder, "scale", [2.0, 3.0])
        output = self._binary(
            builder,
            MUL,
            fc,
            scale,
            (1, 2),
            name="output",
        )
        document.subgraph().outputs = [output]

        self._pass().run(document, self.context)

        fused = document.subgraph().operators[1]
        self.assertEqual(fused.inputs[2], -1)
        np.testing.assert_allclose(
            self._decode(document, fused.inputs[1]),
            np.diag(np.array([2.0, 3.0], np.float32)),
        )

    def test_transpose_conv_without_bias_capable_version_is_preserved(self) -> None:
        """Avoid adding a bias input to an old TransposeConv operator version."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 1, 1, 1],
        )
        tconv = self._transpose_conv(
            builder,
            source,
            np.array([[[[1.0]]]], np.float32),
            None,
        )
        offset = add_f32(builder, "offset", [1.0])
        output = self._binary(
            builder,
            ADD,
            tconv,
            offset,
            (1, 1, 1, 1),
            name="output",
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertFalse(result.modified)

    def test_pre_affine_preserves_fc_fused_activation(self) -> None:
        """Allow pre-FC fusion because the FC post-activation remains in place."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 2],
        )
        offset = add_f32(builder, "offset", [1.0, 2.0])
        shifted = self._binary(
            builder,
            ADD,
            source,
            offset,
            (1, 2),
            name="shifted",
        )
        output = self._fc(
            builder,
            shifted,
            np.eye(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            output_name="output",
            activation=1,
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertTrue(result.modified)
        fused = document.subgraph().operators[1]
        self.assertEqual(fused.inputs[0], source)
        self.assertEqual(fused.builtinOptions.fusedActivationFunction, 1)

    def test_post_fusion_preserves_original_linear_with_live_fanout(self) -> None:
        """Keep the original FC when another live operator still consumes its output."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 2],
        )
        fc = self._fc(
            builder,
            source,
            np.eye(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
        offset = add_f32(builder, "offset", [1.0, 2.0])
        fused_output = self._binary(
            builder,
            ADD,
            fc,
            offset,
            (1, 2),
            name="fused_output",
        )
        side_output = builder.add_operator(
            99,
            inputs=(fc,),
            output_contracts=(static_contract((1, 2)),),
            output_names=("side_output",),
        )[0]
        document.subgraph().outputs = [fused_output, side_output]

        self._pass().run(document, self.context)
        DeadCodeEliminationPass().run(document, self.context)

        codes = [
            operator_builtin_code(document.model, operator)
            for operator in document.subgraph().operators
        ]
        self.assertEqual(codes.count(FULLY_CONNECTED), 2)
        self.assertIn(99, codes)

    def test_horizontal_fc_requires_matching_options(self) -> None:
        """Keep horizontal FC branches whose serialized options differ."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 2],
        )
        left = self._fc(
            builder,
            source,
            np.eye(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            output_name="left",
        )
        right = self._fc(
            builder,
            source,
            np.eye(2, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            output_name="right",
        )
        document.subgraph().operators[1].builtinOptions.keepNumDims = True
        output = self._binary(
            builder,
            ADD,
            left,
            right,
            (1, 2),
            name="output",
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertFalse(result.modified)

    def test_wrong_builtin_options_type_is_preserved(self) -> None:
        """Reject linear and affine operators carrying the wrong options union."""

        cases = ("linear", "affine")
        for mismatch in cases:
            with self.subTest(mismatch=mismatch):
                document = make_empty_document()
                builder = make_builder(document, self.codec)
                source = add_runtime_tensor(
                    document,
                    subgraph_index=0,
                    name="source",
                    shape=[1, 2],
                )
                document.subgraph().inputs = [source]
                fc = self._fc(
                    builder,
                    source,
                    np.eye(2, dtype=np.float32),
                    np.zeros(2, dtype=np.float32),
                )
                offset = add_f32(builder, "offset", [1.0, 2.0])
                output = self._binary(
                    builder,
                    ADD,
                    fc,
                    offset,
                    (1, 2),
                    name="output",
                )
                operators = document.subgraph().operators
                operators[0 if mismatch == "linear" else 1].builtinOptionsType = 999
                document.subgraph().outputs = [output]

                result = self._pass().run(document, self.context)

                self.assertFalse(result.modified)
                self.assertEqual(len(document.subgraph().operators), 2)

    def test_nonfinite_parameter_is_preserved(self) -> None:
        """Reject fusion when a source parameter contains a non-finite value."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 2],
        )
        fc = self._fc(
            builder,
            source,
            np.array([[np.inf, 0.0], [0.0, 1.0]], np.float32),
            np.zeros(2, dtype=np.float32),
        )
        offset = add_f32(builder, "offset", [1.0, 2.0])
        output = self._binary(
            builder,
            ADD,
            fc,
            offset,
            (1, 2),
            name="output",
        )
        document.subgraph().outputs = [output]

        result = self._pass().run(document, self.context)

        self.assertFalse(result.modified)

    def test_apply_rolls_back_new_constants_when_allocation_fails(self) -> None:
        """Restore model sizes when the second replacement constant cannot be built."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="x",
            shape=[1, 2],
        )
        fc = self._fc(
            builder,
            source,
            np.eye(2, dtype=np.float32),
            np.array([1.5, 2.5], np.float32),
        )
        scale = add_f32(builder, "scale", [2.0, 3.0])
        output = self._binary(
            builder,
            MUL,
            fc,
            scale,
            (1, 2),
            name="output",
        )
        document.subgraph().outputs = [output]
        before = (
            len(document.model.buffers),
            len(document.subgraph().tensors),
            tuple(document.subgraph().operators),
        )
        buffer_calls = 0

        def failing_factory(table_name: str):
            """Fail while allocating the second replacement buffer."""

            nonlocal buffer_calls
            if table_name == "Buffer":
                buffer_calls += 1
                if buffer_calls == 2:
                    raise RuntimeError("injected allocation failure")
            return fake_object_factory(table_name)

        fusion = FuseLinearOpsPass(
            builtin_codes=BUILTIN_CODES,
            builtin_options_types=BUILTIN_OPTIONS_TYPES,
            tensor_types=TENSOR_TYPES,
            activation_none=0,
            codec=self.codec,
            object_factory=failing_factory,
        )

        with self.assertRaisesRegex(RuntimeError, "injected allocation failure"):
            fusion.run(document, self.context)

        after = (
            len(document.model.buffers),
            len(document.subgraph().tensors),
            tuple(document.subgraph().operators),
        )
        self.assertEqual(after, before)

    def test_float_reassociation_can_be_disabled(self) -> None:
        """Keep an eligible graph when strict floating-point order is requested."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(
            document,
            subgraph_index=0,
            name="source",
            shape=[1, 2],
        )
        fc = self._fc(
            builder,
            source,
            np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
            np.array([0.5, -0.5], np.float32),
        )
        offset = add_f32(builder, "offset", [1.0, 2.0])
        output = self._binary(
            builder,
            ADD,
            fc,
            offset,
            (1, 2),
            name="output",
        )
        document.subgraph().outputs = [output]

        result = self._pass(
            policy=LinearFusionPolicy(allow_float_reassociation=False)
        ).run(document, self.context)

        self.assertFalse(result.modified)
        self.assertEqual(len(document.subgraph().operators), 2)

    def test_replacement_budget_is_checked_before_mutation(self) -> None:
        """Skip fusion when new parameters exceed the configured byte budget."""

        document = make_empty_document()
        builder = make_builder(document, self.codec)
        source = add_runtime_tensor(document, subgraph_index=0, name="x", shape=[1, 2])
        fc = self._fc(
            builder,
            source,
            np.eye(2, dtype=np.float32),
            np.ones(2, dtype=np.float32),
        )
        scale = add_f32(builder, "scale", [2.0, 3.0])
        output = self._binary(builder, MUL, fc, scale, (1, 2), name="output")
        document.subgraph().outputs = [output]
        before = (
            len(document.model.buffers),
            len(document.subgraph().tensors),
            len(document.subgraph().operators),
        )

        result = self._pass(policy=LinearFusionPolicy(maximum_replacement_bytes=4)).run(
            document, self.context
        )

        self.assertFalse(result.modified)
        after = (
            len(document.model.buffers),
            len(document.subgraph().tensors),
            len(document.subgraph().operators),
        )
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
