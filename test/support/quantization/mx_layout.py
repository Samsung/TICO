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

"""Small final-Circle checks for the explicit Qwen MX NPU deployment policy.

This is a test assertion, not a generic Circle capability rule or NPU execution.
Shape, permutation, index and split-size operands are metadata, not activation data.
"""

from collections import Counter

from circle_schema import circle


def assert_qwen_mx_layout(case, document, *, require_mx=True):
    """Check layout domains and retained MX arithmetic/conversion boundaries.

    Return operator counts to let individual regressions assert their own topology.
    Fully connected weights/bias and RMSNorm gamma are deliberately not treated as
    activation operands. Their distinct policy is checked where applicable below.
    """
    ops = circle.BuiltinOperator.BuiltinOperator
    types = circle.TensorType.TensorType
    # None denotes all inputs; otherwise specify activation-data input positions.
    layout_slots = {
        ops.RESHAPE: (0,),
        ops.TRANSPOSE: (0,),
        ops.SQUEEZE: (0,),
        ops.EXPAND_DIMS: (0,),
        ops.SLICE: (0,),
        ops.STRIDED_SLICE: (0,),
        ops.SPLIT: (1,),
        ops.SPLIT_V: (0,),
        ops.UNPACK: (0,),
        ops.PACK: None,
        ops.CONCATENATION: None,
        ops.TILE: (0,),
        ops.GATHER: (0,),
        ops.GATHER_ND: (0,),
    }
    arithmetic_slots = {
        ops.FULLY_CONNECTED: (0,),
        ops.BATCH_MATMUL: (0, 1),
        ops.ADD: (0, 1),
        ops.SUB: (0, 1),
        ops.MUL: (0, 1),
        ops.DIV: (0, 1),
        ops.SOFTMAX: (0,),
        ops.LOGISTIC: (0,),
        ops.RELU: (0,),
    }
    if hasattr(ops, "RMS_NORM"):
        arithmetic_slots[ops.RMS_NORM] = (0,)
    names = {value: key for key, value in vars(ops).items() if key.isupper()}
    counts: Counter[str] = Counter()
    for graph in document.model.subgraphs:
        for operator in graph.operators:
            code = document.model.operatorCodes[operator.opcodeIndex].builtinCode
            name = names.get(code, str(code))
            counts[name] += 1
            outputs = [int(index) for index in operator.outputs if index >= 0]
            if code in layout_slots:
                slots = layout_slots[code]
                inputs = (
                    list(operator.inputs)
                    if slots is None
                    else [operator.inputs[slot] for slot in slots]
                )
                for index in [*inputs, *outputs]:
                    if index < 0:
                        continue
                    tensor = graph.tensors[index]
                    case.assertIn(
                        tensor.type,
                        {types.FLOAT32, types.INT16, types.UINT8},
                        f"{name} activation layout tensor {tensor.name!r} has unsupported layout domain {tensor.type}",
                    )
            if code in arithmetic_slots:
                inputs = [int(operator.inputs[slot]) for slot in arithmetic_slots[code]]
                activation_types = [
                    graph.tensors[index].type for index in [*inputs, *outputs]
                ]
                if types.MXINT8 in activation_types:
                    case.assertEqual(
                        set(activation_types),
                        {types.MXINT8},
                        f"{name} activation domains",
                    )
                    counts["MX_ARITHMETIC"] += 1
                if code == ops.FULLY_CONNECTED:
                    # Weight and bias do not follow the activation domain.
                    for index in operator.inputs[1:]:
                        if index >= 0:
                            case.assertNotEqual(graph.tensors[index].type, types.MXINT8)
                if hasattr(ops, "RMS_NORM") and code == ops.RMS_NORM:
                    case.assertNotEqual(
                        graph.tensors[operator.inputs[1]].type, types.MXINT8
                    )
            if code == ops.QUANTIZE:
                source = graph.tensors[operator.inputs[0]].type
                target = graph.tensors[operator.outputs[0]].type
                if source == types.MXINT8 and target == types.INT16:
                    counts["MX_TO_AFFINE"] += 1
                if source == types.INT16 and target == types.MXINT8:
                    counts["AFFINE_TO_MX"] += 1
    if require_mx:
        case.assertGreater(counts["MX_ARITHMETIC"], 0, "MX arithmetic was lost")
        case.assertGreater(counts["MX_TO_AFFINE"], 0, "MX output boundary was lost")
        case.assertGreater(counts["AFFINE_TO_MX"], 0, "MX input boundary was lost")
    return counts
