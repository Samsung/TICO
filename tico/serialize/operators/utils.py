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

from typing import Dict, List, Optional

from circle_schema import circle

from tico.serialize.operators.hashable_opcode import OpCode


def create_builtin_opcode(opcode: int, custom_code: Optional[str] = None) -> OpCode:
    op_code = OpCode()
    # deprecatedBuiltinCode is int8, so its maximum value is 127
    # (127 is reserved as a placeholder for greater opcodes)
    # opcode greater than 127 is saved in builtinCode
    op_code.deprecatedBuiltinCode = min(127, opcode)
    op_code.builtinCode = opcode
    op_code.version = 1
    if custom_code is not None:
        op_code.customCode = custom_code
    return op_code


def get_op_index(
    opcode: int, opcode_map: Dict[OpCode, int], custom_code: Optional[str] = None
) -> int:
    op_code = create_builtin_opcode(opcode, custom_code)
    if op_code not in opcode_map:
        op_index = len(opcode_map)
        opcode_map[op_code] = op_index
    else:
        op_index = opcode_map[op_code]
    return op_index


# TODO Move this to CircleSubGraph
def create_builtin_operator(
    graph, op_index: int, inputs: List, outputs: List
) -> circle.Operator.OperatorT:
    operator = circle.Operator.OperatorT()
    operator.opcodeIndex = op_index
    operator.inputs = [graph.get_tid(input) for input in inputs]
    operator.outputs = [graph.get_tid(output) for output in outputs]
    return operator
