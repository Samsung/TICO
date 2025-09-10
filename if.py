""" Example - circle model import/export """

import pycircle
from pycircle.circleir.model import Model
from pycircle.circleir.subgraph import Subgraph
from pycircle.circleir.tensor import Tensor
from pycircle.circleir.operators import CircleAdd, CircleIf
from pycircle.util.alias import TensorType

# 입력 텐서 및 상수 텐서 정의
input_tensor0 = Tensor("input0", [1, 3], TensorType.FLOAT32)
input_tensor1 = Tensor("input1", [1, 3], TensorType.FLOAT32)
weight_add_100 = Tensor("constant0", [1, 3], TensorType.FLOAT32, [100, 100, 100])
weight_sub_100 = Tensor("constant1", [1, 3], TensorType.FLOAT32, [-100, -100, -100])

### then_subgraph ###
then_subgraph = Subgraph()
then_subgraph.inputs = [Tensor("input0", [1, 3], TensorType.FLOAT32), weight_add_100]

add_op_then = CircleAdd()
add_op_then.inputs = [then_subgraph.inputs[0], then_subgraph.inputs[1]]
add_op_then.outputs(0).attribute("add_output_then", [1, 3], TensorType.FLOAT32)
then_subgraph.outputs = [add_op_then.outputs(0)]

### else_subgraph ###
else_subgraph = Subgraph()
else_subgraph.inputs = [Tensor("input0", [1, 3], TensorType.FLOAT32), weight_sub_100]

add_op_else = CircleAdd()
add_op_else.inputs = [else_subgraph.inputs[0], Tensor("input0", [1, 3], TensorType.FLOAT32)]
add_op_else.outputs(0).attribute("add_output_else", [1, 3], TensorType.FLOAT32)
else_subgraph.outputs = [add_op_else.outputs(0)]

### root_subgraph with CircleIf ###
root_subgraph = Subgraph()
root_subgraph.name = "root_subgraph"
condition_tensor = Tensor("condition", [1], TensorType.BOOL)
root_subgraph.inputs = [condition_tensor, input_tensor0, input_tensor1]

circle_if_op = CircleIf(1, 2)
circle_if_op.inputs = [condition_tensor, input_tensor0, input_tensor1]
circle_if_op.outputs(0).attribute("output_tensor", [1, 3], TensorType.FLOAT32)
circle_if_op.then_subgraph_index = 1
circle_if_op.else_subgraph_index = 2
root_subgraph.outputs = [circle_if_op.outputs(0)]

# 모델 구성
circle_model = Model()
circle_model.description = "pycircle example : signature_def"
circle_model.subgraphs = [root_subgraph, then_subgraph, else_subgraph]
circle_model.signature_defs = {
    "root_graph": {"subgraph_index": 0},
    "then_graph": {"subgraph_index": 1},
    "else_graph": {"subgraph_index": 2},
}

# 모델 export
pycircle.export_circle_model(circle_model, "signature_def.circle")

# onert를 통한 추론 (Inference)
import torch
try:
    from onert import infer
except ImportError:
    raise RuntimeError("The 'onert' package is required to run this function.")

session = infer.session("signature_def.circle")
output = session.infer(
    (
        torch.tensor([True]),               # condition tensor
        torch.randn(1, 3),                 # input tensor 0
        torch.tensor([[100., 100., 100.]]),# weights tensor
    ),
    measure=True
)
print(output)
