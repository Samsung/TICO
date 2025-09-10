""" Example - circle model import/export """

import pycircle

from pycircle.circleir.model import Model
from pycircle.circleir.subgraph import Subgraph
from pycircle.circleir.tensor import Tensor
from pycircle.circleir.operators import CircleAdd, CircleCall
from pycircle.util.alias import TensorType


### subgraph 0
### input0, input1 -> call0 (subgraph 1) -> tensor0
### tensor0, weights0 -> add0 ->  tensor1
graph0 = Subgraph()
graph0.name = "graph0"
graph0.inputs = [
    Tensor("sub1_input0", [1, 3], TensorType.FLOAT32),
    Tensor("sub1_input1", [1, 3], TensorType.FLOAT32),
]

call0 = CircleCall()
call0.inputs = [graph0.inputs[0], graph0.inputs[1]]
call0.subgraph = 1
call0.outputs(0).attribute("Call0", [1, 3], TensorType.FLOAT32)


add1 = CircleAdd()
weights0 = Tensor("weights0", [1, 3], TensorType.FLOAT32, [100., 100., 100.])
add1.inputs = [call0.outputs(0), weights0]
add1.outputs(0).attribute("add0", [1, 3], TensorType.FLOAT32)

graph0.outputs = [add1.outputs(0)]

### subgraph 1
### input0, input1 -> ADD -> output
graph1 = Subgraph()
graph1.name = "graph1"
graph1.inputs = [
    Tensor("input0", [1, 3], TensorType.FLOAT32),
    Tensor("input1", [1, 3], TensorType.FLOAT32, [-100., -100., -100.])
]
sub_add = CircleAdd()
sub_add.inputs = [graph1.inputs[0], graph1.inputs[1]]
sub_add.outputs(0).attribute("SubAdd", [1, 3], TensorType.FLOAT32)
graph1.outputs = [sub_add.outputs(0)]

### model
circle_model = Model()
circle_model.subgraphs = [graph0, graph1]
circle_model.signature_defs = {
    "graph0": {
        "subgraph_index": 0
    },
    "graph1": {
        "subgraph_index": 1
    },
}

pycircle.export_circle_model(circle_model, "call.circle")

import torch
try:
    from onert import infer
except ImportError:
    raise RuntimeError("The 'onert' package is required to run this function.")

session_float = infer.session("call.circle")
output = session_float.infer((torch.randn(1,3),torch.randn(1,3),), measure=True)
print(output)