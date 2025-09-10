""" Example - circle model import/export """

import pycircle

from pycircle.circleir.model import Model
from pycircle.circleir.subgraph import Subgraph
from pycircle.circleir.tensor import Tensor
from pycircle.circleir.operators import CircleAdd
from pycircle.util.alias import TensorType

subgraph1 = Subgraph()
subgraph1.name = "subgraph1"
subgraph1.inputs = [
    Tensor("subgraph1_input", [1, 3], TensorType.FLOAT32),
]

weights1 = Tensor("constant1", [1, 3], TensorType.FLOAT32, [0.1, 0.2, 0.3])

add1 = CircleAdd()
add1.inputs = [subgraph1.inputs[0], weights1]
add1.outputs(0).attribute("Add1", [1, 3], TensorType.FLOAT32)

subgraph1.outputs = [add1.outputs(0)]

subgraph2 = Subgraph()
subgraph2.name = "subgraph2"
subgraph2.inputs = [
    Tensor("subgraph2_input1", [1, 3], TensorType.FLOAT32),
    Tensor("subgraph2_input2", [1, 3], TensorType.FLOAT32),
]

add2 = CircleAdd()
add2.inputs = [subgraph2.inputs[0], subgraph2.inputs[1]]
add2.outputs(0).attribute("Add2", [1, 3], TensorType.FLOAT32)

subgraph2.outputs = [add2.outputs(0)]

circle_model = Model()
circle_model.description = "pycircle example : signature_def"
# circle_model.subgraphs = [subgraph2, subgraph1]
circle_model.subgraphs = [subgraph1, subgraph2]
circle_model.signature_defs = {
    "add_constant": {
        "subgraph_index": 0
    },
    "add_two_inputs": {
        "subgraph_index": 1
    },
}

pycircle.export_circle_model(circle_model, "signature_def_original.circle")

import torch
try:
    from onert import infer
except ImportError:
    raise RuntimeError("The 'onert' package is required to run this function.")

session_float = infer.session("signature_def_original.circle")
output = session_float.infer((torch.randn(1,3),))
breakpoint()