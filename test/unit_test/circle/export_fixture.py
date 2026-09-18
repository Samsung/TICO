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

"""Small, schema-independent graphs for export-boundary regression tests."""

from types import SimpleNamespace as NS

import numpy as np

from tico.circle.document import CircleDocument
from tico.circle.value import TensorTypeRegistry, TensorTypeSpec, TensorValueCodec

# Explicit test enums; production passes accept these through their injectable APIs.
CODES = {
    name: index + 1
    for index, name in enumerate(
        (
            "ADD",
            "MUL",
            "CAST",
            "SLICE",
            "STRIDED_SLICE",
            "SPLIT",
            "SPLIT_V",
            "RESHAPE",
            "TRANSPOSE",
            "NEG",
            "MEAN",
        )
    )
}
TYPES = {"FLOAT32": 0, "FLOAT16": 1, "INT32": 2, "UINT8": 3, "INT8": 9, "FLOAT64": 10}
OPTIONS = {"ReshapeOptions": 1, "TransposeOptions": 2}


def codec():
    """Use explicit dense types without consulting the generated schema."""
    specs = []
    for name, dtype in (
        ("FLOAT32", np.float32),
        ("FLOAT16", np.float16),
        ("FLOAT64", np.float64),
        ("INT32", np.int32),
        ("UINT8", np.uint8),
        ("INT8", np.int8),
    ):
        dt = np.dtype(dtype)
        specs.append(TensorTypeSpec(name, TYPES[name], dt, dt, dt.itemsize * 8))
    return TensorValueCodec(TensorTypeRegistry(specs))


def factory(name):
    """Provide Object API field defaults used by builders and compaction."""
    if name == "Buffer":
        return NS(data=None, offset=0, size=0)
    if name == "Operator":
        return NS(
            opcodeIndex=0,
            inputs=[],
            outputs=[],
            intermediates=[],
            mutatingVariableInputs=[],
            builtinOptionsType=0,
            builtinOptions=None,
            builtinOptions2Type=0,
            builtinOptions2=None,
            customOptions=None,
            customOptionsFormat=0,
            largeCustomOptionsOffset=0,
            largeCustomOptionsSize=0,
        )
    if name == "OperatorCode":
        return NS(builtinCode=0, deprecatedBuiltinCode=0, customCode=None, version=1)
    if name == "Tensor":
        return tensor("")
    if name == "QuantizationParameters":
        return NS(
            scale=[],
            zeroPoint=[],
            min=[],
            max=[],
            quantizedDimension=0,
            detailsType=0,
            details=None,
        )
    if name == "ReshapeOptions":
        return NS(newShape=[])
    raise ValueError(name)


def tensor(name, shape=(2, 3), tensor_type=0, buffer=0):
    """Create one static runtime or constant tensor."""
    return NS(
        name=name,
        shape=list(shape),
        shapeSignature=list(shape),
        type=tensor_type,
        buffer=buffer,
        quantization=None,
        isVariable=False,
        sparsity=None,
        hasRank=True,
        variantTensors=None,
    )


def operator(document, name, inputs, outputs, **options):
    """Append a plain operator with activation-free default options."""
    code = factory("OperatorCode")
    code.builtinCode = CODES[name]
    code.deprecatedBuiltinCode = CODES[name]
    document.model.operatorCodes.append(code)
    op = factory("Operator")
    op.opcodeIndex = len(document.model.operatorCodes) - 1
    op.inputs, op.outputs = list(inputs), list(outputs)
    op.builtinOptions = NS(fusedActivationFunction=0, **options)
    if name == "RESHAPE":
        op.builtinOptionsType = OPTIONS["ReshapeOptions"]
    document.subgraph().operators.append(op)
    return op


def empty_document():
    """Create a statically valid empty model with reserved buffer zero."""
    return CircleDocument(
        NS(
            subgraphs=[
                NS(name="main", tensors=[], inputs=[], outputs=[], operators=[])
            ],
            buffers=[factory("Buffer")],
            operatorCodes=[],
            signatureDefs=[],
            metadataBuffer=[],
            metadata=[],
            version=0,
            description="export-test",
        )
    )


def constant(document, name, values, dtype=np.float32, tensor_type=0):
    """Append a dense constant and return its tensor index."""
    array = np.asarray(values, dtype=dtype)
    buf = factory("Buffer")
    buf.data = np.frombuffer(array.tobytes(), dtype=np.uint8).copy()
    document.model.buffers.append(buf)
    index = len(document.subgraph().tensors)
    document.subgraph().tensors.append(
        tensor(name, array.shape, tensor_type, len(document.model.buffers) - 1)
    )
    return index


def mul_document(
    value=1.0,
    *,
    dtype=np.float32,
    tensor_type=0,
    public_mul=False,
    constant_shape=(),
    reverse=False,
):
    """Build x -> MUL(one) -> NEG with an optional public MUL result."""
    document = empty_document()
    graph = document.subgraph()
    graph.tensors = [
        tensor("x", tensor_type=tensor_type),
        tensor("product", tensor_type=tensor_type),
        tensor("output", tensor_type=tensor_type),
    ]
    one = constant(
        document, "factor", np.full(constant_shape, value), dtype, tensor_type
    )
    graph.inputs, graph.outputs = [0], [1] if public_mul else [2]
    operator(document, "MUL", [one, 0] if reverse else [0, one], [1])
    if not public_mul:
        operator(document, "NEG", [1], [2])
    return document


def add_signature(document, *, outputs=None):
    """Bind all input and selected output ports to a named signature."""
    graph = document.subgraph()
    selected_outputs = graph.outputs if outputs is None else outputs
    signature = NS(
        signatureKey="serving_default",
        subgraphIndex=0,
        inputs=[
            NS(name=f"input_{i}", tensorIndex=t) for i, t in enumerate(graph.inputs)
        ],
        outputs=[
            NS(name=f"output_{i}", tensorIndex=t)
            for i, t in enumerate(selected_outputs)
        ],
    )
    document.model.signatureDefs.append(signature)
    return signature
