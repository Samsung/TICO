# Getting Started

TICO converts exportable PyTorch modules and saved
`torch.export.ExportedProgram` files into Circle models. Conversion itself is a Python
workflow; a Circle runtime is needed only when the generated model is executed.

## Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Convert a PyTorch module](#convert-a-pytorch-module)
- [Public conversion APIs](#public-conversion-apis)
- [Keyword arguments](#keyword-arguments)
- [Dynamic shapes](#dynamic-shapes)
- [Compile configuration](#compile-configuration)
- [Convert a saved `.pt2` file](#convert-a-saved-pt2-file)
- [Run a Circle model in Python](#run-a-circle-model-in-python)
- [Inspect and verify Circle artifacts](#inspect-and-verify-circle-artifacts)
- [Troubleshooting](#troubleshooting)
- [Next steps](#next-steps)

## Prerequisites

- Python 3.10 or newer
- A supported PyTorch installation
- An inference-mode, `torch.export`-compatible module

TICO's source tooling supports stable Torch families 2.5 through 2.10 and a pinned
nightly build. The default source-install family is 2.7. TICO warns below 2.5 and
recommends Torch 2.6 or newer when possible because older releases may contain known
security vulnerabilities. See the
[Development Guide](./development.md) for exact installation and CI details.

> [!IMPORTANT]
> Call `eval()` before conversion. TICO rejects training operators such as dropout in
> the exported graph, and training-mode state updates can make PyTorch/Circle parity
> checks unreliable.

## Installation

### Install from PyPI

```bash
pip install tico
```

### Install from source

```bash
git clone https://github.com/Samsung/TICO.git
cd TICO

python3 -m venv .venv
source .venv/bin/activate

# Installs a supported Torch build and TICO in editable mode.
./ccex install
```

Useful source-install options include:

```bash
./ccex install --cpu_only
./ccex install --torch_ver 2.7
./ccex install --torch_ver 2.10.0
./ccex install --torch_ver nightly
./ccex install --cuda_ver 12.8
```

`one-compiler` is not required to create Circle files. It is required for the bundled
Circle interpreter used by `CircleModel.__call__()` and by the default end-to-end test
runtime. Install a compatible ONE release when local Circle execution is needed.

## Convert a PyTorch module

```python
import torch
import tico


class AddModule(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


model = AddModule().eval()
example_inputs = (torch.ones(4), torch.ones(4))

circle_model = tico.convert(model, example_inputs)
circle_model.save("add.circle")
```

`example_inputs` serve two purposes:

1. `torch.export.export()` uses them to capture the program.
2. Their shapes and dtypes define the default Circle input contract.

The result is a `CircleModel` object containing the serialized bytes in
`circle_model.circle_binary`.

## Public conversion APIs

TICO exports the following APIs from the top-level package:

```python
import tico

circle_model = tico.convert(...)
circle_model = tico.convert_from_exported_program(...)
circle_model = tico.convert_from_pt2(...)
```

Their current signatures are conceptually:

```python
tico.convert(
    mod,
    args,
    kwargs=None,
    dynamic_shapes=None,
    strict=True,
    config=tico.get_default_config(),
)

tico.convert_from_exported_program(exported_program, config=...)
tico.convert_from_pt2(pt2_path, config=...)
```

Use `convert()` when starting from an `nn.Module`,
`convert_from_exported_program()` when export has already been performed in memory,
and `convert_from_pt2()` for a saved exported program.

## Keyword arguments

Pass keyword example inputs through the `kwargs` argument:

```python
import torch
import tico


class Affine(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        *,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        return x * scale + bias


model = Affine().eval()
args = (torch.ones(2, 3),)
kwargs = {
    "scale": torch.tensor(2.0),
    "bias": torch.tensor(0.5),
}

circle_model = tico.convert(model, args, kwargs=kwargs)
```

At execution time, TICO binds positional inputs first and then keyword inputs by the
serialized Circle input names. Use the same argument structure, dtype, and static
dimensions that were used for export.

## Dynamic shapes

TICO forwards `dynamic_shapes` and `strict` to `torch.export.export()`.

```python
import torch
import tico


class RowSum(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1)


model = RowSum().eval()
x = torch.randn(2, 4)
batch = torch.export.Dim("batch", min=1, max=8)

circle_model = tico.convert(
    model,
    (x,),
    dynamic_shapes=({0: batch},),
)
circle_model.save("row_sum_dynamic.circle")
```

A symbolic dimension is serialized with a Circle `shapeSignature` value of `-1` and a
concrete placeholder dimension of `1` in `shape`. Static dimensions are preserved.

Dynamic-shape export and dynamic-shape execution are separate capabilities. TICO's
model test harness uses the `onert` runtime for dynamic-shape execution and updates the
runtime input tensor information from the actual inputs. The bundled Python Circle
interpreter is primarily used for static-shape validation.

## Compile configuration

`CompileConfigV1` controls selected conversion rewrites:

```python
import torch
import tico


class Matmul(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", torch.randn(8, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


config = tico.CompileConfigV1()
config.convert_rhs_const_mm_to_fc = True

model = Matmul().eval()
circle_model = tico.convert(model, (torch.randn(2, 8),), config=config)
```

Current version 1.0 fields are:

| Field | Default | Current behavior |
|---|---:|---|
| `legalize_causal_mask_value` | `False` | For recognized lifted attention-mask constants, replaces sufficiently negative mask values with `-120`. Enable only when the finite-mask assumption is valid for the target workflow. |
| `remove_constant_input` | `False` | Omits exported `ConstantArgument` values from the Circle user-input interface. `None` inputs are omitted regardless of this option. |
| `convert_lhs_const_mm_to_fc` | `False` | Enables eligible matrix-multiplication rewrites with a constant left-hand side. |
| `convert_rhs_const_mm_to_fc` | `True` | Enables eligible matrix-multiplication rewrites with a constant right-hand side. |
| `convert_single_batch_lhs_const_bmm_to_fc` | `False` | Enables the supported single-batch `bmm` rewrite with a constant left-hand side. |
| `convert_expand_to_slice_cat` | `False` | Enables lowering of supported `expand` patterns to slice/concatenate operations. |
| `eliminate_rank_round_trip` | `False` | Declared in `CompileConfigV1`, but the current conversion pipeline does not read this field; `EliminateRankRoundTripRegion` is currently enabled unconditionally. Do not rely on this field to disable that pass. |

Create a default configuration with either of the following:

```python
config = tico.CompileConfigV1()
config = tico.get_default_config()  # version 1.0
```

## Convert a saved `.pt2` file

Export and save a program with PyTorch:

```python
import torch

model = AddModule().eval()
example_inputs = (torch.ones(4), torch.ones(4))

exported_program = torch.export.export(model, example_inputs)
torch.export.save(exported_program, "add.pt2")
```

### Python API

```python
import tico

circle_model = tico.convert_from_pt2("add.pt2")
circle_model.save("add.circle")
```

### Command-line interface

```bash
pt2-to-circle -i add.pt2 -o add.circle
```

Supported options:

```text
-i, --input    Input .pt2 path
-o, --output   Output .circle path
-c, --config   Optional YAML compile configuration
-v, --verbose  Enable verbose conversion logging
```

A YAML configuration must include `version`. Version `1.0` is currently supported:

```yaml
version: "1.0"
legalize_causal_mask_value: false
remove_constant_input: false
convert_lhs_const_mm_to_fc: false
convert_rhs_const_mm_to_fc: true
convert_single_batch_lhs_const_bmm_to_fc: false
convert_expand_to_slice_cat: false
eliminate_rank_round_trip: false
```

```bash
pt2-to-circle \
  -i add.pt2 \
  -o add.circle \
  -c config.yaml
```

Unknown YAML keys are not applied by `CompileConfigV1.from_dict()`. Treat misspelled
configuration names as errors in review even though the current loader does not reject
them.

## Run a Circle model in Python

`CircleModel` can be called directly:

```python
import numpy as np
import torch
import tico

model = AddModule().eval()
inputs = (torch.ones(4), torch.ones(4))
circle_model = tico.convert(model, inputs)

output = circle_model(*inputs)
assert isinstance(output, np.ndarray)
print(output)
```

Load an existing Circle file:

```python
from tico.utils.model import CircleModel

circle_model = CircleModel.load("add.circle")
output = circle_model(torch.ones(4), torch.ones(4))
```

Runtime behavior:

- A single Circle output is returned as one `numpy.ndarray`.
- Multiple outputs are returned as a list of NumPy arrays.
- Input count, names, dtypes, ranks, and static dimensions are checked before execution.
- The built-in execution path currently requires a one-subgraph Circle model.

## Inspect and verify Circle artifacts

Use `tico-circle` after serialization:

```bash
tico-circle inspect add.circle --tensors --operators
tico-circle verify add.circle
tico-circle extract model.circle --ops 20-64 -o region.circle
```

`verify` checks the internal consistency of the Circle artifact: indices, dataflow,
buffers, graph interfaces, signatures, and control-flow references. It does not execute
the model, compare numerical outputs, or prove compatibility with a particular NPU
compiler.

See [Circle artifact tools](../tico/circle/README.md) for the Python API, extraction
semantics, cleanup passes, and CLI details.

## Troubleshooting

### Training-mode or dropout error

Convert an evaluation-mode module:

```python
model.eval()
circle_model = tico.convert(model, example_inputs)
```

### Unsupported operator

TICO validates every remaining `call_function` target against the registered Circle
serializer visitors. An unsupported operator is reported with the operator name and,
when available, its source stack trace. The usual fixes are to add a legalization pass,
add a serializer visitor, or change the model wrapper to emit supported ATen patterns.

### Verbose graph changes

Set the log level before starting Python:

```bash
TICO_LOG=4 python my_conversion.py
TICO_LOG=4 pt2-to-circle -i model.pt2 -o model.circle
```

Log levels are:

| Value | Level |
|---:|---|
| `1` | fatal |
| `2` | warning |
| `3` | info |
| `4` | debug, including graph/constant diffs where instrumented |

### Intermediate FX graph images

```bash
TICO_GRAPH_DUMP=1 python my_conversion.py
```

The conversion pipeline writes images such as `1_after_decompose.png`,
`2_after_legalize.png`, and `3_after_quantfold.png` under a timestamped directory in
`.tico_tmp/`. Graph rendering requires `pydot` and a working Graphviz installation.

## Next steps

- [System Design](./design.md): conversion stages, invariants, serialization, and
  extension points
- [Development Guide](./development.md): source setup, tests, formatting, and CI
- [Quantization](../tico/quantization/README.md): quantize a model before Circle export
- [Quantization examples](../tico/quantization/examples/README.md): config-driven LLM
  and VLM workflows
- [Circle artifact tools](../tico/circle/README.md): post-serialization inspection and
  transformations
