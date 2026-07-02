# Getting Started

This guide explains how to use _TICO_ to generate a Circle model from a PyTorch module
and how to run the result directly in Python.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Converting a torch module](#converting-a-torch-module)
- [Compile configuration](#compile-configuration)
- [Converting a .pt2 file](#converting-a-pt2-file)
- [Running Circle models directly in Python](#running-circle-models-directly-in-python)
- [Next steps](#next-steps)

## Prerequisites

Install _TICO_ first — see [Installation](../README.md#installation).

_TICO_ internally uses
[`torch.export`](https://pytorch.org/docs/stable/export.html#torch-export), so the torch
module must be export-able. If you have trouble exporting your module, see
[the limitations of torch.export](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html#limitations-of-torch-export).

Throughout this guide we use this module:

```python
import tico
import torch

class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y
```

## Converting a torch module

```python
torch_module = AddModule()
example_inputs = (torch.ones(4), torch.ones(4))

circle_model = tico.convert(torch_module.eval(), example_inputs)
circle_model.save('add.circle')
```

> [!NOTE]
> Make sure to call `eval()` on the PyTorch module before passing it to the API.
> This ensures the model runs in inference mode, disabling layers like dropout and
> batch normalization updates.

## Compile configuration

Conversion behavior that affects numerics is controlled by an explicit compile
configuration. Pass a `CompileConfigV1` to `tico.convert`:

```python
from test.modules.op.add import AddWithCausalMaskFolded

torch_module = AddWithCausalMaskFolded()
example_inputs = torch_module.get_example_inputs()

config = tico.CompileConfigV1()
config.legalize_causal_mask_value = True
circle_model = tico.convert(torch_module, example_inputs, config=config)
circle_model.save('add_causal_mask_m120.circle')
```

With `legalize_causal_mask_value` on, the causal mask value is converted from
`-inf` to `-120`, creating a more quantization-friendly Circle model at the cost of a
slight accuracy drop.

See the [configuration schema](./design.md#53-configuration-schema-excerpt) in the design
document for the full list of toggles.

## Converting a .pt2 file

A torch module can be exported and saved as a `.pt2` file (from PyTorch 2.1):

```python
module = AddModule()
example_inputs = (torch.ones(4), torch.ones(4))

exported_program = torch.export.export(module, example_inputs)
torch.export.save(exported_program, 'add.pt2')
```

There are two ways to convert a `.pt2` file: the Python API and the command-line tool.

- Python API

```python
circle_model = tico.convert_from_pt2('add.pt2')
circle_model.save('add.circle')
```

- Command-line tool

```bash
pt2-to-circle -i add.pt2 -o add.circle
```

- Command-line tool with configuration

```bash
pt2-to-circle -i add.pt2 -o add.circle -c config.yaml
```

```yaml
# config.yaml

version: '1.0' # You must specify the config version.
legalize_causal_mask_value: True
```

## Running Circle models directly in Python

After export, you can run the Circle model directly in Python. Output types are
`numpy.ndarray`.

> [!NOTE]
> Running Circle models requires the
> [one-compiler](https://github.com/Samsung/ONE/releases) package (for
> `circle-interpreter`). Alternatively, install the `onert` runtime with `pip install onert`.

```python
torch_module = AddModule()
example_inputs = (torch.ones(4), torch.ones(4))

circle_model = tico.convert(torch_module, example_inputs)
circle_model(*example_inputs)
# numpy.ndarray([2., 2., 2., 2.], dtype=float32)
```

## Next steps

- [Quantization](../tico/quantization/README.md) — quantize models with the `prepare`/`convert` API
- [Quantization examples](../tico/quantization/examples/README.md) — config-driven CLI workflows for LLMs/VLMs
- [System design](./design.md) — how the conversion pipeline works internally
