# TICO

_TICO_ (Torch IR to Circle [ONE](https://github.com/Samsung/ONE)) is a python library for converting 
 Pytorch modules into a circle model that is a lightweight and efficient representation in ONE 
designed for optimized on-device neural network inference.

## Table of Contents

### For Users

- [Installation](#installation)
- [Getting Started](#getting-started)
  - [From torch module](#from-torch-module)
  - [From .pt2](#from-pt2)
  - [Running circle models directly in Python](#running-circle-models-directly-in-python)
  - [Quantization](#quantization)

### For Developers

- [Testing & Code Formatting](#testing--code-formatting)
- [Testing](#testing)
- [Code Formatting](#code-formatting)

## For Users

### Installation

0. Prerequisites

- Python 3.10
- (Optional) [one-compiler 1.30.0](https://github.com/Samsung/ONE/releases/tag/1.30.0)
  - It is only required if you intend to run inference with the converted Circle model. If you are only converting models without running them, this dependency is not needed.

We highly recommend to use a virtual env, e.g., conda.

1. Clone this repo

2. Build python package

```bash
./ccex build
```

This will generate `build` and `dist` directories in the root directory.

3. Install generated package

```bash
./ccex install
```

**Available options**
- `--dist` To install the package from .whl (without this option, _TICO_ is installed in an editable mode)
- `--torch_ver <torch version>` To install a specific torch version (default: 2.6).
  - Available <torch version>: 2.5, 2.6, 2.7, 2.8, nightly

4. Now you can convert a torch module to a `.circle`.

### Getting started

This tutorial explains how you can use _TICO_ to generate a circle model from a torch module. 

Let's assume we have a torch module.

```python
import tico
import torch

class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y
```

**NOTE**
_TICO_ internally uses [torch.export](https://pytorch.org/docs/stable/export.html#torch-export).
Therefore, the torch module must be 'export'able. Please see 
[this document](https://pytorch.org/docs/stable/export.html#limitations-of-torch-export) 
if you have any trouble to export.

#### From torch module

You can convert a torch module to a circle model with these steps.

```python
torch_module = AddModule()
example_inputs = (torch.ones(4), torch.ones(4))

circle_model = tico.convert(torch_module.eval(), example_inputs)
circle_model.save('add.circle')
```

**NOTE**
Please make sure to call `eval()` on the PyTorch module before passing it to our API.
This ensures the model runs in inference mode, disabling layers like dropout and
batch normalization updates.

**Compile with configuration**

```python
from test.modules.op.add import AddWithCausalMaskFolded

torch_module = AddWithCausalMaskFolded()
example_inputs = torch_module.get_example_inputs()

config = tico.CompileConfigV1()
config.legalize_causal_mask_value = True
circle_model = tico.convert(torch_module, example_inputs, config = config)
circle_model.save('add_causal_mask_m120.circle')
```

With `legalize_causal_mask_value` option on, causal mask value is converted from 
 -inf to -120, creating a more quantization-friendly circle model with the cost of 
slight accuracy drop.

#### From .pt2

The torch module can be exported and saved as `.pt2` file (from PyTorch 2.1).

```python
module = AddModule()
example_inputs = (torch.ones(4), torch.ones(4))

exported_program = torch.export.export(module, example_inputs)
torch.export.save(exported_program, 'add.pt2')
```

There are two ways to convert `.pt2` file: python api, command line tool.

- Python API

```python
circle_model = tico.convert_from_pt2('add.pt2')
circle_model.save('add.circle')
```

- Command Line Tool

```bash
pt2-to-circle -i add.pt2 -o add.circle
```

- Command Line Tool with configuration

```bash
pt2-to-circle -i add.pt2 -o add.circle -c config.yaml
```

```yaml
# config.yaml

version: '1.0' # You must specify the config version. 
legalize_causal_mask_value: True
```

#### Running circle models directly in Python

After circle export, you can run the model directly in Python.

Note that you should install one-compiler package first.

The output types are numpy.ndarray.

```python
torch_module = AddModule()
example_inputs = (torch.ones(4), torch.ones(4))

circle_model = tico.convert(torch_module, example_inputs)
circle_model(*example_inputs)
# numpy.ndarray([2., 2., 2., 2.], dtype=float32)
```

### Quantization

The `tico.quantization` module provides a unified and modular interface for quantizing
 large language models (LLMs) and other neural networks.
 
It introduces a simple two-step workflow — **prepare** and **convert** — that
 abstracts the details of different quantization algorithms.

#### Basic Usage

```python
from tico.quantization import prepare, convert
from tico.quantization.config.gptq import GPTQConfig
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        return self.linear(x)

model = LinearModel().eval()

# 1. Prepare for quantization
quant_config = GPTQConfig()
prepared_model = prepare(model, quant_config)

# 2. Calibration
for d in dataset:
    prepared_model(d)

# 3. Apply GPTQ
quantized_model = convert(prepared_model, quant_config)
```

For detailed documentation, design notes, and contributing guidelines, 
see [tico/quantization/README.md](./tico/quantization/README.md).


## For Developers

### Testing & Code Formatting

Run below commands to configure testing or formatting environment.

Refer to the dedicated section to have more fine-grained control.

```bash
$ ./ccex configure                          # to set up testing & formatting environment
$ ./ccex configure format                   # to set up only formatting environment
$ ./ccex configure test                     # to set up only testing environment
```

**Available options**
- `--torch_ver <torch version>` To install a specific torch family package(ex. torchvision) version (default: 2.6)
  - Available <torch version>: '2.5', '2.6', 'nightly'

```bash
$ ./ccex configure                          # to set up testing & formatting environment with stable2.6.x version
$ ./ccex configure test                     # to set up only testing environment with stable 2.6.x version
$ ./ccex configure test --torch_ver 2.5     # to set up only testing environment with stable 2.5.x version
$ ./ccex configure test --torch_ver nightly     # to set up only testing environment with nightly version
```

### Testing

#### Test congifure

Run below commands to install requirements for testing.

**NOTE** `TICO` will be installed in an editable mode.

```bash
./ccex configure test

# without editable install
./ccex configure test --dist
```

#### Test All

Run below commands to run the all unit tests.

**NOTE** Unit tests don't include model test.

```bash
./ccex test
# OR
./ccex test run-all-tests
```

#### Test Subset

To run subset of `test.modules.*`,
Run `./ccex test -k <keyword>`


For example, to run tests in specific sub-directory (op, net, ..)
```bash
# To run tests in specific sub-directory (op/, net/ ..)
./ccex test -k op
./ccex test -k net

# To run tests in one file (single/op/add, single/op/sub, ...)
./ccex test -k add
./ccex test -k sub

# To run SimpleAdd test in test/modules/single/op/add.py
./ccex test -k SimpleAdd
```

To see the full debug log, add `-v` or `TICO_LOG=4`.

```bash
TICO_LOG=4 ./ccex test -k add
# OR
./ccex test -v -k add
```

#### Test Model

If you want to test them locally, you can do so by navigating to each model directory, 
 installing the dependencies listed in its `requirements.txt`, and running the tests one by one.
```bash
$ pip install -r test/modules/model/<model_name>/requirements.txt
# Run test for a single model
$ ./ccex test -m <model_name>
# Run models whose names contain "Llama" (e.g., Llama, LlamaDecoderLayer, LlamaWithGQA, etc.)
# Note that you should use quotes for the wildcard(*) pattern
$ ./ccex test -m "Llama*"
```

For example, to run a single model
```
./ccex test -m InceptionV3
```

#### Runtime Options

By default, `./ccex test` runs all modules with the `circle-interpreter` engine.
 You can override this and run tests using the `onert` runtime instead.


##### 0. Install ONERT

```bash
pip install onert
```

##### 1. Command-Line Flag

Use the `--runtime` (or `-r`) flag to select a runtime:

```bash
# Run with the default circle-interpreter
./ccex test

# Run all tests with onert
./ccex test --runtime onert
# or
./ccex test -r onert
```

##### 2. Environment Variable

You can also set the `CCEX_RUNTIME` environment variable:

```bash
# Temporarily override for one command
CCEX_RUNTIME=onert ./ccex test

# Persist in your shell session
export CCEX_RUNTIME=onert
./ccex test
```

##### Supported Runtimes

- circle-interpreter (default): uses the Circle interpreter for inference.
- onert: uses the ONERT package for inference, useful when the Circle interpreter
 cannot run a given module.

### Code Formatting

#### Format configure

Run below commands to install requirements for formatting.

```bash
./ccex configure format
```

#### Format run

```bash
./ccex format
```