<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logo/banner/tico-banner-dark.png">
    <img src="./assets/logo/banner/tico-banner-light.png" alt="TICO — Torch IR to Circle" width="640">
  </picture>
</p>

_TICO_ (**T**orch **I**R to **C**ircle [ONE](https://github.com/Samsung/ONE)) is a Python
library that converts PyTorch modules into Circle models — a lightweight and efficient
representation in ONE designed for optimized on-device neural network inference.

## Highlights

- **One-call conversion** — `tico.convert()` turns an `nn.Module` into a ready-to-deploy `.circle` binary.
- **`.pt2` support** — convert saved `torch.export` programs via the Python API or the `pt2-to-circle` command-line tool.
- **Run Circle models in Python** — execute converted models directly for quick parity checks against PyTorch.
- **Circle artifact tools** — inspect, check, extract, and clean up exported `.circle` graphs with the `tico.circle` API or `tico-circle` CLI.
- **Quantization toolkit** — a unified `prepare`/`convert` API with GPTQ, PTQ (WrapQ), SmoothQuant, SpinQuant, and CLE, plus config-driven CLI recipes for LLMs and VLMs.

## Installation

### Prerequisites

- Python **3.10+**
- (Optional) [one-compiler](https://github.com/Samsung/ONE/releases) — only required to run
  inference with converted Circle models. Conversion itself does not need it.

We highly recommend using a virtual environment (e.g., conda, venv).

### From PyPI

```bash
pip install tico
```

### From source

```bash
git clone https://github.com/Samsung/TICO.git
cd TICO

./ccex build     # generates build/ and dist/
./ccex install   # installs the package
```

**`./ccex install` options**

| Option | Description |
|---|---|
| `--dist` | Install from the built wheel (default is editable mode) |
| `--torch_ver <ver>` | Torch version to install: a family (`2.5` ~ `2.10`), an exact version (e.g. `2.7.0+cu118`), or `nightly`. Default: `2.7` |
| `--cuda_ver <maj.min>` | Override the detected CUDA version (e.g. `12.1`) |
| `--cpu_only` | Force a CPU-only Torch installation |

## Quick start

```python
import tico
import torch

class AddModule(torch.nn.Module):
    def forward(self, x, y):
        return x + y

torch_module = AddModule()
example_inputs = (torch.ones(4), torch.ones(4))

circle_model = tico.convert(torch_module.eval(), example_inputs)
circle_model.save('add.circle')
```

> [!NOTE]
> Call `eval()` on the module before conversion. _TICO_ internally uses
> [`torch.export`](https://pytorch.org/docs/stable/export.html#torch-export), so the module
> must be export-able.

Converting a saved `.pt2` file from the command line:

```bash
pt2-to-circle -i add.pt2 -o add.circle
```

See the [Getting Started guide](./docs/getting_started.md) for compile configurations,
`.pt2` conversion, and running Circle models directly in Python.

## Circle artifact tools

`tico.circle` provides Python APIs and the `tico-circle` CLI for inspecting and
transforming exported `.circle` files.

```bash
tico-circle inspect model.circle --tensors --operators
tico-circle verify model.circle
tico-circle extract model.circle --ops 20-64 -o region.circle
```

`verify` performs a static internal-consistency check of the Circle artifact itself, including
indices, graph connections, buffers, signatures, and subgraph references. It does not
run inference or validate numerical accuracy or backend compatibility.

See the [Circle artifact tools guide](./tico/circle/README.md) for the Python API,
exact verification rules, extraction semantics, cleanup passes, multi-subgraph and
signature behavior, and standard-input/standard-output pipelines.

## Quantization

The [`tico.quantization`](./tico/quantization/README.md) module provides a unified,
modular interface for quantizing neural networks — including large language models —
through a simple two-step **prepare** → **convert** workflow:

```python
from tico.quantization import prepare, convert
from tico.quantization.config.gptq import GPTQConfig

prepared_model = prepare(model.eval(), GPTQConfig())

for d in dataset:          # calibration
    prepared_model(d)

quantized_model = convert(prepared_model, GPTQConfig())
```

- [Quantization overview](./tico/quantization/README.md) — API, architecture, and how to add a new algorithm
- [Quantization algorithms](./tico/quantization/algorithm/README.md) — GPTQ, SmoothQuant, SpinQuant, CLE, …
- [Config-driven CLI examples](./tico/quantization/examples/README.md) — quantize, evaluate, and inspect LLM/VLM recipes from the command line

## Documentation

### For users

| Document | Description |
|---|---|
| [Getting Started](./docs/getting_started.md) | Converting modules and `.pt2` files, compile configuration, running Circle models directly in Python |
| [Circle artifact tools](./tico/circle/README.md) | Inspecting, verifying, extracting, and cleaning up exported Circle models |
| [Quantization](./tico/quantization/README.md) | The `prepare`/`convert` quantization API and toolkit |
| [Quantization examples](./tico/quantization/examples/README.md) | Command-line quantization, evaluation, and debugging workflows |

### For developers

| Document | Description |
|---|---|
| [Development guide](./docs/development.md) | Environment setup, testing, and code formatting with `./ccex` |
| [System design](./docs/design.md) | Architecture, pass pipeline, invariants, and behavior design |
| [Circle artifact tools](./tico/circle/README.md#writing-a-new-circle-pass) | Circle pass contracts, index rewriting, verification, and test expectations |
| [Requirements](./docs/requirements.md) | Functional and non-functional requirements |
| [System tests](./docs/system_test.md) | System-level test coverage |

## Contributing

Contributions are welcome! For quantization algorithms, start with the
[quantization contribution guide](./tico/quantization/README.md#contributing-adding-a-new-algorithm)
and the [recipes developer guide](./tico/quantization/recipes/README.md).
Before submitting a PR, set up the development environment and run the tests and
formatter as described in the [development guide](./docs/development.md).

## License

Licensed under the Apache License 2.0 — see [LICENSE](./LICENSE).
