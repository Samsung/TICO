# quantization

## Overview

The _quantization_ module provides a unified, modular interface for quantizing neural
networks, including large language models (LLMs) and vision-language models (VLMs).
Its primary goal is to simplify the process of preparing, calibrating, and converting
models into their quantized versions while abstracting the underlying complexities of
different quantization algorithms.
At the top level, the module exposes two primary public interface functions:
_prepare_ and _convert_.

- Public Interface
  - `prepare(model, quant_config, args, kwargs, inplace)`
    - Prepares the given model for quantization based on the provided algorithm-specific
      configuration. This involves setting up necessary observers or hooks, and may
      optionally use example inputs (provided as separate positional and keyword
      arguments), which is particularly useful for activation quantization.
  - `convert(model, quant_config)`
    - Converts the model that has been prepared and calibrated into its quantized form.
      This function leverages the statistics collected during calibration to perform the
      quantization transformation.

- Algorithm-Specific Configuration
  - Users supply algorithm-specific configuration objects (e.g., `PTQConfig` for
    activation quantization or `GPTQConfig` for weight quantization) to the public
    interface. These configuration objects encapsulate the parameters for each
    quantization method.

- Internal Dispatch to Quantizer Implementations
  - `prepare` and `convert` internally dispatch an appropriate quantizer according to
    the type of `quant_config`.

This design ensures that users interact only with the high-level functions and
configuration objects, while the internal details of each quantization algorithm are
encapsulated within dedicated quantizer classes.

## Basic Usage

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

## Purpose

The module is designed to:
- Simplify Model Quantization: Provide a simple, high-level API that abstracts the
  calibration and conversion complexities.
- Support Multiple Algorithms: Enable users to choose quantization algorithms by
  specifying configuration objects (e.g., `PTQConfig`, `GPTQConfig`).
- Promote Modularity and Extensibility: Offer a clear separation between the public API,
  configuration classes, and internal quantizer implementations, ensuring the module is
  easy to extend and maintain.

## Directory Structure

```bash
quantization/
├── algorithm/        # Algorithm-specific quantizers
│   ├── cle/          #   Cross-Layer Equalization
│   ├── fpi_gptq/     #   Fixed-point-iteration GPTQ variant
│   ├── gptq/         #   GPTQ weight quantization
│   ├── qwen3_vl_gptq/#   GPTQ specialization for Qwen3-VL
│   ├── smoothquant/  #   SmoothQuant
│   └── spinquant/    #   SpinQuant (rotation-based)
├── config/           # Configuration definitions (PTQConfig, GPTQConfig, ...)
├── evaluation/       # Evaluation utilities
├── examples/         # Config-driven CLI examples (quantize / evaluate / inspect)
├── passes/           # Graph-level passes and transformations
├── recipes/          # Reusable pipeline layer behind the example CLIs
└── wrapq/            # Wrapper-based PTQ infrastructure (observers, wrappers)
```

The module is composed of two main implementation layers:

- Algorithm layer ([`quantization/algorithm/`](./algorithm/README.md))
  - Implements algorithm-specific quantization logic (e.g., GPTQ, SmoothQuant,
    SpinQuant, CLE).
- Infrastructure layer ([`quantization/wrapq/`](./wrapq/README.md))
  - Provides a generic, wrapper-based quantization backend shared across algorithms.

On top of these, two layers make the toolkit usable end-to-end from the command line:

- Recipes ([`quantization/recipes/`](./recipes/README.md))
  - The reusable pipeline layer: model-family adapters, algorithm stages, calibration
    data builders, evaluation, export, and debug helpers.
- Examples ([`quantization/examples/`](./examples/README.md))
  - Thin CLI entry points (`quantize.py`, `evaluate.py`, `inspector.py`) driven by YAML
    presets in [`examples/configs/`](./examples/configs/README.md).

## Documentation Map

| Document | Description |
|---|---|
| [algorithm/README.md](./algorithm/README.md) | Algorithm module design and usage guidelines |
| [algorithm/gptq/README.md](./algorithm/gptq/README.md) | GPTQ |
| [algorithm/smoothquant/README.md](./algorithm/smoothquant/README.md) | SmoothQuant |
| [algorithm/spinquant/README.md](./algorithm/spinquant/README.md) | SpinQuant |
| [algorithm/cle/README.md](./algorithm/cle/README.md) | Cross-Layer Equalization |
| [wrapq/README.md](./wrapq/README.md) | Wrapper-based PTQ infrastructure |
| [recipes/README.md](./recipes/README.md) | Recipes developer guide (adapters, stages, import rules) |
| [examples/README.md](./examples/README.md) | CLI examples: quantize, evaluate, inspect |
| [examples/configs/README.md](./examples/configs/README.md) | YAML config preset reference |

## Contributing: Adding a New Algorithm

We welcome contributions that enhance the module by adding support for new quantization
algorithms. If you are interested in contributing a new algorithm, please follow these
steps:

1. Implement New Configuration and Quantizer Classes

- Configuration
  - Create a new configuration class (e.g., `NewAlgoConfig`) that inherits from
    `BaseConfig`. Include all necessary parameters and default values specific to your
    algorithm.

- Quantizer
  - Implement a new quantizer class (e.g., `NewAlgoQuantizer`) that inherits from
    `BaseQuantizer`. Implement the _prepare_ and _convert_ methods.

2. Update Public Interface Dispatch (if necessary)

Modify the dispatch logic in the public interface functions (`prepare` and `convert`) to
recognize your new configuration type and instantiate your new quantizer accordingly.

3. Write Tests

Ensure that you add unit tests and integration tests for your new algorithm. Tests
should cover both the _prepare_ and _convert_ phases and validate that the quantized
model meets performance expectations.

4. Document Your Changes

Update the README and any related documentation to include details about the new
algorithm, its configuration parameters, and usage examples.

To expose the new algorithm through the CLI examples as a pipeline stage, follow the
[recipes developer guide](./recipes/README.md) — most new workflows should be added as
YAML presets, not new scripts.
