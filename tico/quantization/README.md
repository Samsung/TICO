# quantization

[Documentation index](../../docs/README.md) · [Guide map](#documentation-map)

## Overview

The _quantization_ module provides the public `prepare` / `convert` API for neural
networks, including LLMs and VLMs. This guide owns the shared API and quantizer
registration contract. Algorithm-specific options and numerical behavior are
documented in the [algorithm guides](./algorithm/README.md); wrapper lifecycle and
frozen-state contracts are documented in [WrapQ](./wrapq/README.md).

- Public Interface
  - `prepare(model, quant_config, args, kwargs, inplace)`
    - Prepares the given model for quantization based on the provided algorithm-specific
      configuration. This involves setting up necessary observers or hooks, and may
      optionally use example inputs (provided as separate positional and keyword
      arguments), which is particularly useful for activation quantization.
  - `convert(model, inplace=True)`
    - Converts the model that has been prepared and calibrated into its quantized form.
      It retrieves the quantizer attached by `prepare()` and uses the existing
      configuration and calibration state. It does not accept `quant_config` or
      select or instantiate a new quantizer. PTQ wrappers retain their resolved
      quantization policies.

- Algorithm-Specific Configuration
  - Users supply algorithm-specific configuration objects (e.g., `PTQConfig` for
    activation quantization or `GPTQConfig` for weight quantization) to `prepare()`.
    These configuration objects encapsulate the parameters for each
    quantization method.

- Internal Dispatch to Quantizer Implementations
  - `prepare()` selects the algorithm-specific quantizer through
    `get_quantizer(quant_config)` and attaches it to the returned prepared model as
    `tico_quantizer`. That quantizer owns its configuration; `convert()` reuses the
    attached instance.

After successful conversion, the returned model no longer carries `tico_quantizer`.
If conversion fails, the original prepared model retains the handle for retry.
Retaining the handle does not roll back arbitrary in-place algorithm mutations.
For algorithms supporting `convert(prepared_model, inplace=False)`, the converted
copy has no handle and the original prepared model keeps its handle even after
success. GPTQ continues to require `inplace=True`.

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
quantized_model = convert(prepared_model)
```

## Directory Structure

```bash
quantization/
├── algorithm/        # Algorithm-specific quantizers
│   ├── adaround/     #   Conv2d weight rounding
│   ├── block_reconstruction/ # Activation-qparam reconstruction
│   ├── cle/          #   Cross-Layer Equalization
│   ├── fpi_gptq/     #   Fixed-point-iteration GPTQ variant
│   ├── gemma4_gptq/  #   GPTQ specialization for Gemma4
│   ├── gptq/         #   GPTQ weight quantization
│   ├── qwen3_vl_gptq/#   GPTQ specialization for Qwen3-VL
│   ├── smoothquant/  #   SmoothQuant
│   ├── spinquant/    #   SpinQuant (rotation-based)
│   └── universal_gptq/ # Replay-based GPTQ with configurable caching
├── analysis/         # Numerical ablation, clipping, and sensitivity
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

| Task | Guide |
|---|---|
| Use the Python `prepare` / `convert` API | [Basic usage](#basic-usage) |
| Choose a quantization algorithm or model-specific implementation | [Algorithm catalog](./algorithm/README.md#available-algorithms) |
| Quantize weights with GPTQ | [GPTQ guide](./algorithm/gptq/README.md) |
| Apply smoothing, rotations, or equalization | [SmoothQuant](./algorithm/smoothquant/README.md), [SpinQuant](./algorithm/spinquant/README.md), and [CLE](./algorithm/cle/README.md) |
| Understand wrappers, observers, calibration, and frozen state | [WrapQ guide](./wrapq/README.md) |
| Compare PyTorch and exported Circle outputs | [Numerical evaluation API](./evaluation/README.md) |
| Run model benchmarks through recipes | [Recipe evaluation guide](./recipes/evaluation/README.md) |
| Analyze PTQ error, clipping, and sensitivity | [Quantization analysis](./analysis/README.md) |
| Run quantization, evaluation, export, generation, or inspection from the CLI | [CLI examples](./examples/README.md) |
| Configure pipelines and override policies | [YAML configuration reference](./examples/configs/README.md) |
| Export checkpoints and staged Circle artifacts | [Export guide](./recipes/export/README.md) |
| Trace outputs and inspect wrapper or export regressions | [Debug guide](./recipes/debug/README.md) |
| Add adapters, stages, or reusable workflows | [Recipes developer guide](./recipes/README.md) |
| Understand fake-quant lowering, qparam folding, and requantization | [Quantized Circle design](../../docs/design.md#102-conversion-time-quantized-graph-legalization) |

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

2. Register the Quantizer

Decorate the quantizer with `register_quantizer(NewAlgoConfig)` from
`tico/quantization/quantizer_registry.py`. `prepare()` resolves the quantizer by exact
config type; configs whose `name` matches `tico.quantization.algorithm.<name>.quantizer`
are also imported lazily on first use. Registration rules:

- One config type maps to exactly one quantizer class. Re-registering the same class is
  a no-op, so repeated activation of an extension is safe.
- Registering a different quantizer for an already mapped config type raises
  `ValueError`. Do not reuse a built-in config type for a new algorithm; define your own
  config class, including for quantizers shipped in separately installed packages.
- `registered_quantizer(config_cls)` returns the current mapping without triggering the
  lazy import.

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
