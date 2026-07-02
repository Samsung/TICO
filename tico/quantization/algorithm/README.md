# algorithm

The _algorithm_ module provides a collection of state-of-the-art quantization algorithms
for deep learning. These methods rewrite target graphs prior to the actual quantization
step, ensuring minimal performance loss and improved quantization accuracy.

## Available Algorithms

| Algorithm | Directory | Description |
|---|---|---|
| GPTQ | [`gptq/`](./gptq/README.md) | Accurate post-training weight quantization |
| SmoothQuant | [`smoothquant/`](./smoothquant/README.md) | Migrates activation outliers into weights via per-channel smoothing |
| SpinQuant | [`spinquant/`](./spinquant/README.md) | Rotation-based outlier reduction |
| CLE | [`cle/`](./cle/README.md) | Cross-Layer Equalization |
| FPI-GPTQ | `fpi_gptq/` | Fixed-point-iteration GPTQ variant |
| Qwen3-VL GPTQ | `qwen3_vl_gptq/` | GPTQ specialization for Qwen3-VL models |

## Design Philosophy

This module is designed to be **self-contained** and **independent** of other components
in the codebase. The primary reasons for this design are:

### 1. External Dependencies

The _algorithm_ module relies on external libraries such as `transformers`, which may not
be compatible with the minimal dependencies required by other parts of the project.

### 2. Modular and Maintainable Code

Each algorithm in this module is implemented as a standalone component to:
- Avoid tight coupling with internal project code.
- Ensure ease of testing, maintenance, and future updates.

## Usage Guidelines

### Do not Cross-Integrate

The _algorithm_ module should not be directly referenced by other modules or components
within the _TICO_ codebase. Instead, this module is designed to be used independently
for quantization and other algorithm-specific tasks.

### Dependency Isolation

Ensure that any code or scripts utilizing this module explicitly install the required
dependencies, such as `transformers`. Dependencies for this module should not be
propagated to other project components.

## Example Use Case

Algorithms are used through the public `prepare`/`convert` interface with their
algorithm-specific config:

```python
from tico.quantization import prepare, convert
from tico.quantization.config.gptq import GPTQConfig

model = model.eval()

# 1. Prepare for quantization
quant_config = GPTQConfig()
prepared_model = prepare(model, quant_config)

# 2. Calibration
for d in dataset:
    prepared_model(d)

# 3. Convert
quantized_model = convert(prepared_model, quant_config)
```

Some algorithms need extra dependencies; install them from the algorithm's requirements
file first (e.g., `pip install -r smoothquant/smooth_quant.txt`). See each algorithm's
README for its configuration class and usage details. Any output or data exchange
between _algorithm_ and other modules should be done via well-defined interfaces.

## Contributing to the Module

- New algorithms should be implemented as standalone modules with minimal
  interdependencies.
- Avoid introducing code that requires circular imports or tight integration with
  internal project components.
- Document any external dependencies clearly in the module's README or requirements
  file.

See also the top-level [quantization contribution guide](../README.md#contributing-adding-a-new-algorithm)
and the [recipes developer guide](../recipes/README.md) for exposing an algorithm as a
CLI pipeline stage.
