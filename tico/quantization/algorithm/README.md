# algorithm

The _algorithm_ module implements quantization algorithms and preprocessing methods
for deep learning. Some methods transform model weights before a later quantization
stage, while others quantize weights or reconstruct activation quantization parameters.
They are not all graph rewrites or preprocessing steps.

## Available Algorithms

| Algorithm | Directory | Description |
|---|---|---|
| GPTQ | [`gptq/`](./gptq/README.md) | Accurate post-training weight quantization |
| Block reconstruction | [`block_reconstruction/`](./block_reconstruction/README.md) | Fixed-weight activation-qparam reconstruction |
| AdaRound | [`adaround/`](./adaround/README.md) | Validation-aware hard Conv2d weight rounding |
| SmoothQuant | [`smoothquant/`](./smoothquant/README.md) | Migrates activation outliers into weights via per-channel smoothing |
| SpinQuant | [`spinquant/`](./spinquant/README.md) | Rotation-based outlier reduction |
| CLE | [`cle/`](./cle/README.md) | Cross-Layer Equalization |
| FPI-GPTQ | `fpi_gptq/` | Fixed-point-iteration GPTQ variant |
| Qwen3-VL GPTQ | `qwen3_vl_gptq/` | GPTQ specialization for Qwen3-VL models |

## Design Philosophy

Algorithm implementations have defined responsibilities within TICO's quantization
stack. They integrate with configuration classes, the quantizer registry, the public
`prepare`/`convert` API, and shared WrapQ infrastructure while keeping recipe
orchestration outside the algorithm implementation.

### 1. External Dependencies

Some algorithms or model-family integrations require optional libraries such as
`transformers`. Keep those dependencies local to the paths that need them and document
their installation requirements.

### 2. Modular and Maintainable Code

Each algorithm owns its numerical implementation and quantizer lifecycle. Shared
wrapper infrastructure belongs in `wrapq/`, configuration belongs in `config/`, and
pipeline orchestration belongs in `recipes/stages/`. These boundaries support focused
testing and maintenance without duplicating shared behavior.

## Usage Guidelines

### Integration Boundaries

Use the public `prepare`/`convert` API and algorithm-specific configurations for model
quantization. Framework code may import algorithm implementations and utilities through
their owning modules: the quantizer registry resolves implementations, and recipe
stages orchestrate their use. Algorithms should not depend on example entrypoints or
recipe orchestration, which would introduce circular dependencies.

### Dependency Isolation

Install the dependencies required by the selected algorithm or model family. Avoid
making optional algorithm dependencies mandatory for unrelated project components.

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
quantized_model = convert(prepared_model)
```

Some algorithms need extra dependencies; install them from the algorithm's requirements
file first (e.g., `pip install -r smoothquant/smooth_quant.txt`). See each algorithm's
README for its configuration class and usage details. Any output or data exchange
between _algorithm_ and other modules should be done via well-defined interfaces.

## Contributing to the Module

- Implement new algorithms in this package and expose their configurations through
  the quantizer registry and public API.
- Reuse shared infrastructure through defined interfaces and avoid circular imports.
- Document any external dependencies clearly in the module's README or requirements
  file.

See also the top-level [quantization contribution guide](../README.md#contributing-adding-a-new-algorithm)
and the [recipes developer guide](../recipes/README.md) for exposing an algorithm as a
CLI pipeline stage.
