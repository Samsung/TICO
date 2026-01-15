# quantization

## Overview

The _quantization_ module provides a unified, modular interface for quantizing large language models (LLMs). 
Its primary goal is to simplify the process of preparing, calibrating, and converting models into their 
quantized versions while abstracting the underlying complexities of different quantization algorithms. 
At the top level, the module exposes two primary public interface functions: _prepare_ and _convert_.

- Public Interface
  - prepare(model, quant_config, args, kwargs, inplace)
    - Prepares the given model for quantization based on the provided algorithm-specific configuration. 
      This involves setting up necessary observers or hooks, and may optionally use example inputs 
      (provided as separate positional and keyword arguments), which is particularly useful for 
      activation quantization.
  - convert(model, quant_config)
    - Converts the model that has been prepared and calibrated into its quantized form. This function 
      leverages the statistics collected during calibration to perform the quantization transformation.

- Algorithm-Specific Configuration
  - Users supply algorithm-specific configuration objects (e.g., PTQConfig for activation quantization 
    or GPTQConfig for weight quantization) to the public interface. These configuration objects encapsulate
    the parameters for each quantization method.

- Internal Dispatch to Quantizer Implementations
  - `prepare` and `convert` internally dispatch an appropriate quantizer according to the type of `quant_config`.

This design ensures that users interact only with the high‑level functions and configuration objects, 
while the internal details of each quantization algorithm are encapsulated within dedicated quantizer classes.

## Purpose

The module is designed to:
- Simplify Model Quantization: Provide a simple, high-level API that abstracts the calibration and 
  conversion complexities.
- Support Multiple Algorithms: Enable users to choose quantization algorithms by specifying configuration 
  objects (e.g., PTQConfig, GPTQConfig).
- Promote Modularity and Extensibility: Offer a clear separation between the public API, configuration 
  classes, and internal quantizer implementations, ensuring the module is easy to extend and maintain.

## Directory Structure

The module is composed of two main layers:

- Algorithm layer (quantization/algorithm/)
  - Implements algorithm-specific quantization logics (e.g., GPTQ).
- Infrastructure layer (quantization/wrapq/)
  - Provides a generic, wrapper-based quantization backend shared across algorithms.

Ready-to-run examples can be found in `quantization/wrapq/examples/`.

```bash
quantization/
├── algorithm/        # Algorithm-specific quantizers
│   ├── gptq/
│   ├── smoothquant/
│   └── ...
├── config/           # Configuration definitions
├── evaluation/       # Evaluation utilities
├── passes/           # Graph-level passes and transformations
└── wrapq/            # Wrapper-based quantization infrastructure
    ├── examples
    ├── observers
    ├── utils
    └── wrappers/
        ├── llama/
        └── nn/
```

## Contributing: Adding a New Algorithm

We welcome contributions that enhance the module by adding support for new quantization algorithms. 
If you are interested in contributing a new algorithm, please follow these steps:

1. Implement New Configuration and Quantizer Classes

- Configuration
  - Create a new configuration class (e.g., NewAlgoConfig) that inherits from BaseConfig. Include 
    all necessary parameters and default values specific to your algorithm.

- Quantizer
  - Implement a new quantizer class (e.g., NewAlgoQuantizer) that inherits from BaseQuantizer. 
    Implement the _prepare_ and _convert_ methods.

2. Update Public Interface Dispatch (if necessary)

Modify the dispatch logic in the public interface functions (prepare and convert) to recognize 
your new configuration type and instantiate your new quantizer accordingly.

3. Write Tests

Ensure that you add unit tests and integration tests for your new algorithm. Tests should cover 
both the _prepare_ and _convert_ phases and validate that the quantized model meets performance 
expectations.

4. Document Your Changes

Update the README and any related documentation to include details about the new algorithm, 
its configuration parameters, and usage examples.
