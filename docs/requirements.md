# Software Requirements Specification (SRS)

## Table of Contents

- [1. Introduction](#1-introduction)
  - [1.1 Purpose](#11-purpose)
  - [1.2 Scope](#12-scope)
  - [1.3 Stakeholders](#13-stakeholders)
  - [1.4 Definitions and Acronyms](#14-definitions-and-acronyms)
- [2. Functional Requirements](#2-functional-requirements)
- [3. Non-Functional Requirements](#3-non-functional-requirements)
- [4. Constraints](#4-constraints)
- [5. Appendices](#5-appendices)

## 1. Introduction

### 1.1 Purpose

The purpose of this document is to define the requirements for developing a Python library that enables the direct conversion of PyTorch models into NPU-compatible format (Circle) without external dependencies. The library will also support modern quantization and optimization techniques, improving model performance and developer productivity.

### 1.2 Scope

This project focuses on:

- Providing a direct conversion mechanism from PyTorch to Circle format.
- Enabling model optimization and quantization for NPU deployment.
- Offering user-friendly APIs for developers and researchers to minimize complexity in the model deployment pipeline.
- Ensuring compatibility with evolving NPU compilers and PyTorch versions.

### 1.3 Stakeholders

- **Developers**: Utilize the library to convert and optimize models for NPU execution.
- **Researchers**: Apply advanced quantization and optimization techniques for experimental models.
- **Product Teams**: Benefit from simplified workflows and faster time-to-market.

### 1.4 Definitions and Acronyms

- **PyTorch**: Open-source machine learning framework.
- **NPU**: Neural Processing Unit.
- **Circle**: Open-source intermediate representation (IR) format compatible with NPU compilers.
- **Quantization**: Technique to reduce numerical precision (e.g., 16-bit, 8-bit) for performance improvement.
- **Optimization**: Process of improving model efficiency, such as kernel fusion or operator reordering.


## 2. Functional Requirements

| ID    | Requirement Name                 | Description |
|-------|----------------------------------|-------------|
| RF-1  | PyTorch Model Conversion         | Convert PyTorch models directly into Circle format, ensuring structural integrity and operator compliance. |
| RF-2  | Optimization Algorithm Support   | Support advanced optimization techniques (e.g., kernel fusion, operator reordering), with options for user-defined configurations. |
| RF-3  | Quantization Integration Support | Consume pre-quantized parameters, fold/remove redundant (de)quant ops, quantize missing biases where required, and insert bridging quantize ops on dtype mismatches to keep the graph legal for Circle/NPU. |
| RF-4  | Verification & Debugging Tools   | Offer debugging mode, display detailed error messages, and validate correctness of outputs compared to original PyTorch models. |
| RF-5  | NPU Compiler Compatibility       | Automatically test converted Circle files against NPU compilers and report compatibility results. |


## 3. Non-Functional Requirements

| ID     | Requirement Name               | Description |
|--------|--------------------------------|-------------|
| RNF-1  | Performance: Conversion Speed  | Conversion of a baseline model (**Llama-3.2-1B**) must complete within **10 seconds** on a standard Linux environment (Ubuntu 22.04, Python 3.10, PyTorch 2.6, CPU-only). For larger models, conversion time should scale approximately linearly with parameter count. |
| RNF-2  | Performance: File Size         | Converted Circle files should be equal to or smaller than the serialized PyTorch model size saved via `torch.save(model.state_dict())`. |
| RNF-3  | Reliability                    | Converted model outputs should remain close to the PyTorch reference outputs. The similarity is measured using **PEIR (Peak Error Interval Ratio)** defined as `max(\|a - b\|) / (max(a) - min(a))`. The PEIR must not exceed **3% (0.03)** for any output tensor. |
| RNF-4  | Maintainability: Operator Support | The library should support operator coverage sufficient for modern Transformer and LLM architectures. |
| RNF-5  | Maintainability: Schema & Framework Updates | Must remain compatible with recent PyTorch versions (2.5 ~ latest stable) and the Circle schema specification. |
| RNF-6  | Usability: API Design          | Provide an intuitive API that enables transformation, optimization, and quantization with minimal code. |
| RNF-7  | Operating Environment          | Guarantee operation on Python 3.10+ and Linux-based systems. |
| RNF-8  | Compliance                     | Must comply with open-source policies and internal NPU-related regulations. |


## 4. Constraints

- **NPU Compiler Requirements**: Converted models must strictly adhere to Circle format specifications. Updates must be made promptly in response to Circle schema changes.  
- **PyTorch Version Dependency**: Support latest stable version and major LTS versions of PyTorch.  
- **Operating Environment**: Guarantee operation on Python 3.10+ and Linux-based operating systems.  
- **Compliance**: Follow open-source policies and internal NPU-related regulations.  


## 5. Appendices

- **References**
  - PyTorch official documentation
  - Circle format schema
  - NPU compiler specifications
  - Research papers on SmoothQuant, GPTQ, and other quantization techniques
