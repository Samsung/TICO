# System Test Document

## Table of Contents

- [1. Test Scope Overview](#1-test-scope-overview)
- [2. Mapping of Requirements → Test Modules](#2-mapping-of-requirements--test-modules)
- [3. Organization (Directory Layout)](#3-test-organization-directory-layout)
- [4. Example Test Flow](#4-example-test-flow)
- [5. Performance & Reliability Test Guidelines](#5-performance--reliability-test-guidelines)
- [6. Continuous Integration (CI)](#6-continuous-integration-ci)
- [7. Maintenance Process](#7-maintenance-process)
- [8. References](#8-references)


## 1. Test Scope Overview

| Scope | Description | Related Requirement IDs |
|-------|-------------|--------------------------|
| **Functional Tests** | Verify that each functional requirement is satisfied. | RF‑1, RF‑2, RF‑3, RF‑4, RF‑5 |
| **Performance Tests** | Measure conversion speed and output file size. | RNF‑1, RNF‑2 |
| **Reliability / Accuracy Tests** | Check output parity (PEIR ≤ 0.03) and robustness of quantization. | RNF‑3 |
| **Maintainability Tests** | Ensure operator coverage and compatibility with future PyTorch / Circle schema updates. | RNF‑4, RNF‑5 |
| **Usability / API Tests** | Validate that the public API is intuitive and behaves as documented. | RNF‑6 |
| **Compliance Tests** | Confirm that the library runs on Python 3.10+ and Linux, and respects open‑source licensing. | RNF‑7, RNF‑8 |


## 2. Mapping of Requirements → Test Modules

| Requirement | Test Module(s) | Test File(s) | Notes |
|-------------|----------------|--------------|-------|
| **RF‑1**: PyTorch Model Conversion | `test/pt2_to_circle_test/` | `test_pt2_to_circle.py`, `test_op.py` | End‑to‑end conversion of exported programs and PT2 models. |
| **RF‑2**: Optimization Passes | `test/unit_test/pass_test/` | `test_remove_redundant_reshape.py`, `test_fuse_redundant_reshape_to_mean.py`, … | Each pass is exercised with representative graphs. |
| **RF‑3**: Quantization Integration | `test/quantization/` | `test_propagate_quant_param.py`, `test_insert_quantize_on_dtype_mismatch.py`, `test_fold_quant_ops.py` | Covers folding, bias quantization, dtype‑mismatch insertion, and forward/backward propagation. |
| **RF‑4**: Verification & Debugging | `test/pt2_to_qcircle_test/` | `test_op.py` | Runs the generated Circle model on the reference interpreter (`circle-interpreter` or `onert`) and compares outputs. |
| **RF‑5**: NPU Compiler Compatibility | `test/pt2_to_qcircle_test/` | `test_op.py` | Uses the `onert` runtime (installed via `requirements_pre_*.txt`) to validate compatibility. |
| **RNF‑1**: Conversion Speed | `test/performance/` | `benchmark_perf.py` | Benchmark script should time `tico.convert()` on Llama‑3.2‑1B and larger models. |
| **RNF‑2**: File Size | `test/performance/` | `benchmark_perf.py` | Compare Circle file size against `torch.save(...).size`. |
| **RNF‑3**: Accuracy (PEIR ≤ 0.03) | `test/pt2_to_qcircle_test/` | `test_op.py` | Uses `tico.experimental.quantization.evaluation.metric.compute_peir` to compute PEIR for each output tensor. |
| **RNF‑4 / RNF‑5**: Operator Coverage & Schema Updates | `test/unit_test/pass_test/` & `test/unit_test/quantization_test/` | Various | New operators are added to the test suite when they are supported. |
| **RNF‑6**: API Usability | `test/README.md` (manual) & example scripts in `test/` | – | Example scripts (`dump_exported_program.py`, `dump_pt2_model.py`) demonstrate the public API. |
| **RNF‑7 / RNF‑8**: Environment & Compliance | CI workflows (`.github/workflows/*.yaml`) | – | CI runs on Linux with Python 3.10+, checks license headers. |


## 3. Test Organization (Directory Layout)

```
test/
├── README.md                # Test module description
├── dump_exported_program.py # Example for ExportedProgram dumping
├── dump_pt2_model.py        # Example for PT2 model generation
├── requirements_*.txt       # Dependency lists for different PyTorch versions
├── unit_test/
│   ├── pass_test/           # Unit tests for each conversion pass
│   └── quantization_test/   # Unit tests for quantization passes
├── pt2_to_circle_test/      # End‑to‑end conversion tests (PT2 → Circle)
├── pt2_to_qcircle_test/     # End‑to‑end conversion + quantization tests
├── performance/             # Performance tests
├── quantization/
│   └── pass/                # Quantization‑specific pass tests
└── utils/                   # Helper utilities for testing
```

*Each test file follows the standard `unittest` pattern and can be executed via*:

```bash
python -m unittest discover -s test
```


## 4. Example Test Flow

1. **Model Generation** – Create a reference PyTorch model and export it.
2. **Conversion & Optimization** – Run TICO conversion with selected passes.
3. **Runtime Validation** – Execute the generated Circle model on the target runtime and compare outputs.

Example

```python
# 1. Generate model
model = MyModel()
example_inputs = model.get_example_inputs()

# 2. Convert
circle_model = tico.convert(model, example_inputs, config=my_config)

# 3. Run on runtime
runtime = CircleRuntime()  # wraps onert or circle-interpreter
output = runtime.run(circle_model, example_inputs)

# 4. Validate PEIR
peir = compute_peir(torch_output, output)
self.assertLessEqual(peir, 0.03)
```

All functional requirements are exercised by variations of this flow.


## 5. Performance & Reliability Test Guidelines

| Test Type | Metric | Acceptance Criteria |
|-----------|--------|---------------------|
| **Speed** | Conversion time (seconds) on Llama‑3.2‑1B and Llama-3.2-3B | ≤ 60 s (180 s for 3B) (RNF‑1) |
| **Size** | Circle file size vs. original `state_dict` size | ≤ 101 % (RNF‑2) |
| **Accuracy** | PEIR per output tensor | ≤ 0.03 (RNF‑3) |
| **Stability** | Re‑run same conversion 5×, variance < 5 % | – |
| **Compatibility** | Successful execution on `onert` 0.2.0.dev* | – |

Performance test is executed with `ccex test -p`. Details are implemented in `test/peformance/benchmark_perf.py`.


## 6. Continuous Integration (CI)

- **GitHub Actions** run the full test suite on every push.
- Separate jobs for **unit**, **integration**, and **performance** tests.
- Fail the workflow if any requirement‑specific test fails or if performance thresholds are not met.


## 7. Maintenance Process

1. **When a new functional requirement is added** – create a corresponding test module under `test/` and update the mapping table above.  
2. **When a new pass is introduced** – add a unit test in `test/unit_test/pass_test/` that exercises the pass on a minimal graph.  
3. **When the Circle schema evolves** – update the compatibility tests in `test/pt2_to_qcircle_test/` and bump the version in `requirements_pre_*.txt`.  


## 8. References

- **Software Requirements Specification** – `docs/requirements.md`  
- **System Design Document** – `docs/design.md`  
