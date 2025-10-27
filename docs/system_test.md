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
- [9. Test Results Summary](#9-test-results-summary)


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
| **RNF‑3**: Accuracy (PEIR ≤ 0.03) | `test/pt2_to_qcircle_test/` | `test_op.py` | Uses `tico.quantization.evaluation.metric.compute_peir` to compute PEIR for each output tensor. |
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

## 9. Test Results Summary

This section summarizes the latest test results (as of **2025-10-13**).
See the **Full Logs** section for raw outputs and detailed per-test information.

| Test Category | Key Metric / Criteria | Result | Pass Rate | Notes |
|----------------|-----------------------|----------|------------|--------|
| **Functional Tests (RF-1 – RF-5)** | All unit tests pass | ✅ 793 passed / 0 failed (6 skipped) | **100 %** | From `./ccex test -i` run. |
| **Performance (RNF-1)** | 1B ≤ 60 s, 3B ≤ 180 s | ✅ 1B: 26.56 s / 3B: 54.47 s | **100 %** (2 / 2 targets met) | From `./ccex test -p`. |
| **File Size (RNF-2)** | Circle ≤ 101 % of state dict size | ✅ 1B: 1.00165× / 3B: 1.00132× | **100 %** (2 / 2 targets met) | Circle / state dict ratio. |
| **Accuracy (RNF-3)** | PEIR ≤ 0.03 per tensor | ✅ 793 passed / 0 failed (6 skipped) | **100 %** | From `./ccex test -i` run. |
| **Compatibility (RF-5)** | Executes on `onert 0.2.0.dev*` | ✅ 793 passed / 0 failed (6 skipped) | **100 %** | This run used `circle-interpreter` by default. |
| **Maintainability (RNF-4 / RNF-5)** | Operator coverage & schema update tests | ✅ 793 passed / 0 failed (6 skipped) | **100 %** | Covered by unit tests in this run. |
| **Usability (RNF-6)** | Public API behaves as documented | ✅ 793 passed / 0 failed (6 skipped) | **100 %** | Covered by unit tests in this run. |
| **Compliance (RNF-7 / RNF-8)** | Python 3.10+ and license checks | – | – | Verified through CI. |

### Summary Statistics
- **Total Tests Executed:** 793  
- **Passed:** 793  
- **Failed:** 0
- **Skipped:** 6  
- **Overall Pass Rate (executed):** 100 %

### Environment (this run)
- **Runtime:** `circle-interpreter` (default)  
- **Invocation:** `./ccex test -p` and `./ccex test -i`  

### Full logs

#### Performance Tests

```bash
./ccex test -p
RUN performance tests ...
Start performance test with Llama-3.2-1B
Mean conversion time (Single decoder layer * num_hidden_layers): 26.56s (threshold 60s)
Circle size: 243694812 bytes
State dict size: 243292805 bytes
Circle / State dict ratio: 1.0016523587699193
Start performance test with Llama-3.2-3B
Mean conversion time (Single decoder layer * num_hidden_layers): 54.47s (threshold 180s)
Circle size: 403217628 bytes
State dict size: 402684549 bytes
Circle / State dict ratio: 1.0013238128985178
```

#### Functional & Integration Tests

```
./ccex test -i                    
RUN ALL unit tests ...
test.modules.net.ConvEmbed.ConvEmbed (pt2_to_circle_test.test_net.test.modules.net.ConvEmbed) ... ok
test.modules.net.ConvEmbed.ConvEmbedWithKwargs (pt2_to_circle_test.test_net.test.modules.net.ConvEmbed) ... ok
test.modules.net.KVCache.KVCacheSlice (pt2_to_circle_test.test_net.test.modules.net.KVCache) ... ok
test.modules.net.KVCache.KVCacheUpdate (pt2_to_circle_test.test_net.test.modules.net.KVCache) ... ok
test.modules.net.KVCache.RepeatKV (pt2_to_circle_test.test_net.test.modules.net.KVCache) ... ok
test.modules.net.RMSNorm.LlamaRMSNorm (pt2_to_circle_test.test_net.test.modules.net.RMSNorm) ... ok
test.modules.net.RoPE.RoPE (pt2_to_circle_test.test_net.test.modules.net.RoPE) ... ok
test.modules.net.SDPA.SDPA (pt2_to_circle_test.test_net.test.modules.net.SDPA) ... ok
test.modules.net.SDPA.SDPACausal (pt2_to_circle_test.test_net.test.modules.net.SDPA) ... ok
test.modules.net.SDPA.SDPAMasked (pt2_to_circle_test.test_net.test.modules.net.SDPA) ... ok
test.modules.net.mlp.MLP (pt2_to_circle_test.test_net.test.modules.net.mlp) ... ok
test.modules.net.mlp_dyn.MLP_DynamicShape (pt2_to_circle_test.test_net.test.modules.net.mlp_dyn) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.abs.SimpleAbs (pt2_to_circle_test.test_op.test.modules.op.abs) ... ok
test.modules.op.abs.SimpleAbsWithNone (pt2_to_circle_test.test_op.test.modules.op.abs) ... ok
test.modules.op.add.AddWithBuffer (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.add.AddWithBuiltinFloat (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.add.AddWithBuiltinInt (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.add.AddWithCausalMaskFolded (pt2_to_circle_test.test_op.test.modules.op.add) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node lifted_tensor_0 target lifted_tensor_0 lifted_tensor_0 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
ok
test.modules.op.add.AddWithCausalMaskLegalized (pt2_to_circle_test.test_op.test.modules.op.add) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node lifted_tensor_0 target lifted_tensor_0 lifted_tensor_0 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
ok
test.modules.op.add.AddWithNonPersistentBuffer (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.add.ScalarAddFloat (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.add.ScalarAddInt (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.add.SimpleAdd (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.add.SimpleAddWithDifferentMemoryFormat (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.add.SimpleAddWithoutPt2 (pt2_to_circle_test.test_op.test.modules.op.add) ... ok
test.modules.op.addmm.SimpleAddmmWith1DInput (pt2_to_circle_test.test_op.test.modules.op.addmm) ... ok
test.modules.op.addmm.SimpleAddmmWith2DInput (pt2_to_circle_test.test_op.test.modules.op.addmm) ... ok
test.modules.op.addmm.SimpleAddmmWithNanInputZeroBeta (pt2_to_circle_test.test_op.test.modules.op.addmm) ... ok
test.modules.op.addmm.SimpleAddmmWithZeroAlpha (pt2_to_circle_test.test_op.test.modules.op.addmm) ... ok
test.modules.op.addmm.SimpleAddmmWithZeroAlphaAndBeta (pt2_to_circle_test.test_op.test.modules.op.addmm) ... ok
test.modules.op.addmm.SimpleAddmmWithZeroBeta (pt2_to_circle_test.test_op.test.modules.op.addmm) ... ok
test.modules.op.alias_copy.SimpleAliasCopy (pt2_to_circle_test.test_op.test.modules.op.alias_copy) ... ok
test.modules.op.alias_copy.SimpleAliasCopyWithConstantTensor (pt2_to_circle_test.test_op.test.modules.op.alias_copy) ... ok
test.modules.op.any.SimpleAnyBool (pt2_to_circle_test.test_op.test.modules.op.any) ... ok
test.modules.op.any.SimpleAnyBool2D (pt2_to_circle_test.test_op.test.modules.op.any) ... ok
test.modules.op.any.SimpleAnyBool2DKeepDimTrue (pt2_to_circle_test.test_op.test.modules.op.any) ... ok
test.modules.op.any.SimpleAnyFloat (pt2_to_circle_test.test_op.test.modules.op.any) ... ok
test.modules.op.any.SimpleAnyFloat2D (pt2_to_circle_test.test_op.test.modules.op.any) ... ok
test.modules.op.any.SimpleAnyFloat2DDim2 (pt2_to_circle_test.test_op.test.modules.op.any) ... ok
test.modules.op.any.SimpleAnyInt (pt2_to_circle_test.test_op.test.modules.op.any) ... ok
test.modules.op.any.SimpleAnyIntMinus (pt2_to_circle_test.test_op.test.modules.op.any) ... ok
test.modules.op.arange.SimpleArangeStartStepWithDifferentType (pt2_to_circle_test.test_op.test.modules.op.arange) ... ok
test.modules.op.arange.SimpleArangeStartStepWithFloat (pt2_to_circle_test.test_op.test.modules.op.arange) ... ok
test.modules.op.arange.SimpleArangeStartStepWithInt (pt2_to_circle_test.test_op.test.modules.op.arange) ... ok
test.modules.op.argmax.SimpleArgMax (pt2_to_circle_test.test_op.test.modules.op.argmax) ... ok
test.modules.op.argmax.SimpleArgMaxWithNegativeDim (pt2_to_circle_test.test_op.test.modules.op.argmax) ... ok
test.modules.op.argmax.SimpleArgMaxWithRankThreeTensor (pt2_to_circle_test.test_op.test.modules.op.argmax) ... ok
test.modules.op.avg_pool2d.AdaptiveAvgPool (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.AvgPoolFunctionalWithoutStride (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.AvgPoolNonSquareWindow (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.AvgPoolWithNoPaddingNoCountIncludePad (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.AvgPoolWithNoSamePaddingNoCountIncludePad (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... skipped 'Not supported yet'
test.modules.op.avg_pool2d.AvgPoolWithPadding (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.AvgPoolWithPaddingKwargs (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.AvgPoolWithSamePadding (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.AvgPoolWithSamePaddingNoCountIncludePad (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.AvgPoolWithoutStride (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.SimpleAvgPool (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... ok
test.modules.op.avg_pool2d.SimpleAvgPoolDynamicShape (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.avg_pool2d.SimpleAvgPoolWithAddDynamicShape (pt2_to_circle_test.test_op.test.modules.op.avg_pool2d) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.bmm.SimpleBatchMatMul (pt2_to_circle_test.test_op.test.modules.op.bmm) ... ok
test.modules.op.bmm.SimpleSingleBatchLhsConstBmm (pt2_to_circle_test.test_op.test.modules.op.bmm) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node const_lhs target const_lhs const_lhs of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.bmm.SimpleSingleBatchLhsConstBmm_NEG (pt2_to_circle_test.test_op.test.modules.op.bmm) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node const_lhs target const_lhs const_lhs of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
Fail to load model: OperationValidator failed at line 122
ok
test.modules.op.cat.SimpleCatDefault (pt2_to_circle_test.test_op.test.modules.op.cat) ... ok
test.modules.op.cat.SimpleCatThreeTensors (pt2_to_circle_test.test_op.test.modules.op.cat) ... ok
test.modules.op.cat.SimpleCatWithDim (pt2_to_circle_test.test_op.test.modules.op.cat) ... ok
test.modules.op.clamp.ClampIntInputFloatMinMax (pt2_to_circle_test.test_op.test.modules.op.clamp) ... ok
test.modules.op.clamp.DoubleClampsFrom0to6 (pt2_to_circle_test.test_op.test.modules.op.clamp) ... ok
test.modules.op.clamp.DoubleClampsFrom0to6WithBigMax (pt2_to_circle_test.test_op.test.modules.op.clamp) ... ok
test.modules.op.clamp.SimpleClampFrom0to6Float (pt2_to_circle_test.test_op.test.modules.op.clamp) ... ok
test.modules.op.clamp.SimpleClampFrom0to6Int (pt2_to_circle_test.test_op.test.modules.op.clamp) ... ok
test.modules.op.clamp.SimpleClampMaxOnly (pt2_to_circle_test.test_op.test.modules.op.clamp) ... ok
test.modules.op.clamp.SimpleClampMinMaxBoth (pt2_to_circle_test.test_op.test.modules.op.clamp) ... ok
test.modules.op.clamp.SimpleClampMinOnly (pt2_to_circle_test.test_op.test.modules.op.clamp) ... ok
test.modules.op.clone.SimpleClone (pt2_to_circle_test.test_op.test.modules.op.clone) ... ok
test.modules.op.clone.SimpleCloneWithMemoryFormat (pt2_to_circle_test.test_op.test.modules.op.clone) ... ok
test.modules.op.clone.SimpleCloneWithMemoryFormatChannelsLast (pt2_to_circle_test.test_op.test.modules.op.clone) ... ok
test.modules.op.clone.SimpleCloneWithMemoryFormatContiguous (pt2_to_circle_test.test_op.test.modules.op.clone) ... ok
test.modules.op.constant_pad_nd.ConstantPad2d (pt2_to_circle_test.test_op.test.modules.op.constant_pad_nd) ... ok
test.modules.op.constant_pad_nd.ConstantPad2dWith3dInput (pt2_to_circle_test.test_op.test.modules.op.constant_pad_nd) ... ok
test.modules.op.constant_pad_nd.DifferentRightBottom (pt2_to_circle_test.test_op.test.modules.op.constant_pad_nd) ... ok
test.modules.op.constant_pad_nd.SimpleConstantPad1DInput4D (pt2_to_circle_test.test_op.test.modules.op.constant_pad_nd) ... ok
test.modules.op.constant_pad_nd.SimpleConstantPad2DInput4D (pt2_to_circle_test.test_op.test.modules.op.constant_pad_nd) ... ok
test.modules.op.constant_pad_nd.SimpleConstantPad3DInput3D (pt2_to_circle_test.test_op.test.modules.op.constant_pad_nd) ... ok
test.modules.op.constant_pad_nd.SimpleConstantPad3DInput4D (pt2_to_circle_test.test_op.test.modules.op.constant_pad_nd) ... ok
test.modules.op.constant_pad_nd.SimpleConstantPad4DInput4D (pt2_to_circle_test.test_op.test.modules.op.constant_pad_nd) ... ok
test.modules.op.conv1d.Conv1dNoBias (pt2_to_circle_test.test_op.test.modules.op.conv1d) ... ok
test.modules.op.conv1d.Conv1dPaddingOne (pt2_to_circle_test.test_op.test.modules.op.conv1d) ... ok
test.modules.op.conv1d.Conv1dPaddingSame (pt2_to_circle_test.test_op.test.modules.op.conv1d) ... ok
test.modules.op.conv1d.Conv1dPaddingTwo (pt2_to_circle_test.test_op.test.modules.op.conv1d) ... ok
test.modules.op.conv1d.Conv1dPaddingValid (pt2_to_circle_test.test_op.test.modules.op.conv1d) ... ok
test.modules.op.conv1d.Conv1dPaddingZero (pt2_to_circle_test.test_op.test.modules.op.conv1d) ... ok
test.modules.op.conv1d.DepthwiseConv1d (pt2_to_circle_test.test_op.test.modules.op.conv1d) ... ok
test.modules.op.conv2d.ConvNonSquareKernelSamePadding (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvNonSquareKernelValidPadding (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvStirdePadding (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvValidPaddingWithNoBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithDilation (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithIntPadding (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithNoBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithNoStrideNoBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithPadding (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithSamePadding (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithSamePadding2 (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithTensorWeightAndBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.ConvWithTensorWeightNoBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.GroupedConv2dDifferentICAndOCWithTensorWeightBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.GroupedConv2dDifferentICAndOCWithTensorWeightNoBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.GroupedConv2dWithTensorWeightBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.GroupedConv2dWithTensorWeightNoBias (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.SimpleConv (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.SimpleConvDynamicShape (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.conv2d.SimpleGroupedConv2d (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.SimpleGroupedConv2dWithSamePaddingInList (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.SimpleGroupedConv2dWithSamePaddingInStr (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.SimpleGroupedConv2dWithValidPaddingInList (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.SimpleGroupedConv2dWithValidPaddingInStr (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.SimpleQuantizedConv (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv2d.TwoConvSameInput (pt2_to_circle_test.test_op.test.modules.op.conv2d) ... ok
test.modules.op.conv_transpose2d.ConvTSamePad (pt2_to_circle_test.test_op.test.modules.op.conv_transpose2d) ... ok
test.modules.op.conv_transpose2d.ConvTStride2OutPad1 (pt2_to_circle_test.test_op.test.modules.op.conv_transpose2d) ... ok
test.modules.op.conv_transpose2d.ConvTUpsample2x (pt2_to_circle_test.test_op.test.modules.op.conv_transpose2d) ... ok
test.modules.op.conv_transpose2d.SimpleConvTranspose (pt2_to_circle_test.test_op.test.modules.op.conv_transpose2d) ... ok
test.modules.op.conv_transpose2d.SimpleConvTransposeDynamicShape (pt2_to_circle_test.test_op.test.modules.op.conv_transpose2d) ... skipped 'luci-interpreter does not support dynamic shape yet && onert does not support TransposeConv yet'
test.modules.op.copy.SimpleCopy (pt2_to_circle_test.test_op.test.modules.op.copy) ... ok
test.modules.op.copy.SimpleCopyWithBroadcastTo (pt2_to_circle_test.test_op.test.modules.op.copy) ... ok
test.modules.op.cos.SimpleCos (pt2_to_circle_test.test_op.test.modules.op.cos) ... ok
test.modules.op.cumsum.SimpleCumsumDim0 (pt2_to_circle_test.test_op.test.modules.op.cumsum) ... ok
test.modules.op.cumsum.SimpleCumsumDim1 (pt2_to_circle_test.test_op.test.modules.op.cumsum) ... ok
test.modules.op.cumsum.SimpleCumsumDim2 (pt2_to_circle_test.test_op.test.modules.op.cumsum) ... ok
test.modules.op.cumsum.SimpleCumsumInt32 (pt2_to_circle_test.test_op.test.modules.op.cumsum) ... ok
test.modules.op.cumsum.SimpleCumsumInt64 (pt2_to_circle_test.test_op.test.modules.op.cumsum) ... ok
test.modules.op.depthwise_conv2d.DepthwiseConvWithTensorWeightBias (pt2_to_circle_test.test_op.test.modules.op.depthwise_conv2d) ... ok
test.modules.op.depthwise_conv2d.DepthwiseConvWithTensorWeightNoBias (pt2_to_circle_test.test_op.test.modules.op.depthwise_conv2d) ... ok
test.modules.op.depthwise_conv2d.SimpleDepthwiseConv (pt2_to_circle_test.test_op.test.modules.op.depthwise_conv2d) ... ok
test.modules.op.depthwise_conv2d.SimpleDepthwiseConvDynamicShape (pt2_to_circle_test.test_op.test.modules.op.depthwise_conv2d) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.depthwise_conv2d.SimpleDepthwiseConvWithSamePaddingInList (pt2_to_circle_test.test_op.test.modules.op.depthwise_conv2d) ... ok
test.modules.op.depthwise_conv2d.SimpleDepthwiseConvWithSamePaddingInStr (pt2_to_circle_test.test_op.test.modules.op.depthwise_conv2d) ... ok
test.modules.op.depthwise_conv2d.SimpleDepthwiseConvWithValidPaddingInList (pt2_to_circle_test.test_op.test.modules.op.depthwise_conv2d) ... ok
test.modules.op.depthwise_conv2d.SimpleDepthwiseConvWithValidPaddingInStr (pt2_to_circle_test.test_op.test.modules.op.depthwise_conv2d) ... ok
test.modules.op.detach.SimpleDetach (pt2_to_circle_test.test_op.test.modules.op.detach) ... ok
test.modules.op.detach.SimpleDetachConst (pt2_to_circle_test.test_op.test.modules.op.detach) ... ok
test.modules.op.div.DivWithDifferentDtypeInputs (pt2_to_circle_test.test_op.test.modules.op.div) ... ok
test.modules.op.div.DivWithDifferentDtypeInputs2 (pt2_to_circle_test.test_op.test.modules.op.div) ... ok
test.modules.op.div.DivWithDifferentDtypeInputs3 (pt2_to_circle_test.test_op.test.modules.op.div) ... ok
test.modules.op.div.SimpleDiv (pt2_to_circle_test.test_op.test.modules.op.div) ... ok
test.modules.op.embedding.PaddedEmbedding (pt2_to_circle_test.test_op.test.modules.op.embedding) ... ok
test.modules.op.embedding.ScaleGradEmbedding (pt2_to_circle_test.test_op.test.modules.op.embedding) ... ok
test.modules.op.embedding.SimpleEmbedding (pt2_to_circle_test.test_op.test.modules.op.embedding) ... ok
test.modules.op.embedding.SparseEmbedding (pt2_to_circle_test.test_op.test.modules.op.embedding) ... ok
test.modules.op.eq.SimpleEq (pt2_to_circle_test.test_op.test.modules.op.eq) ... ok
test.modules.op.eq.SimpleEqWithDifferentTypeScalar (pt2_to_circle_test.test_op.test.modules.op.eq) ... ok
test.modules.op.eq.SimpleEqWithEqualSign (pt2_to_circle_test.test_op.test.modules.op.eq) ... ok
test.modules.op.eq.SimpleEqWithScalarFloat (pt2_to_circle_test.test_op.test.modules.op.eq) ... ok
test.modules.op.eq.SimpleEqWithScalarInt (pt2_to_circle_test.test_op.test.modules.op.eq) ... ok
test.modules.op.exp.SimpleExp (pt2_to_circle_test.test_op.test.modules.op.exp) ... ok
test.modules.op.expand_copy.SimpleExpandCopy (pt2_to_circle_test.test_op.test.modules.op.expand_copy) ... ok
test.modules.op.expand_copy.SimpleExpandCopyMinusDim (pt2_to_circle_test.test_op.test.modules.op.expand_copy) ... ok
test.modules.op.full.SimpleFull (pt2_to_circle_test.test_op.test.modules.op.full) ... ok
test.modules.op.full.SimpleFullBool (pt2_to_circle_test.test_op.test.modules.op.full) ... ok
test.modules.op.full_like.SimpleFullLike (pt2_to_circle_test.test_op.test.modules.op.full_like) ... ok
test.modules.op.full_like.SimpleFullLikeBool (pt2_to_circle_test.test_op.test.modules.op.full_like) ... ok
test.modules.op.full_like.SimpleFullLikeWithDtypeBoolToInt (pt2_to_circle_test.test_op.test.modules.op.full_like) ... ok
test.modules.op.full_like.SimpleFullLikeWithDtypeIntToBool (pt2_to_circle_test.test_op.test.modules.op.full_like) ... ok
test.modules.op.full_like.SimpleFullLikeWithDtypeIntToFloat (pt2_to_circle_test.test_op.test.modules.op.full_like) ... ok
test.modules.op.full_like.TrivialDtypeKwargsExample (pt2_to_circle_test.test_op.test.modules.op.full_like) ... ok
test.modules.op.ge.SimpleGeWithDifferentTypeTensor (pt2_to_circle_test.test_op.test.modules.op.ge) ... ok
test.modules.op.ge.SimpleGeWithScalarFloat (pt2_to_circle_test.test_op.test.modules.op.ge) ... ok
test.modules.op.ge.SimpleGeWithScalarInt (pt2_to_circle_test.test_op.test.modules.op.ge) ... ok
test.modules.op.ge.SimpleGeWithTensor (pt2_to_circle_test.test_op.test.modules.op.ge) ... ok
test.modules.op.gelu.GeluWithApproximate (pt2_to_circle_test.test_op.test.modules.op.gelu) ... ok
test.modules.op.gelu.SimpleGelu (pt2_to_circle_test.test_op.test.modules.op.gelu) ... ok
test.modules.op.gt.SimpleGtWithDifferentTypeTensor (pt2_to_circle_test.test_op.test.modules.op.gt) ... ok
test.modules.op.gt.SimpleGtWithScalarFloat (pt2_to_circle_test.test_op.test.modules.op.gt) ... ok
test.modules.op.gt.SimpleGtWithScalarInt (pt2_to_circle_test.test_op.test.modules.op.gt) ... ok
test.modules.op.gt.SimpleGtWithTensor (pt2_to_circle_test.test_op.test.modules.op.gt) ... ok
test.modules.op.hardtanh.SimpleHardtanhFrom0to6 (pt2_to_circle_test.test_op.test.modules.op.hardtanh) ... ok
test.modules.op.index.IndexTensor2x1 (pt2_to_circle_test.test_op.test.modules.op.index) ... ok
test.modules.op.index.IndexTensorAxis0And1 (pt2_to_circle_test.test_op.test.modules.op.index) ... ok
test.modules.op.index.IndexTensorAxis1 (pt2_to_circle_test.test_op.test.modules.op.index) ... ok
test.modules.op.index.IndexTensorWithSlice (pt2_to_circle_test.test_op.test.modules.op.index) ... ok
test.modules.op.index.SimpleAtenIndexTensorAxis1 (pt2_to_circle_test.test_op.test.modules.op.index) ... ok
test.modules.op.index.SimpleAtenIndexTensorAxis2 (pt2_to_circle_test.test_op.test.modules.op.index) ... ok
test.modules.op.index.SimpleIndexTensor (pt2_to_circle_test.test_op.test.modules.op.index) ... ok
test.modules.op.index.SimpleIndexTensorBuffer (pt2_to_circle_test.test_op.test.modules.op.index) ... ok
test.modules.op.index_select.SimpleIndexSelectWithConstIndex (pt2_to_circle_test.test_op.test.modules.op.index_select) ... ok
test.modules.op.index_select.SimpleIndexSelectWithConstScalarIndex (pt2_to_circle_test.test_op.test.modules.op.index_select) ... ok
test.modules.op.index_select.SimpleIndexSelectWithDim0 (pt2_to_circle_test.test_op.test.modules.op.index_select) ... ok
test.modules.op.index_select.SimpleIndexSelectWithDim1 (pt2_to_circle_test.test_op.test.modules.op.index_select) ... ok
test.modules.op.instance_norm.SimpleInstanceNorm (pt2_to_circle_test.test_op.test.modules.op.instance_norm) ... ok
test.modules.op.instance_norm.SimpleInstanceNorm2d (pt2_to_circle_test.test_op.test.modules.op.instance_norm) ... ok
test.modules.op.interpolate.InterpolateDouble (pt2_to_circle_test.test_op.test.modules.op.interpolate) ... ok
test.modules.op.interpolate.InterpolateOnePointFive (pt2_to_circle_test.test_op.test.modules.op.interpolate) ... ok
test.modules.op.interpolate.InterpolateThreeTimes (pt2_to_circle_test.test_op.test.modules.op.interpolate) ... ok
test.modules.op.le.SimpleLeWithDifferentTypeTensor (pt2_to_circle_test.test_op.test.modules.op.le) ... ok
test.modules.op.le.SimpleLeWithScalarFloat (pt2_to_circle_test.test_op.test.modules.op.le) ... ok
test.modules.op.le.SimpleLeWithScalarInt (pt2_to_circle_test.test_op.test.modules.op.le) ... ok
test.modules.op.le.SimpleLeWithTensor (pt2_to_circle_test.test_op.test.modules.op.le) ... ok
test.modules.op.leaky_relu.SimpleLeakyRelu (pt2_to_circle_test.test_op.test.modules.op.leaky_relu) ... ok
test.modules.op.linear.FQLinearWithFp32Bias (pt2_to_circle_test.test_op.test.modules.op.linear) ... ok
test.modules.op.linear.LinearWithDictOutput (pt2_to_circle_test.test_op.test.modules.op.linear) ... ok
test.modules.op.linear.LinearWithTreeOutput (pt2_to_circle_test.test_op.test.modules.op.linear) ... ok
test.modules.op.linear.LinearWithUnusedInput (pt2_to_circle_test.test_op.test.modules.op.linear) ... ok
test.modules.op.linear.SimpleLinear (pt2_to_circle_test.test_op.test.modules.op.linear) ... ok
test.modules.op.log.SimpleLog (pt2_to_circle_test.test_op.test.modules.op.log) ... ok
test.modules.op.log1p.SimpleLog1p (pt2_to_circle_test.test_op.test.modules.op.log1p) ... ok
test.modules.op.logical_and.SimpleLogicalAnd (pt2_to_circle_test.test_op.test.modules.op.logical_and) ... ok
test.modules.op.logical_not.SimpleLogicalNot (pt2_to_circle_test.test_op.test.modules.op.logical_not) ... ok
test.modules.op.lt.SimpleLt (pt2_to_circle_test.test_op.test.modules.op.lt) ... ok
test.modules.op.lt.SimpleLtWithAngleBracket (pt2_to_circle_test.test_op.test.modules.op.lt) ... ok
test.modules.op.max_dim.MaxDimKeepDim (pt2_to_circle_test.test_op.test.modules.op.max_dim) ... ok
test.modules.op.max_dim.SimpleMaxDim (pt2_to_circle_test.test_op.test.modules.op.max_dim) ... ok
test.modules.op.max_pool2d.MaxPoolFunctionalNoStride (pt2_to_circle_test.test_op.test.modules.op.max_pool2d) ... ok
test.modules.op.max_pool2d.MaxPoolNoStride (pt2_to_circle_test.test_op.test.modules.op.max_pool2d) ... ok
test.modules.op.max_pool2d.MaxPoolNonSquareWindow (pt2_to_circle_test.test_op.test.modules.op.max_pool2d) ... ok
test.modules.op.max_pool2d.MaxPoolReturningIndices (pt2_to_circle_test.test_op.test.modules.op.max_pool2d) ... skipped 'Not Support Operator'
test.modules.op.max_pool2d.MaxPoolWithPadding (pt2_to_circle_test.test_op.test.modules.op.max_pool2d) ... ok
test.modules.op.max_pool2d.MaxPoolWithSamePadding (pt2_to_circle_test.test_op.test.modules.op.max_pool2d) ... ok
test.modules.op.max_pool2d.SimpleMaxPool (pt2_to_circle_test.test_op.test.modules.op.max_pool2d) ... ok
test.modules.op.max_pool2d.SimpleMaxPoolDynamicShape (pt2_to_circle_test.test_op.test.modules.op.max_pool2d) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.maximum.MaximumWithFloatConstRight (pt2_to_circle_test.test_op.test.modules.op.maximum) ... ok
test.modules.op.maximum.MaximumWithIntConstRight (pt2_to_circle_test.test_op.test.modules.op.maximum) ... ok
test.modules.op.maximum.MaximumWithTensorLeftConstRight (pt2_to_circle_test.test_op.test.modules.op.maximum) ... ok
test.modules.op.maximum.MaximumWithTwoInputs (pt2_to_circle_test.test_op.test.modules.op.maximum) ... ok
test.modules.op.mean.MeanWithRedundantView (pt2_to_circle_test.test_op.test.modules.op.mean) ... ok
test.modules.op.mean.MeanWithRedundantViewAndDtype (pt2_to_circle_test.test_op.test.modules.op.mean) ... ok
test.modules.op.mean.SimpleMean (pt2_to_circle_test.test_op.test.modules.op.mean) ... ok
test.modules.op.mean.SimpleMeanKeepDim (pt2_to_circle_test.test_op.test.modules.op.mean) ... ok
test.modules.op.mean.SimpleMeanNegativeTwoDim (pt2_to_circle_test.test_op.test.modules.op.mean) ... ok
test.modules.op.mean.SimpleMeanNegativeTwoDim2 (pt2_to_circle_test.test_op.test.modules.op.mean) ... ok
test.modules.op.mean.SimpleMeanTwoDim (pt2_to_circle_test.test_op.test.modules.op.mean) ... ok
test.modules.op.mean.SimpleMeanTwoDim2 (pt2_to_circle_test.test_op.test.modules.op.mean) ... ok
test.modules.op.minimum.MinimumWithFloatConstRight (pt2_to_circle_test.test_op.test.modules.op.minimum) ... ok
test.modules.op.minimum.MinimumWithIntConstRight (pt2_to_circle_test.test_op.test.modules.op.minimum) ... ok
test.modules.op.minimum.MinimumWithTensorLeftConstRight (pt2_to_circle_test.test_op.test.modules.op.minimum) ... ok
test.modules.op.minimum.MinimumWithTwoInputs (pt2_to_circle_test.test_op.test.modules.op.minimum) ... ok
test.modules.op.mm.SimpleMatmul (pt2_to_circle_test.test_op.test.modules.op.mm) ... ok
test.modules.op.mm.SimpleMatmulConstLhsOnert (pt2_to_circle_test.test_op.test.modules.op.mm) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node weight target weight weight of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
Fail to load model: OperationValidator failed at line 122
ok
test.modules.op.mm.SimpleMatmulConstLhsOnertWithLinearConversion (pt2_to_circle_test.test_op.test.modules.op.mm) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node weight target weight weight of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.mm.SimpleMatmulConstRhs (pt2_to_circle_test.test_op.test.modules.op.mm) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node weight target weight weight of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
ok
test.modules.op.mm.SimpleMatmulConstRhsOnert (pt2_to_circle_test.test_op.test.modules.op.mm) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node weight target weight weight of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.mul.MulWithBuiltinFloat (pt2_to_circle_test.test_op.test.modules.op.mul) ... ok
test.modules.op.mul.MulWithBuiltinInt (pt2_to_circle_test.test_op.test.modules.op.mul) ... ok
test.modules.op.mul.SimpleMulWithScalar (pt2_to_circle_test.test_op.test.modules.op.mul) ... ok
test.modules.op.mul.SimpleMulWithTensor (pt2_to_circle_test.test_op.test.modules.op.mul) ... ok
test.modules.op.native_batch_norm.BatchNorm2DWithNoAffine (pt2_to_circle_test.test_op.test.modules.op.native_batch_norm) ... ok
test.modules.op.native_batch_norm.NativeBatchNormLegitNoTraining (pt2_to_circle_test.test_op.test.modules.op.native_batch_norm) ... ok
test.modules.op.native_batch_norm.SimpleBatchNorm2D (pt2_to_circle_test.test_op.test.modules.op.native_batch_norm) ... ok
test.modules.op.native_group_norm.SimpleNativeGroupNorm (pt2_to_circle_test.test_op.test.modules.op.native_group_norm) ... ok
test.modules.op.native_group_norm.SimpleNativeGroupNormNonAffine (pt2_to_circle_test.test_op.test.modules.op.native_group_norm) ... ok
test.modules.op.native_group_norm.SimpleNativeGroupNormRedundantReshape (pt2_to_circle_test.test_op.test.modules.op.native_group_norm) ... ok
test.modules.op.native_group_norm.SimpleNativeGroupNormWithLayerNorm3DInput (pt2_to_circle_test.test_op.test.modules.op.native_group_norm) ... ok
test.modules.op.native_group_norm.SimpleNativeGroupNormWithLayerNormClass (pt2_to_circle_test.test_op.test.modules.op.native_group_norm) ... ok
test.modules.op.native_group_norm.SimpleNativeGroupNormWithoutWeightBias (pt2_to_circle_test.test_op.test.modules.op.native_group_norm) ... ok
test.modules.op.native_layer_norm.NativeLayerNormChannelLastInput (pt2_to_circle_test.test_op.test.modules.op.native_layer_norm) ... ok
test.modules.op.native_layer_norm.SimpleNativeLayerNorm (pt2_to_circle_test.test_op.test.modules.op.native_layer_norm) ... ok
test.modules.op.native_layer_norm.SimpleNativeLayerNormForMultiDimensionLayer (pt2_to_circle_test.test_op.test.modules.op.native_layer_norm) ... ok
test.modules.op.native_layer_norm.SimpleNativeLayerNormNonAffine (pt2_to_circle_test.test_op.test.modules.op.native_layer_norm) ... ok
test.modules.op.native_layer_norm.SimpleNativeLayerNormRedundantReshape (pt2_to_circle_test.test_op.test.modules.op.native_layer_norm) ... ok
test.modules.op.native_layer_norm.SimpleNativeLayerNormWithLayerNormClass (pt2_to_circle_test.test_op.test.modules.op.native_layer_norm) ... ok
test.modules.op.native_layer_norm.SimpleNativeLayerNormWithoutWeightBias (pt2_to_circle_test.test_op.test.modules.op.native_layer_norm) ... ok
test.modules.op.ne.SimpleNeWithDifferentTypeScalar (pt2_to_circle_test.test_op.test.modules.op.ne) ... ok
test.modules.op.ne.SimpleNeWithDifferentTypeTensor (pt2_to_circle_test.test_op.test.modules.op.ne) ... ok
test.modules.op.ne.SimpleNeWithScalarFloat (pt2_to_circle_test.test_op.test.modules.op.ne) ... ok
test.modules.op.ne.SimpleNeWithScalarInt (pt2_to_circle_test.test_op.test.modules.op.ne) ... ok
test.modules.op.ne.SimpleNeWithTensorFloat (pt2_to_circle_test.test_op.test.modules.op.ne) ... ok
test.modules.op.ne.SimpleNeWithTensorInt (pt2_to_circle_test.test_op.test.modules.op.ne) ... ok
test.modules.op.neg.SimpleNeg (pt2_to_circle_test.test_op.test.modules.op.neg) ... ok
test.modules.op.permute.PermuteWithView (pt2_to_circle_test.test_op.test.modules.op.permute) ... ok
test.modules.op.permute.PermuteWithViewContiguous (pt2_to_circle_test.test_op.test.modules.op.permute) ... ok
test.modules.op.permute.SimplePermute (pt2_to_circle_test.test_op.test.modules.op.permute) ... ok
test.modules.op.pow.Int32TensorInt64Scalar (pt2_to_circle_test.test_op.test.modules.op.pow) ... ok
test.modules.op.pow.SimplePowFloatTensorIntScalar (pt2_to_circle_test.test_op.test.modules.op.pow) ... ok
test.modules.op.pow.SimplePowFloatTensorIntScalar2 (pt2_to_circle_test.test_op.test.modules.op.pow) ... ok
test.modules.op.pow.SimplePowFloatTensorIntTensor (pt2_to_circle_test.test_op.test.modules.op.pow) ... ok
test.modules.op.pow.SimplePowIntTensorFloatScalar (pt2_to_circle_test.test_op.test.modules.op.pow) ... ok
test.modules.op.pow.SimplePowIntTensorFloatTensor (pt2_to_circle_test.test_op.test.modules.op.pow) ... ok
test.modules.op.pow.SimplePowTensorScalar (pt2_to_circle_test.test_op.test.modules.op.pow) ... ok
test.modules.op.pow.SimplePowTensorTensor (pt2_to_circle_test.test_op.test.modules.op.pow) ... ok
test.modules.op.prelu.SimplePReLU (pt2_to_circle_test.test_op.test.modules.op.prelu) ... ok
test.modules.op.reciprocal.SimpleReciprocal2 (pt2_to_circle_test.test_op.test.modules.op.reciprocal) ... ok
test.modules.op.reciprocal.SimpleReciprocalOperator (pt2_to_circle_test.test_op.test.modules.op.reciprocal) ... ok
test.modules.op.relu.SimpleRelu (pt2_to_circle_test.test_op.test.modules.op.relu) ... ok
test.modules.op.relu6.SimpleReLU6 (pt2_to_circle_test.test_op.test.modules.op.relu6) ... ok
test.modules.op.repeat.RepeatLongerRepeats (pt2_to_circle_test.test_op.test.modules.op.repeat) ... skipped 'Not Support Operator'
test.modules.op.repeat.RepeatThreetimesHW (pt2_to_circle_test.test_op.test.modules.op.repeat) ... ok
test.modules.op.repeat.RepeatTwiceH (pt2_to_circle_test.test_op.test.modules.op.repeat) ... ok
test.modules.op.repeat.RepeatTwiceHThreeTimesW (pt2_to_circle_test.test_op.test.modules.op.repeat) ... ok
test.modules.op.repeat.RepeatTwiceHW (pt2_to_circle_test.test_op.test.modules.op.repeat) ... ok
test.modules.op.repeat.SimpleRepeat (pt2_to_circle_test.test_op.test.modules.op.repeat) ... ok
test.modules.op.repeat.SimpleRepeat2 (pt2_to_circle_test.test_op.test.modules.op.repeat) ... ok
test.modules.op.repeat.SimpleRepeatDynamicShape (pt2_to_circle_test.test_op.test.modules.op.repeat) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.reshape.ReshapeChannelLastTensor (pt2_to_circle_test.test_op.test.modules.op.reshape) ... ok
test.modules.op.reshape.ReshapeTorchAPI (pt2_to_circle_test.test_op.test.modules.op.reshape) ... ok
test.modules.op.reshape.SimpleReshapeFirstDimMinus (pt2_to_circle_test.test_op.test.modules.op.reshape) ... ok
test.modules.op.reshape.SimpleReshapeLastDimMinus (pt2_to_circle_test.test_op.test.modules.op.reshape) ... ok
test.modules.op.round.SimpleRound (pt2_to_circle_test.test_op.test.modules.op.round) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/onert/common/basesession.py:126: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed. __array__ must implement 'dtype' and 'copy' keyword arguments. To learn more, see the migration guide https://numpy.org/devdocs/numpy_2_0_migration_guide.html#adapting-to-changes-in-the-copy-keyword
  input_array = np.array(inputs_array[i], dtype=input_tensorinfo.dtype)
ok
test.modules.op.rsqrt.SimpleRsqrt (pt2_to_circle_test.test_op.test.modules.op.rsqrt) ... ok
test.modules.op.scalar_tensor.SimpleScalarTensor (pt2_to_circle_test.test_op.test.modules.op.scalar_tensor) ... ok
test.modules.op.scalar_tensor.SimpleScalarTensorBool (pt2_to_circle_test.test_op.test.modules.op.scalar_tensor) ... ok
test.modules.op.scalar_tensor.SimpleScalarTensorInt (pt2_to_circle_test.test_op.test.modules.op.scalar_tensor) ... ok
test.modules.op.select.SimpleConstIndex (pt2_to_circle_test.test_op.test.modules.op.select) ... ok
test.modules.op.select.SimpleSelect (pt2_to_circle_test.test_op.test.modules.op.select) ... ok
test.modules.op.select.SimpleSelect2 (pt2_to_circle_test.test_op.test.modules.op.select) ... ok
test.modules.op.select_copy.SimpleSelectCopy (pt2_to_circle_test.test_op.test.modules.op.select_copy) ... ok
test.modules.op.select_copy.SimpleSelectCopy2 (pt2_to_circle_test.test_op.test.modules.op.select_copy) ... ok
test.modules.op.sigmoid.SimpleSigmoid (pt2_to_circle_test.test_op.test.modules.op.sigmoid) ... ok
test.modules.op.sin.SimpleSin (pt2_to_circle_test.test_op.test.modules.op.sin) ... ok
test.modules.op.slice_copy.SimpleSliceCopy (pt2_to_circle_test.test_op.test.modules.op.slice_copy) ... ok
test.modules.op.slice_copy.SimpleSliceCopyWithInvalidArgs (pt2_to_circle_test.test_op.test.modules.op.slice_copy) ... ok
test.modules.op.slice_copy.SimpleSliceCopyWithMinus (pt2_to_circle_test.test_op.test.modules.op.slice_copy) ... ok
test.modules.op.slice_copy.SimpleSliceCopyWithOutOfBound (pt2_to_circle_test.test_op.test.modules.op.slice_copy) ... ok
test.modules.op.slice_copy.SimpleSliceOperator (pt2_to_circle_test.test_op.test.modules.op.slice_copy) ... ok
test.modules.op.slice_copy.SimpleSliceOperatorWithMinus (pt2_to_circle_test.test_op.test.modules.op.slice_copy) ... ok
test.modules.op.slice_copy.SimpleSliceOperatorWithOutOfBound (pt2_to_circle_test.test_op.test.modules.op.slice_copy) ... ok
test.modules.op.slice_scatter.SimpleScatterCopyDim0 (pt2_to_circle_test.test_op.test.modules.op.slice_scatter) ... ok
test.modules.op.slice_scatter.SimpleScatterCopyDim1 (pt2_to_circle_test.test_op.test.modules.op.slice_scatter) ... ok
test.modules.op.slice_scatter.SimpleScatterCopyDim2 (pt2_to_circle_test.test_op.test.modules.op.slice_scatter) ... ok
test.modules.op.softmax.SimpleSafeSoftMax (pt2_to_circle_test.test_op.test.modules.op.softmax) ... ok
test.modules.op.softmax.SimpleSoftMax (pt2_to_circle_test.test_op.test.modules.op.softmax) ... ok
test.modules.op.softmax.SimpleSoftMaxDimMinus (pt2_to_circle_test.test_op.test.modules.op.softmax) ... ok
test.modules.op.split_with_sizes.SimpleSplitWithSizes (pt2_to_circle_test.test_op.test.modules.op.split_with_sizes) ... ok
test.modules.op.split_with_sizes.SimpleSplitWithSizesCopy (pt2_to_circle_test.test_op.test.modules.op.split_with_sizes) ... ok
test.modules.op.split_with_sizes.SimpleSplitWithSizesCopyWithDim1 (pt2_to_circle_test.test_op.test.modules.op.split_with_sizes) ... ok
test.modules.op.split_with_sizes.SimpleSplitWithSizesWithDim1 (pt2_to_circle_test.test_op.test.modules.op.split_with_sizes) ... ok
test.modules.op.sqrt.SimpleSqrt (pt2_to_circle_test.test_op.test.modules.op.sqrt) ... ok
test.modules.op.squeeze.SimpleSqueeze (pt2_to_circle_test.test_op.test.modules.op.squeeze) ... ok
test.modules.op.squeeze.SimpleSqueezeWithDims (pt2_to_circle_test.test_op.test.modules.op.squeeze) ... ok
test.modules.op.squeeze.SimpleSqueezeWithSingleDim (pt2_to_circle_test.test_op.test.modules.op.squeeze) ... ok
test.modules.op.sub.SimpleSub (pt2_to_circle_test.test_op.test.modules.op.sub) ... ok
test.modules.op.sub.SubWithBuiltinFloat (pt2_to_circle_test.test_op.test.modules.op.sub) ... ok
test.modules.op.sub.SubWithBuiltinInt (pt2_to_circle_test.test_op.test.modules.op.sub) ... ok
test.modules.op.sub.SubWithOut (pt2_to_circle_test.test_op.test.modules.op.sub) ... ok
test.modules.op.sum.SimpleSumDim2Int (pt2_to_circle_test.test_op.test.modules.op.sum) ... skipped 'Not supported yet'
test.modules.op.sum.SimpleSumDim2Keepdim (pt2_to_circle_test.test_op.test.modules.op.sum) ... ok
test.modules.op.sum.SimpleSumDimMinus1 (pt2_to_circle_test.test_op.test.modules.op.sum) ... ok
test.modules.op.sum.SimpleSumDimMinus1With3D (pt2_to_circle_test.test_op.test.modules.op.sum) ... ok
test.modules.op.tanh.SimpleTanh (pt2_to_circle_test.test_op.test.modules.op.tanh) ... ok
test.modules.op.to.RedundantDeviceToCopy (pt2_to_circle_test.test_op.test.modules.op.to) ... ok
test.modules.op.to.RedundantDtypeToCopy (pt2_to_circle_test.test_op.test.modules.op.to) ... ok
test.modules.op.to.ToWithIntegerPlusFloat (pt2_to_circle_test.test_op.test.modules.op.to) ... ok
test.modules.op.to.ToWithIntegerType (pt2_to_circle_test.test_op.test.modules.op.to) ... ok
test.modules.op.to.ToWithMemoryFormat (pt2_to_circle_test.test_op.test.modules.op.to) ... skipped 'Not Support Operator'
test.modules.op.to_dim_order_copy.SimpleToF32I32 (pt2_to_circle_test.test_op.test.modules.op.to_dim_order_copy) ... ok
test.modules.op.to_dim_order_copy.SimpleToI32F32 (pt2_to_circle_test.test_op.test.modules.op.to_dim_order_copy) ... ok
test.modules.op.unsqueeze.SimpleUnsqueeze (pt2_to_circle_test.test_op.test.modules.op.unsqueeze) ... ok
test.modules.op.view.SimpleView (pt2_to_circle_test.test_op.test.modules.op.view) ... ok
test.modules.op.view.SimpleViewFirstDimMinus (pt2_to_circle_test.test_op.test.modules.op.view) ... ok
test.modules.op.view.SimpleViewLastDimMinus (pt2_to_circle_test.test_op.test.modules.op.view) ... ok
test.modules.op.where.SimpleWhereWithConstantTensor (pt2_to_circle_test.test_op.test.modules.op.where) ... ok
test.modules.op.where.SimpleWhereWithScalar (pt2_to_circle_test.test_op.test.modules.op.where) ... ok
test.modules.op.where.SimpleWhereWithTensor (pt2_to_circle_test.test_op.test.modules.op.where) ... ok
test.modules.op.add.SimpleAdd (pt2_to_qcircle_test.test_op.test.modules.op.add) ... ok
test.modules.op.conv2d.SimpleQuantizedConv (pt2_to_qcircle_test.test_op.test.modules.op.conv2d) ... ok
test_model (quantization.algorithm.test_gptq.GPTQTest) ... You are using the default legacy behaviour of the <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file you can ignore this message.
ok
test_net (quantization.algorithm.test_gptq.GPTQTest) ... ok
test_output_hook_on_fairseq_like_relu_bridge (quantization.algorithm.test_smooth_quant.SmoothQuantOutputHookTest)
Verifies that: ... ok
test_value (quantization.algorithm.test_smooth_quant.SmoothQuantTest) ... ok
test_convert_to_torch_tensor_mixed (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test _convert_to_torch_tensor with mixed numpy and torch tensors ... ok
test_convert_to_torch_tensor_numpy (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test _convert_to_torch_tensor with numpy arrays ... ok
test_evaluate_invalid_backend (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test evaluate with invalid backend ... ok
test_evaluate_invalid_circle_model (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test evaluate with invalid circle model ... ok
test_evaluate_invalid_mode (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test evaluate with invalid mode ... ok
test_evaluate_invalid_torch_module (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test evaluate with invalid torch module ... ok
test_validate_input_data_invalid_types (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test _validate_input_data with invalid types ... ok
test_validate_input_data_length_mismatch (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test _validate_input_data with length mismatch ... ok
test_validate_input_data_none (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test _validate_input_data with None input ... ok
test_validate_input_data_valid (quantization.evaluation.test_evaluation.TestEvaluateFunctions)
Test _validate_input_data with valid input ... ok
test_dequantize_int16 (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test dequantize function with int16 dtype ... ok
test_dequantize_invalid_dtype (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test dequantize function with invalid dtype ... ok
test_dequantize_uint8 (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test dequantize function with uint8 dtype ... ok
test_ensure_list_list (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test ensure_list with list (should return unchanged) ... ok
test_ensure_list_single_element (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test ensure_list with single element ... ok
test_ensure_list_tuple (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test ensure_list with tuple ... ok
test_find_invalid_types_all_valid (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test find_invalid_types with all valid types ... ok
test_find_invalid_types_duplicates (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test find_invalid_types handles duplicate invalid types ... ok
test_find_invalid_types_mixed (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test find_invalid_types with mixed valid and invalid types ... ok
test_get_graph_input_output (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test get_graph_input_output function ... ok
test_plot_two_outputs (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test plot_two_outputs function ... ok
test_quantize_int16 (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test quantize function with int16 dtype ... ok
test_quantize_invalid_dtype (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test quantize function with invalid dtype ... ok
test_quantize_uint8 (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test quantize function with uint8 dtype ... ok
test_quantize_zero_scale (quantization.evaluation.test_evaluation.TestEvaluationUtils)
Test quantize function with zero scale ... /home/seongwoo/TICO/tico/quantization/evaluation/utils.py:47: DeprecationWarning: The 'warn' method is deprecated, use 'warning' instead
  logger.warn("WARNING: scale value is 0. 1e-7 will be used instead.")
WARNING:tico.quantization.evaluation.utils:WARNING: scale value is 0. 1e-7 will be used instead.
ok
test_build_fqn_map (quantization.evaluation.test_evaluation.TestIntrospectionUtils)
Test build_fqn_map function ... ok
test_build_fqn_map_nested (quantization.evaluation.test_evaluation.TestIntrospectionUtils)
Test build_fqn_map with nested modules ... ok
test_compare_layer_outputs_collect_mode (quantization.evaluation.test_evaluation.TestIntrospectionUtils)
Test compare_layer_outputs in collect mode ... ok
test_compare_layer_outputs_no_reference (quantization.evaluation.test_evaluation.TestIntrospectionUtils)
Test compare_layer_outputs with no cached reference ... ok
test_compare_layer_outputs_print_mode (quantization.evaluation.test_evaluation.TestIntrospectionUtils)
Test compare_layer_outputs in print mode ... ok
test_save_fp_outputs (quantization.evaluation.test_evaluation.TestIntrospectionUtils)
Test save_fp_outputs function ... ok
test_save_fp_outputs_no_fp_name (quantization.evaluation.test_evaluation.TestIntrospectionUtils)
Test save_fp_outputs with module that has no fp_name ... ok
test_squeeze (quantization.pass.test_convert_layout_op_to_reshape.ConvertLayoutOpToReshapeTest) ... ok
test_unsqueeze (quantization.pass.test_convert_layout_op_to_reshape.ConvertLayoutOpToReshapeTest) ... ok
test_view (quantization.pass.test_convert_layout_op_to_reshape.ConvertLayoutOpToReshapeTest) ... ok
test_pass (quantization.pass.test_fold_quant_ops.FoldQuantOpsTest) ... ok
test_requantize (quantization.pass.test_fold_quant_ops.FoldQuantOpsTest) ... ok
test_i16_to_u8_add (quantization.pass.test_insert_quantize_on_dtype_mismatch.AddTest) ... ok
test_i16o8 (quantization.pass.test_insert_quantize_on_dtype_mismatch.AddTest) ... ok
test_mismatch_input_dtypes_add (quantization.pass.test_insert_quantize_on_dtype_mismatch.AddTest) ... ok
test_no_mismatch_add (quantization.pass.test_insert_quantize_on_dtype_mismatch.AddTest) ... ok
test_unsupported_add_dtype (quantization.pass.test_insert_quantize_on_dtype_mismatch.AddTest) ... ok
test_no_mismatch_bmm (quantization.pass.test_insert_quantize_on_dtype_mismatch.BMMDtypeMismatchTest) ... ok
test_u8_to_i16_bmm (quantization.pass.test_insert_quantize_on_dtype_mismatch.BMMDtypeMismatchTest) ... ok
test_unsupported_bmm_dtype (quantization.pass.test_insert_quantize_on_dtype_mismatch.BMMDtypeMismatchTest) ... ok
test_i16o8 (quantization.pass.test_insert_quantize_on_dtype_mismatch.BMMTest) ... ok
test_i8o16 (quantization.pass.test_insert_quantize_on_dtype_mismatch.BMMTest) ... ok
test_i16_to_u8_cat (quantization.pass.test_insert_quantize_on_dtype_mismatch.CatTest) ... ok
test_no_mismatch_cat (quantization.pass.test_insert_quantize_on_dtype_mismatch.CatTest) ... ok
test_unsupported_cat_dtype (quantization.pass.test_insert_quantize_on_dtype_mismatch.CatTest) ... ok
test_no_mismatch_linear (quantization.pass.test_insert_quantize_on_dtype_mismatch.LinearDtypeMismatchTest) ... ok
test_u8_to_i16_linear (quantization.pass.test_insert_quantize_on_dtype_mismatch.LinearDtypeMismatchTest) ... ok
test_unsupported_linear_dtype (quantization.pass.test_insert_quantize_on_dtype_mismatch.LinearDtypeMismatchTest) ... ok
test_i8o16 (quantization.pass.test_insert_quantize_on_dtype_mismatch.LinearTest) ... ok
test_no_mismatch_mul (quantization.pass.test_insert_quantize_on_dtype_mismatch.MulDtypeMismatchTest) ... ok
test_unsupported_mul_dtype (quantization.pass.test_insert_quantize_on_dtype_mismatch.MulDtypeMismatchTest) ... ok
test_i16o8 (quantization.pass.test_insert_quantize_on_dtype_mismatch.MulTest) ... ok
test_no_mismatch_permute (quantization.pass.test_insert_quantize_on_dtype_mismatch.PermuteDtypeMismatchTest) ... ok
test_u8_to_i16_permute (quantization.pass.test_insert_quantize_on_dtype_mismatch.PermuteDtypeMismatchTest) ... ok
test_unsupported_permute_dtype (quantization.pass.test_insert_quantize_on_dtype_mismatch.PermuteDtypeMismatchTest) ... ok
test_i16o8 (quantization.pass.test_insert_quantize_on_dtype_mismatch.PermuteTest) ... ok
test_i8o16 (quantization.pass.test_insert_quantize_on_dtype_mismatch.PermuteTest) ... ok
test_no_mismatch_relu (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReluDtypeMismatchTest) ... ok
test_u8_to_i16_relu (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReluDtypeMismatchTest) ... ok
test_unsupported_relu_dtype (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReluDtypeMismatchTest) ... ok
test_i16o8 (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReluTest) ... ok
test_i8o16 (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReluTest) ... ok
test_no_mismatch_reshape (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReshapeDtypeMismatchTest) ... ok
test_u8_to_i16_reshape (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReshapeDtypeMismatchTest) ... ok
test_unsupported_reshape_dtype (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReshapeDtypeMismatchTest) ... ok
test_i16o8 (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReshapeTest) ... ok
test_i8o16 (quantization.pass.test_insert_quantize_on_dtype_mismatch.ReshapeTest) ... ok
test_s16 (quantization.pass.test_propagate_qparam_backward.CatTest) ... ok
test_u8 (quantization.pass.test_propagate_qparam_backward.CatTest) ... ok
test_s16 (quantization.pass.test_propagate_qparam_backward.PermuteTest) ... ok
test_u8 (quantization.pass.test_propagate_qparam_backward.PermuteTest) ... ok
test_s16 (quantization.pass.test_propagate_qparam_backward.ReshapeTest) ... ok
test_u8 (quantization.pass.test_propagate_qparam_backward.ReshapeTest) ... ok
test_s16 (quantization.pass.test_propagate_quant_param.CatTest) ... ok
test_s16_different_scale (quantization.pass.test_propagate_quant_param.CatTest) ... ok
test_s16 (quantization.pass.test_propagate_quant_param.NegTest) ... ok
test_s16 (quantization.pass.test_propagate_quant_param.PermuteTest) ... ok
test_u8 (quantization.pass.test_propagate_quant_param.PermuteTest) ... ok
test_pass (quantization.pass.test_propagate_quant_param.PropagateQParamForwardTest) ... ok
test_s16 (quantization.pass.test_propagate_quant_param.ReshapeTest) ... ok
test_u8 (quantization.pass.test_propagate_quant_param.ReshapeTest) ... ok
test_s16 (quantization.pass.test_propagate_quant_param.SliceTest) ... ok
test_u8 (quantization.pass.test_propagate_quant_param.SliceTest) ... ok
test_pass (quantization.pass.test_remove_weight_dequant_op.RemoveWeightDequantOpTest) ... ok
test_degenerate_constant_cases (quantization.wrapq.observers.test_affine_base.TestAffineObserverBase) ... ok
test_fake_quant_requires_qparams (quantization.wrapq.observers.test_affine_base.TestAffineObserverBase) ... ok
test_load_qparams_and_fake_quant (quantization.wrapq.observers.test_affine_base.TestAffineObserverBase) ... ok
test_per_channel_asymm_stats_and_qparams (quantization.wrapq.observers.test_affine_base.TestAffineObserverBase) ... ok
test_per_channel_fake_quant_path (quantization.wrapq.observers.test_affine_base.TestAffineObserverBase) ... ok
test_per_tensor_asymm_qparams (quantization.wrapq.observers.test_affine_base.TestAffineObserverBase) ... ok
test_per_tensor_symmetric_qparams (quantization.wrapq.observers.test_affine_base.TestAffineObserverBase) ... ok
test_reset_clears_minmax_and_qparams (quantization.wrapq.observers.test_affine_base.TestAffineObserverBase) ... ok
test_collect_respects_enabled_flag (quantization.wrapq.observers.test_base.TestObserverBase) ... ok
test_compute_qparams_contract_allows_none (quantization.wrapq.observers.test_base.TestObserverBase) ... ok
test_repr_smoke (quantization.wrapq.observers.test_base.TestObserverBase) ... ok
test_ema_updates (quantization.wrapq.observers.test_ema.TestEMAObserver) ... ok
test_first_batch (quantization.wrapq.observers.test_ema.TestEMAObserver) ... ok
test_per_channel_ema (quantization.wrapq.observers.test_ema.TestEMAObserver) ... ok
test_collect_is_noop (quantization.wrapq.observers.test_identity.TestIdentityObserver) ... ok
test_compute_qparams_constant (quantization.wrapq.observers.test_identity.TestIdentityObserver) ... ok
test_fake_quant_identity (quantization.wrapq.observers.test_identity.TestIdentityObserver) ... ok
test_initial_state (quantization.wrapq.observers.test_identity.TestIdentityObserver) ... ok
test_per_channel_minmax (quantization.wrapq.observers.test_minmax.TestMinMaxObserver) ... ok
test_per_tensor_minmax (quantization.wrapq.observers.test_minmax.TestMinMaxObserver) ... ok
test_reset (quantization.wrapq.observers.test_minmax.TestMinMaxObserver) ... ok
test_negative_scalar_uint8 (quantization.wrapq.observers.test_minmax.TestScalarObserver) ... ok
test_positive_scalar_uint8 (quantization.wrapq.observers.test_minmax.TestScalarObserver) ... ok
test_zero_scalar_uint8 (quantization.wrapq.observers.test_minmax.TestScalarObserver) ... ok
test_axis_is_independent_from_base_channel_axis (quantization.wrapq.observers.test_mx.TestMXObserver)
MXObserver.axis must be used for shared-exponent grouping regardless of base.channel_axis. ... ok
test_compute_qparams_returns_none_and_collect_noop (quantization.wrapq.observers.test_mx.TestMXObserver)
MXObserver does not produce affine qparams; compute_qparams() returns None. ... ok
test_fake_quant_calls_quantize_mx_with_expected_args (quantization.wrapq.observers.test_mx.TestMXObserver)
fake_quant(x) must delegate to quantize_mx with the configured arguments. ... ok
test_fake_quant_still_runs_when_disabled (quantization.wrapq.observers.test_mx.TestMXObserver)
Even when 'enabled' is False (no more stats collection), fake_quant should still run. ... ok
test_repr_smoke (quantization.wrapq.observers.test_mx.TestMXObserver)
repr() should include class name and observer name for debugging. ... ok
test_presets (quantization.wrapq.test_dtype.TestDType) ... ok
test_range_signed (quantization.wrapq.test_dtype.TestDType) ... ok
test_range_unsigned (quantization.wrapq.test_dtype.TestDType) ... ok
test_str (quantization.wrapq.test_dtype.TestDType) ... ok
test_member_names (quantization.wrapq.test_mode.TestModeEnum) ... ok
test_ordering (quantization.wrapq.test_mode.TestModeEnum) ... ok
test_str_representation (quantization.wrapq.test_mode.TestModeEnum) ... ok
test_enum_members (quantization.wrapq.test_qscheme.TestQScheme) ... ok
test_is_per_channel (quantization.wrapq.test_qscheme.TestQScheme) ... ok
test_is_symmetric (quantization.wrapq.test_qscheme.TestQScheme) ... ok
test_str (quantization.wrapq.test_qscheme.TestQScheme) ... ok
test_config_default_when_neither_user_nor_wrapper (quantization.wrapq.test_quant_config.TestObserverAndDTypePrecedence)
If neither user nor wrapper provides dtype/qscheme, fallback to QuantConfig defaults. ... ok
test_other_kwargs_user_override_precedence (quantization.wrapq.test_quant_config.TestObserverAndDTypePrecedence)
For keys without QuantConfig-level defaults (like qscheme/channel_axis), ... ok
test_quantconfig_get_kwargs_does_not_inject_dtype (quantization.wrapq.test_quant_config.TestObserverAndDTypePrecedence)
Ensure QuantConfig.get_kwargs() itself doesn't inject dtype anymore. ... ok
test_user_override_wins (quantization.wrapq.test_quant_config.TestObserverAndDTypePrecedence)
If user supplies both dtype and observer, they must override ... ok
test_wrapper_default_when_no_user_override (quantization.wrapq.test_quant_config.TestObserverAndDTypePrecedence)
If the user supplies nothing for a given name, wrapper defaults must ... ok
test_default_dtype_applied (quantization.wrapq.test_quant_config.TestQuantConfig) ... ok
test_observer_override (quantization.wrapq.test_quant_config.TestQuantConfig) ... ok
test_per_observer_dtype_override (quantization.wrapq.test_quant_config.TestQuantConfig) ... ok
test_child_inherits_default_dtype (quantization.wrapq.test_quant_config.TestQuantConfigChild) ... ok
test_child_inherits_default_qscheme (quantization.wrapq.test_quant_config.TestQuantConfigChild) ... ok
test_child_is_view_not_copy (quantization.wrapq.test_quant_config.TestQuantConfigChild) ... ok
test_child_override_applied (quantization.wrapq.test_quant_config.TestQuantConfigChild) ... ok
test_default_qscheme_applied (quantization.wrapq.test_quant_config.TestQuantConfigQScheme) ... ok
test_per_observer_qscheme_override (quantization.wrapq.test_quant_config.TestQuantConfigQScheme) ... ok
test_bidirectional_consistency (quantization.wrapq.utils.test_introspection.TestBuildFqnMap) ... ok
test_direct_child_name (quantization.wrapq.utils.test_introspection.TestBuildFqnMap) ... ok
test_root_included (quantization.wrapq.utils.test_introspection.TestBuildFqnMap) ... ok
test_sequential_children (quantization.wrapq.utils.test_introspection.TestBuildFqnMap) ... ok
test_total_entries (quantization.wrapq.utils.test_introspection.TestBuildFqnMap) ... ok
test_custom_metric (quantization.wrapq.utils.test_introspection.TestSmoothQuantPTQDiff) ... ok
test_layerwise_diff (quantization.wrapq.utils.test_introspection.TestSmoothQuantPTQDiff) ... ok
test_layerwise_peir (quantization.wrapq.utils.test_introspection.TestSmoothQuantPTQDiff) ... ok
test_metric_subset_selection (quantization.wrapq.utils.test_introspection.TestSmoothQuantPTQDiff) ... ok
test_non_default_ignore_index (quantization.wrapq.utils.test_metrics.TestPerplexitySlidingWindow) ... ok
test_returns_positive_float (quantization.wrapq.utils.test_metrics.TestPerplexitySlidingWindow) ... ok
test_short_sequence_equivalence (quantization.wrapq.utils.test_metrics.TestPerplexitySlidingWindow) ... ok
test_stride_invariance_short (quantization.wrapq.utils.test_metrics.TestPerplexitySlidingWindow) ... ok
test_keep_dim0 (quantization.wrapq.utils.test_reduce_utils.TestChannelwiseMinMax) ... ok
test_keep_middle_dim (quantization.wrapq.utils.test_reduce_utils.TestChannelwiseMinMax) ... ok
test_keep_negative_index (quantization.wrapq.utils.test_reduce_utils.TestChannelwiseMinMax) ... ok
test_export_decoder_single_step_invokes_convert (quantization.wrapq.wrappers.fairseq.test_decoder_export_single_step.TestDecoderExportSingleStep) ... Saved decoder single-step export to: dummy.circle
ok
test_forward_shapes (quantization.wrapq.wrappers.fairseq.test_decoder_export_single_step.TestDecoderExportSingleStep) ... ok
test_kv_arg_count_mismatch (quantization.wrapq.wrappers.fairseq.test_decoder_export_single_step.TestDecoderExportSingleStep) ... ok
test_make_example_inputs_shapes (quantization.wrapq.wrappers.fairseq.test_decoder_export_single_step.TestDecoderExportSingleStep) ... ok
test_meta_inference (quantization.wrapq.wrappers.fairseq.test_decoder_export_single_step.TestDecoderExportSingleStep) ... ok
test_buffered_future_mask_shape_and_values (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_extract_features_alignment (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_forward_external_step (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_forward_logits_and_extra (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_get_normalized_probs (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_incremental_decoding_two_steps (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_lifecycle_propagation (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_max_positions (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_reorder_incremental_state_scripting_noop (quantization.wrapq.wrappers.fairseq.test_quant_decoder.TestQuantFairseqDecoder) ... ok
test_cross_self_attention_path (quantization.wrapq.wrappers.fairseq.test_quant_decoder_layer.TestQuantFairseqDecoderLayer) ... ok
test_forward_external_single_step (quantization.wrapq.wrappers.fairseq.test_quant_decoder_layer.TestQuantFairseqDecoderLayer) ... ok
test_forward_shapes_tuple (quantization.wrapq.wrappers.fairseq.test_quant_decoder_layer.TestQuantFairseqDecoderLayer) ... ok
test_lifecycle_propagation (quantization.wrapq.wrappers.fairseq.test_quant_decoder_layer.TestQuantFairseqDecoderLayer) ... ok
test_maybe_apply_head_scale (quantization.wrapq.wrappers.fairseq.test_quant_decoder_layer.TestQuantFairseqDecoderLayer) ... ok
test_need_head_weights_shape (quantization.wrapq.wrappers.fairseq.test_quant_decoder_layer.TestQuantFairseqDecoderLayer) ... ok
test_prev_attn_state_static_kv_reuse (quantization.wrapq.wrappers.fairseq.test_quant_decoder_layer.TestQuantFairseqDecoderLayer) ... ok
test_prev_self_attn_state_accumulate (quantization.wrapq.wrappers.fairseq.test_quant_decoder_layer.TestQuantFairseqDecoderLayer) ... ok
test_external_inputs_dict_return (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_external_inputs_tensor_return (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_forward_standard_dict_shapes (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_forward_torchscript_wrapper (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_lifecycle_propagation (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_max_positions_with_and_without_positional (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_numerical_sanity_with_or_without_embed_scale (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_padding_zeroing_before_stack (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_reorder_encoder_out (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_upgrade_state_dict_named_behavior (quantization.wrapq.wrappers.fairseq.test_quant_encoder.TestQuantFairseqEncoder) ... ok
test_additive_float_masks (quantization.wrapq.wrappers.fairseq.test_quant_encoder_layer.TestQuantFairseqEncoderLayer) ... ok
test_forward_shapes_flags (quantization.wrapq.wrappers.fairseq.test_quant_encoder_layer.TestQuantFairseqEncoderLayer) ... ok
test_lifecycle_propagation (quantization.wrapq.wrappers.fairseq.test_quant_encoder_layer.TestQuantFairseqEncoderLayer) ... ok
test_mask_handling (quantization.wrapq.wrappers.fairseq.test_quant_encoder_layer.TestQuantFairseqEncoderLayer) ... ok
test_numerical_sanity (quantization.wrapq.wrappers.fairseq.test_quant_encoder_layer.TestQuantFairseqEncoderLayer) ... ok
test_qcfg_child_overrides_for_activation_observer (quantization.wrapq.wrappers.fairseq.test_quant_encoder_layer.TestQuantFairseqEncoderLayer) ... ok
test_return_fc_semantics_pre_norm (quantization.wrapq.wrappers.fairseq.test_quant_encoder_layer.TestQuantFairseqEncoderLayer) ... ok
test_submodules_wrapped (quantization.wrapq.wrappers.fairseq.test_quant_encoder_layer.TestQuantFairseqEncoderLayer) ... ok
test_cross_attention_static_kv_reuse (quantization.wrapq.wrappers.fairseq.test_quant_mha.TestQuantFairseqMHA) ... ok
test_forward_self_attention_shapes (quantization.wrapq.wrappers.fairseq.test_quant_mha.TestQuantFairseqMHA) ... ok
test_incremental_self_attention_accumulates (quantization.wrapq.wrappers.fairseq.test_quant_mha.TestQuantFairseqMHA) ... ok
test_lifecycle_propagation (quantization.wrapq.wrappers.fairseq.test_quant_mha.TestQuantFairseqMHA) ... ok
test_need_head_weights_shape (quantization.wrapq.wrappers.fairseq.test_quant_mha.TestQuantFairseqMHA) ... ok
test_observer_lookup_and_uniqueness (quantization.wrapq.wrappers.fairseq.test_quant_mha.TestQuantFairseqMHA) ... ok
test_return_new_kv_shapes (quantization.wrapq.wrappers.fairseq.test_quant_mha.TestQuantFairseqMHA) ... ok
test_static_causal_mask_upper_triangle_small (quantization.wrapq.wrappers.fairseq.test_quant_mha.TestQuantFairseqMHA) ... ok
test_cache_grows_across_multiple_single_token_steps (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_cache_mock_object_update (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_cache_tuple_concat_prefill_then_decode (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_forward_diff (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_forward_with_float_attention_mask (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_mask_slicing_with_cache_q_len_lt_k_len (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_mode_transitions (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_per_projection_override (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_use_cache_no_past_equivalence (quantization.wrapq.wrappers.llama.test_quant_attn.TestQuantLlamaAttention) ... ok
test_forward_diff (quantization.wrapq.wrappers.llama.test_quant_decoder_layer.TestQuantLlamaDecoderLayer) ... ok
test_layernorm_preserved (quantization.wrapq.wrappers.llama.test_quant_decoder_layer.TestQuantLlamaDecoderLayer) ... ok
test_mode_transitions (quantization.wrapq.wrappers.llama.test_quant_decoder_layer.TestQuantLlamaDecoderLayer) ... ok
test_mode_and_forward (quantization.wrapq.wrappers.llama.test_quant_mlp.TestQuantLlamaMLP) ... ok
test_calib_quant_export (quantization.wrapq.wrappers.llama.test_quant_mlp.TestSubgraphExport) ... ok
test_dtype_override (quantization.wrapq.wrappers.nn.test_quant_layernorm.TestQuantLayerNorm) ... ok
test_mode_transitions (quantization.wrapq.wrappers.nn.test_quant_layernorm.TestQuantLayerNorm) ... ok
test_no_affine_path (quantization.wrapq.wrappers.nn.test_quant_layernorm.TestQuantLayerNorm) ... ok
test_no_quant_matches_reference_various_shapes (quantization.wrapq.wrappers.nn.test_quant_layernorm.TestQuantLayerNorm)
In NO_QUANT mode, the wrapper must match PyTorch LayerNorm exactly, ... ok
test_quantised_output_close (quantization.wrapq.wrappers.nn.test_quant_layernorm.TestQuantLayerNorm)
After a standard calibrate->freeze cycle, quantized output should: ... ok
test_weight_stats_survive (quantization.wrapq.wrappers.nn.test_quant_layernorm.TestQuantLayerNorm)
Re-running a calibration cycle should not change fixed affine param stats. ... ok
test_dtype_override (quantization.wrapq.wrappers.nn.test_quant_linear.TestQuantLinear) ... ok
test_mode_transitions (quantization.wrapq.wrappers.nn.test_quant_linear.TestQuantLinear) ... ok
test_quantised_output_close (quantization.wrapq.wrappers.nn.test_quant_linear.TestQuantLinear) ... ok
test_weight_stats_survive (quantization.wrapq.wrappers.nn.test_quant_linear.TestQuantLinear) ... ok
test_dtype_override (quantization.wrapq.wrappers.nn.test_quant_silu.TestQuantSiLU) ... ok
test_mode_transitions (quantization.wrapq.wrappers.nn.test_quant_silu.TestQuantSiLU) ... ok
test_quantised_output (quantization.wrapq.wrappers.nn.test_quant_silu.TestQuantSiLU) ... ok
test_smoke_forward_quantized (quantization.wrapq.wrappers.test_ptq_wrapper.TestPTQSmoke) ... ok
test_default_mode_is_no_quant (quantization.wrapq.wrappers.test_ptq_wrapper.TestPTQWrapper) ... ok
test_mode_transitions (quantization.wrapq.wrappers.test_ptq_wrapper.TestPTQWrapper) ... ok
test_switch_activation_observer (quantization.wrapq.wrappers.test_ptq_wrapper.TestPTQWrapper) ... ok
test_weight_fake_quant_channelwise (quantization.wrapq.wrappers.test_ptq_wrapper.TestPTQWrapper) ... ok
test_parent_propagation_calls_each_observer_once (quantization.wrapq.wrappers.test_ptq_wrapper.TestPTQWrapperNoDoubleProcessing) ... ok
test_all_observers_is_empty_for_wrapper (quantization.wrapq.wrappers.test_ptq_wrapper.TestPTQWrapperObserverSurface) ... ok
test_named_and_get_observer_are_proxies (quantization.wrapq.wrappers.test_ptq_wrapper.TestPTQWrapperObserverSurface) ... ok
test_dtype_override (quantization.wrapq.wrappers.test_quant_elementwise.TestElementwiseWrappers) ... ok
test_missing_FUNC_raises (quantization.wrapq.wrappers.test_quant_elementwise.TestElementwiseWrappers) ... ok
test_mode_and_forward_diff (quantization.wrapq.wrappers.test_quant_elementwise.TestElementwiseWrappers) ... ok
test_registry_and_factory (quantization.wrapq.wrappers.test_quant_elementwise.TestElementwiseWrappers) ... ok
test_does_not_descend_into_quant_modules (quantization.wrapq.wrappers.test_quant_module_base.TestChildQuantModulesDiscovery) ... ok
test_lifecycle_propagates_through_containers (quantization.wrapq.wrappers.test_quant_module_base.TestChildQuantModulesDiscovery)
enable_calibration()/freeze_qparams() should propagate to all quant modules, ... ok
test_yields_immediate_descendants_across_containers (quantization.wrapq.wrappers.test_quant_module_base.TestChildQuantModulesDiscovery) ... ok
test_child_inherits_default_observer (quantization.wrapq.wrappers.test_quant_module_base.TestQuantConfigDefaultObserver) ... ok
test_global_default_observer (quantization.wrapq.wrappers.test_quant_module_base.TestQuantConfigDefaultObserver) ... ok
test_observer_override_precedence (quantization.wrapq.wrappers.test_quant_module_base.TestQuantConfigDefaultObserver) ... ok
test_extra_repr (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleBase) ... ok
test_fp_name_storage (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleBase) ... ok
test_fq_collect_and_quantize (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleBase) ... ok
test_get_observer (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleBase) ... ok
test_make_obs_override (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleBase) ... ok
test_mode_cycle (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleBase) ... ok
test_named_observers (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleBase) ... ok
test_config_default_qscheme (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleQScheme) ... ok
test_user_override_qscheme_wins (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleQScheme) ... ok
test_wrapper_default_qscheme_applied (quantization.wrapq.wrappers.test_quant_module_base.TestQuantModuleQScheme) ... ok
test_register_and_lookup (quantization.wrapq.wrappers.test_registry.TestRegistry) ... ok
test_try_register_graceful_skip (quantization.wrapq.wrappers.test_registry.TestRegistry) ... ok
test_try_register_success (quantization.wrapq.wrappers.test_registry.TestRegistry) ... ok
test_get_quantizer_returns_instance_not_class (quantization.test_quantizer_registry.QuantizerRegistryTest)
Factory must instantiate the quantizer with the provided config. ... ok
test_lazy_import_by_naming_convention_success (quantization.test_quantizer_registry.QuantizerRegistryTest)
get_quantizer should import: ... ok
test_lazy_import_raises_meaningful_error_on_failure (quantization.test_quantizer_registry.QuantizerRegistryTest)
If the naming-convention import fails outright, get_quantizer should raise ... ok
test_not_found_after_successful_import_raises (quantization.test_quantizer_registry.QuantizerRegistryTest)
Even if the module is importable, if it does NOT call register_quantizer, ... ok
test_register_and_lookup_exact_type (quantization.test_quantizer_registry.QuantizerRegistryTest)
Decorator registration should resolve by exact config type. ... ok
test_pass (unit_test.pass_test.test_cast_clamp_mixed_type_args.CastClampFloatInputIntMaxTest) ... ok
test_pass (unit_test.pass_test.test_cast_clamp_mixed_type_args.CastClampFloatInputIntMinTest) ... ok
test_pass (unit_test.pass_test.test_cast_clamp_mixed_type_args.CastClampIntInputFloatMaxTest) ... ok
test_pass (unit_test.pass_test.test_cast_clamp_mixed_type_args.CastClampIntInputFloatMinTest) ... ok
test_pass (unit_test.pass_test.test_convert_conv1d_to_conv2d.ConvertConv1dPaddingOneTest) ... ok
test_pass (unit_test.pass_test.test_convert_conv1d_to_conv2d.ConvertConv1dPaddingValidTest) ... ok
test_pass (unit_test.pass_test.test_convert_conv1d_to_conv2d.ConvertDepthwiseConv1dTest) ... ok
test_pass (unit_test.pass_test.test_convert_expand_to_slice_cat.ConvertKVCacheExpandTest) ... ok
test_pass (unit_test.pass_test.test_convert_expand_to_slice_cat.ConvertNonKVCacheExpandTest) ... ok
test_pass (unit_test.pass_test.test_decompose_fake_quantize.DecomposeFakeQuantizePerChannel) ... ok
test_pass (unit_test.pass_test.test_decompose_fake_quantize_tensor_qparam.DecomposeFakeQuantizeTensorQParamPerTensor) ... ok
test_pass (unit_test.pass_test.test_decompose_fake_quantize_tensor_qparam.DecomposeFakeQuantizeTensorQParamUint4Dtype) ... ok
test_decompose_int16 (unit_test.pass_test.test_decompose_fake_quantize_tensor_qparam.DecomposeFakeQuantizeTensorQParamsCacheMaskTest) ... ok
test_decompose_uint8 (unit_test.pass_test.test_decompose_fake_quantize_tensor_qparam.DecomposeFakeQuantizeTensorQParamsCacheMaskTest) ... ok
test_pass (unit_test.pass_test.test_decompose_grouped_conv2d.DecomposeGroupedConv2dTest) ... ok
test_pass (unit_test.pass_test.test_decompose_grouped_conv2d.DecomposeGroupedConv2dWithPaddingTest) ... ok
test_pass (unit_test.pass_test.test_fuse_leading_unsqueeze_reshape.LeadingUnsqueezeReshapeNet2kTest) ... ok
test_pass (unit_test.pass_test.test_fuse_leading_unsqueeze_reshape.LeadingUnsqueezeReshapeNetTest) ... ok
test_pass (unit_test.pass_test.test_fuse_redundant_reshape_to_mean.FuseRedundantReshapeToMeanTest) ... ok
test_pass (unit_test.pass_test.test_legalize_causal_mask_value.LegalizeCausalMaskValueTest) ... ok
test_pass (unit_test.pass_test.test_lower_pow2_to_mul.LowerPow2ToMulTest) ... ok
test_pass (unit_test.pass_test.test_lower_to_slice.TestLowerIndexSelectToSliceWithLongIndice) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node segm_index0_1 target segm_index0 segm_index0 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node segm_index1_1 target segm_index1 segm_index1 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node segm_index2_1 target segm_index2 segm_index2 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
ok
test_pass (unit_test.pass_test.test_lower_to_slice.TestLowerIndexSelectToSliceWithScalarIndex) ... ok
test_pass (unit_test.pass_test.test_lower_to_slice.TestLowerSelectCopyToSlice) ... ok
test_pass (unit_test.pass_test.test_merge_consecutive_cat.MergeConsecutiveCatTest) ... ok
test_pass_neg (unit_test.pass_test.test_merge_consecutive_cat.NotMergedConsecutiveCatTest1) ... ok
test_pass_neg (unit_test.pass_test.test_merge_consecutive_cat.NotMergedConsecutiveCatTest2) ... ok
test_pass (unit_test.pass_test.test_remove_nop.TestCloneContiguous) ... ok
test_pass (unit_test.pass_test.test_remove_nop.TestRemoveDetach) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_expand.RemoveRedundantExpandTest) ... ok
test_pass_with_minus_one (unit_test.pass_test.test_remove_redundant_expand.RemoveRedundantExpandTest) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_permute.RemoveRedundantPermuteFusedTest) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_permute.RemoveRedundantPermuteTest) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_reshape.RemoveRedundantReshapePattern1Test) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_reshape.RemoveRedundantReshapePattern2Test) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_reshape.RemoveRedundantReshapePattern3BroadcastedTest) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_reshape.RemoveRedundantReshapePattern3DifferentSoftmaxLengthTest) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_reshape.RemoveRedundantReshapePattern3Test) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_reshape.RemoveRedundantReshapePattern4Test) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_reshape.RemoveRedundantReshapePattern5Test) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_slice.RemoveRedundantSliceTest) ... ok
test_pass_neg (unit_test.pass_test.test_remove_redundant_to_copy.NonRedundantToCopyTest) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_to_copy.RemoveRedundantDeviceToCopyPassTest) ... ok
test_pass (unit_test.pass_test.test_remove_redundant_to_copy.RemoveRedundantDtypeToCopyPassTest) ... ok
test_pass (unit_test.pass_test.test_segment_index_select.TestSegmentIndexSelect) ... /home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/export/_unlift.py:75: UserWarning: Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule! Call GraphModule.add_submodule to add the necessary submodule, GraphModule.add_parameter to add the necessary Parameter, or nn.Module.register_buffer to add the necessary buffer
  getattr_node = gm.graph.get_attr(lifted_node)
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node segm_index0_1 target segm_index0 segm_index0 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node segm_index1_1 target segm_index1 segm_index1 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
/home/seongwoo/TICO/.venv/lib/python3.10/site-packages/torch/fx/graph.py:1801: UserWarning: Node segm_index2_1 target segm_index2 segm_index2 of  does not reference an nn.Module, nn.Parameter, or buffer, which is what 'get_attr' Nodes typically target
  warnings.warn(
ok
test_already_annotated (unit_test.quantization_test.test_adaptive_avg_pool2d.TestAnnotateAdaptiveAvgPool2d)
Test that already annotated nodes are not re-annotated ... ok
test_annotate_adaptive_avg_pool2d (unit_test.quantization_test.test_adaptive_avg_pool2d.TestAnnotateAdaptiveAvgPool2d)
Test basic annotation functionality ... ok
test_filter_fn_false (unit_test.quantization_test.test_adaptive_avg_pool2d.TestAnnotateAdaptiveAvgPool2d)
Test that filter_fn=False prevents annotation ... ok
test_filter_fn_true (unit_test.quantization_test.test_adaptive_avg_pool2d.TestAnnotateAdaptiveAvgPool2d)
Test that filter_fn=True allows annotation ... ok
test_input_annotated (unit_test.quantization_test.test_adaptive_avg_pool2d.TestAnnotateAdaptiveAvgPool2d)
Test behavior when input is already annotated ... ok
test_invalid_node_target (unit_test.quantization_test.test_adaptive_avg_pool2d.TestAnnotateAdaptiveAvgPool2d)
Test that nodes with wrong target are not annotated ... ok
test_none_quantization_config (unit_test.quantization_test.test_adaptive_avg_pool2d.TestAnnotateAdaptiveAvgPool2d)
Test behavior with None quantization config ... ok
test_compile_and_run_inference (unit_test.quantization_test.test_circle_executor.TestCircleExecutor) ... ok
test_init_raises_runtime_error_if_compiler_not_found (unit_test.quantization_test.test_circle_executor.TestCircleExecutor) ... ok
test_run_inference_before_compile_raises_error (unit_test.quantization_test.test_circle_executor.TestCircleExecutor) ... ok
test_run_inference_with_single_tensor_output (unit_test.quantization_test.test_circle_executor.TestCircleExecutor) ... ok
test_evaluate_simple_linear (unit_test.quantization_test.test_evaluate.EvaluateTest) ... ok
test_compute_all_metrics (unit_test.quantization_test.test_metric.TestMetricCalculator) ... ok
test_compute_selected_metrics (unit_test.quantization_test.test_metric.TestMetricCalculator) ... ok
test_custom_metric_and_duplicate_rejection (unit_test.quantization_test.test_metric.TestMetricCalculator) ... ok
test_unknown_metric_error (unit_test.quantization_test.test_metric.TestMetricCalculator) ... ok
test_max_abs_diff_basic (unit_test.quantization_test.test_metric.TestMetricKernels) ... ok
test_max_abs_diff_shape_mismatch (unit_test.quantization_test.test_metric.TestMetricKernels) ... ok
test_mse_basic (unit_test.quantization_test.test_metric.TestMetricKernels) ... ok
test_peir_basic (unit_test.quantization_test.test_metric.TestMetricKernels) ... ok
test_peir_shape_mismatch (unit_test.quantization_test.test_metric.TestMetricKernels) ... ok
test_peir_zero_interval (unit_test.quantization_test.test_metric.TestMetricKernels) ... ok
test_invalid_format_neg (unit_test.serialize_test.operator.test_utils.TestGetIntegerDtypeMin) ... ok
test_signed_types (unit_test.serialize_test.operator.test_utils.TestGetIntegerDtypeMin) ... ok
test_too_small_bitwidth_neg (unit_test.serialize_test.operator.test_utils.TestGetIntegerDtypeMin) ... ok
test_unsigned_types (unit_test.serialize_test.operator.test_utils.TestGetIntegerDtypeMin) ... ok
test_duplicate_names (unit_test.serialize_test.test_circle_graph.CircleGraphTest) ... ok
test_is_const (unit_test.serialize_test.test_circle_graph.CircleGraphTest) ... ok
test_validate_circle_shape (unit_test.serialize_test.test_circle_mapping.CircleSerializeTest) ... ok
test_pack_dtype_mismatch_neg (unit_test.serialize_test.test_pack.PackTest) ... ok
test_pack_uint4 (unit_test.serialize_test.test_pack.PackTest) ... ok
test_pack_uint4_odd (unit_test.serialize_test.test_pack.PackTest) ... ok
test_args (unit_test.utils_test.test_convert.ConvertFromExportedProgramTest) ... ok
test_args (unit_test.utils_test.test_convert.ConvertFromPt2Test) ... ok
test_args (unit_test.utils_test.test_convert.ConvertTest) ... ok
test_args_kwargs (unit_test.utils_test.test_convert.ConvertTest) ... ok
test_kwargs (unit_test.utils_test.test_convert.ConvertTest) ... ok
test_disable_when_decorator_with_false_predicate (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test disable_when decorator with False predicate ... ok
test_disable_when_decorator_with_true_predicate (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test disable_when decorator with True predicate ... ok
test_get_const_size_with_model_having_state_dict (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test get_const_size function with model having state_dict ... ok
test_get_const_size_with_scalar_tensors (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test get_const_size function handling scalar tensors ... ok
test_get_const_size_with_simple_model (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test get_const_size function with simple model ... ok
test_strdiff_type_assertion (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test strdiff function type assertions ... ok
test_strdiff_with_different_strings (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test strdiff function with different strings ... ok
test_strdiff_with_empty_strings (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test strdiff function with empty strings ... ok
test_strdiff_with_identical_strings (unit_test.utils_test.test_diff_graph.TestDiffGraph)
Test strdiff function with identical strings ... ok
test_all (unit_test.utils_test.test_enforce_type.EnforceTypeTest_Combined) ... ok
test_all_neg (unit_test.utils_test.test_enforce_type.EnforceTypeTest_Combined) ... ok
test_optional_list (unit_test.utils_test.test_enforce_type.EnforceTypeTest_ListUnion) ... ok
test_optional_list_neg (unit_test.utils_test.test_enforce_type.EnforceTypeTest_ListUnion) ... ok
test_optional_list (unit_test.utils_test.test_enforce_type.EnforceTypeTest_OptionalList) ... ok
test_optional_list_neg (unit_test.utils_test.test_enforce_type.EnforceTypeTest_OptionalList) ... ok
test_simple (unit_test.utils_test.test_enforce_type.EnforceTypeTest_SimpleDatatype) ... ok
test_simple_neg (unit_test.utils_test.test_enforce_type.EnforceTypeTest_SimpleDatatype) ... ok
test_simple (unit_test.utils_test.test_enforce_type.EnforceTypeTest_SimpleDict) ... ok
test_simple_neg (unit_test.utils_test.test_enforce_type.EnforceTypeTest_SimpleDict) ... ok
test_args0 (unit_test.utils_test.test_enforce_type.EnforceTypeTest_Trivial) ... ok
test_args1 (unit_test.utils_test.test_enforce_type.EnforceTypeTest_Trivial) ... ok
test_args1_neg (unit_test.utils_test.test_enforce_type.EnforceTypeTest_Trivial) ... ok
test_union (unit_test.utils_test.test_enforce_type.EnforceTypeTest_UnionOptionalList) ... ok
test_union_neg (unit_test.utils_test.test_enforce_type.EnforceTypeTest_UnionOptionalList) ... ok
test_avgpool_reverse_kwargs (unit_test.utils_test.test_infer.InferAvgPoolReverseKwargsTest) ... ok
test_concat (unit_test.utils_test.test_infer.InferCatTest) ... ok
test_concat_with_dim (unit_test.utils_test.test_infer.InferCatTest) ... ok
test_add_float (unit_test.utils_test.test_infer.InferSimpleAddTest) ... ok
test_add_float_builtin (unit_test.utils_test.test_infer.InferSimpleAddTest) ... ok
test_add_float_numpy (unit_test.utils_test.test_infer.InferSimpleAddTest) ... ok
test_export (unit_test.utils_test.test_mx.QuantizeMXTest) ... ok
test_ones (unit_test.utils_test.test_mx.QuantizeMXTest) ... ok
test_random_values (unit_test.utils_test.test_mx.QuantizeMXTest) ... ok
test_args (unit_test.utils_test.test_record_input.RecordInputTest) ... ok
test_args_kwargs (unit_test.utils_test.test_record_input.RecordInputTest) ... ok
test_condition (unit_test.utils_test.test_record_input.RecordInputTest) ... ok
test_input_to_remove (unit_test.utils_test.test_record_input.RecordInputTest) ... ok
test_kwargs (unit_test.utils_test.test_record_input.RecordInputTest) ... ok
test_circle_avgpool2d_with_custom_params (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleAvgPool2D with custom parameters ... ok
test_circle_avgpool2d_with_defaults (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleAvgPool2D with default parameters ... ok
test_circle_conv2d_invalid_groups (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleConv2d with invalid groups parameter ... ok
test_circle_conv2d_padding_with_defaults (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleConv2dPadding with default parameters ... ok
test_circle_conv2d_padding_with_string_padding (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleConv2dPadding with string padding ... ok
test_circle_conv2d_with_custom_params (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleConv2d with custom parameters ... ok
test_circle_conv2d_with_defaults (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleConv2d with default parameters ... ok
test_circle_depthwise_conv2d_basic (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleDepthwiseConv2d basic functionality ... ok
test_circle_depthwise_conv2d_invalid_groups_assertion (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleDepthwiseConv2d with invalid groups assertion ... ok
test_circle_depthwise_conv2d_padding_basic (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleDepthwiseConv2dPadding basic functionality ... ok
test_circle_depthwise_conv2d_with_custom_params (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleDepthwiseConv2d with custom parameters ... ok
test_circle_instance_norm_basic (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleInstanceNorm basic functionality ... ok
test_circle_instance_norm_with_custom_params (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleInstanceNorm with custom parameters ... ok
test_circle_maxpool2d_with_custom_params (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleMaxPool2D with custom parameters ... ok
test_circle_maxpool2d_with_defaults (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleMaxPool2D with default parameters ... ok
test_circle_quantize_mx_int8 (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleQuantizeMX with int8 format ... ok
test_circle_quantize_mx_mocked (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleQuantizeMX with mocked _quantize_mx function ... ok
test_circle_quantize_mx_unsupported_format (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleQuantizeMX with unsupported format ... ok
test_circle_quantize_mx_with_custom_params (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleQuantizeMX with custom parameters ... ok
test_circle_resize_nearest_neighbor_basic (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleResizeNearestNeighbor basic functionality ... ok
test_circle_resize_nearest_neighbor_unequal_scale_factors (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleResizeNearestNeighbor with unequal scale factors ... ok
test_circle_rms_norm_basic (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleRMSNorm basic functionality ... ok
test_circle_rms_norm_with_custom_eps (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleRMSNorm with custom epsilon ... ok
test_circle_transpose_conv_basic (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleTransposeConv basic functionality ... ok
test_circle_transpose_conv_invalid_groups (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleTransposeConv with invalid groups parameter ... ok
test_circle_transpose_conv_with_custom_params (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test CircleTransposeConv with custom parameters ... ok
test_custom_ops_with_conv2d_different_data_types (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test custom ops with conv2d different data types ... ok
test_custom_ops_with_different_tensor_shapes (unit_test.utils_test.test_register_custom_op.TestRegisterCustomOp)
Test custom ops with different tensor shapes ... ok
test_invalid_cmd_neg (unit_test.utils_test.test_run_bash_cmd.RunBashCmdTest) ... ok
test_simple_bash_cmd (unit_test.utils_test.test_run_bash_cmd.RunBashCmdTest) ... ok
test_validate_circle_shape (unit_test.utils_test.test_serialize.CircleSerializeTest) ... ok
test_validate_tensor_shape_neg (unit_test.utils_test.test_serialize.CircleSerializeTest) ... ok
test_bind_check_success (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_bind_input_num_mismatch (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_bind_missing_arg_fail (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_bind_multi_level_tuple (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_bind_multiple_values_fail (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_bind_shape_check_fail (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_bind_too_many_positional_fail (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_bind_tuple (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_bind_type_check_fail (unit_test.utils_test.test_signature.UtilsSignatureTest) ... ok
test_false_cases_neg (unit_test.utils_test.test_utils.BroadcastableTest) ... ok
test_true_cases (unit_test.utils_test.test_utils.BroadcastableTest) ... ok
test_supported_ranges (unit_test.utils_test.test_utils.TestGetQuantDtype) ... ok
test_unsupported_ranges_neg (unit_test.utils_test.test_utils.TestGetQuantDtype) ... ok

----------------------------------------------------------------------
Ran 793 tests in 148.484s

OK (skipped=6)
```
