# TICO Requirements and Supported Contract

This document records the behavior that the current TICO repository is designed and
tested to provide. It distinguishes the core conversion contract from optional
quantization workflows, local runtime support, Circle artifact tooling, and benchmark
targets.

The implementation and automated tests remain the source of truth. Version-sensitive
facts must be updated here when the corresponding code changes.

## Contents

- [1. Product scope](#1-product-scope)
- [2. Functional requirements](#2-functional-requirements)
- [3. Quality requirements](#3-quality-requirements)
- [4. Environment and compatibility](#4-environment-and-compatibility)
- [5. Performance benchmark targets](#5-performance-benchmark-targets)
- [6. Constraints and non-goals](#6-constraints-and-non-goals)
- [7. Requirement-to-test traceability](#7-requirement-to-test-traceability)
- [8. Maintenance rules](#8-maintenance-rules)

## 1. Product scope

TICO is a Python library and toolset for:

- exporting eligible PyTorch inference modules through `torch.export`
- legalizing and optimizing exported ATen graphs for Circle representation
- serializing supported graphs into Circle FlatBuffers
- preserving supported static and symbolic input contracts
- completing Circle-facing legalization for already quantized graphs
- quantizing models through a separate public quantization API and recipe layer
- executing supported Circle models locally for parity checks
- inspecting, verifying, extracting, and rewriting serialized Circle artifacts

TICO is intended for model conversion and deployment preparation. It is not a training
framework or a complete NPU compiler.

## 2. Functional requirements

### FR-1: Accept supported PyTorch program forms

TICO shall accept:

- an inference-mode `torch.nn.Module` with positional and optional keyword example
  inputs
- an in-memory `torch.export.ExportedProgram`
- a saved `.pt2` exported program

The module path shall forward `dynamic_shapes` and `strict` to
`torch.export.export()`.

### FR-2: Produce a Circle artifact

For a supported exported graph, TICO shall produce Circle bytes with the `CIR0` file
identifier and return them through `CircleModel` or write them through the
`pt2-to-circle` command.

The current core serializer shall create one Circle subgraph and register supported
user inputs and non-`None` user outputs.

### FR-3: Legalize supported exported graphs

The converter shall apply the current ordered decomposition and pass pipeline before
serialization. Rewrites shall preserve the observable program contract for their stated
matching conditions.

Passes that alter graph structure shall preserve valid graph-signature bindings and
provide correct tensor metadata for downstream passes and serialization.

### FR-4: Reject unsupported or training graphs clearly

Before Circle serialization, TICO shall reject remaining function targets without a
registered serializer visitor. Diagnostics should include the operator and source stack
trace when available.

TICO shall reject known training operators, and users shall convert modules in
`eval()` mode.

### FR-5: Support versioned compile configuration

TICO shall provide a versioned core compile configuration. Version `1.0` currently maps
to `CompileConfigV1`.

Fields that affect behavior shall have explicit defaults, implementation wiring, tests,
and documentation. A field must not be described as effective when the current
pipeline does not consume it.

### FR-6: Preserve supported symbolic input dimensions

For supported `torch.SymInt` dimensions, TICO shall serialize:

- `1` in the corresponding Circle `shape` entry
- `-1` in the corresponding `shapeSignature` entry

Static dimensions shall remain concrete. Input validation shall enforce rank, dtype,
and static dimensions while accepting dimensions marked dynamic.

### FR-7: Handle already quantized graphs

When the exported graph contains supported quantization operations, TICO shall run the
conditional quantization bundle needed to produce a legal Circle graph, including
qparam folding/propagation, quantized bias handling, safe constant propagation, unused
placeholder cleanup, and dtype-bridge insertion.

The core conversion API is not required to calibrate or quantize a floating-point model.
That responsibility belongs to `tico.quantization`.

### FR-8: Provide model quantization APIs

The quantization subsystem shall provide a lifecycle equivalent to:

```text
prepare(model, quant_config, example inputs)
    -> calibration/statistics collection
    -> convert(prepared_model)
```

Quantizer selection shall be configuration-driven through the quantizer registry.
Model-family-specific workflow behavior shall live in recipe adapters rather than in
generic algorithm stages where practical.

### FR-9: Provide local Circle execution for validation

`CircleModel` shall support:

- construction from Circle bytes
- saving and loading Circle files
- local execution of supported one-subgraph models

Input binding shall validate count, dtype, rank, and static shape dimensions. One output
shall be returned as a NumPy array; multiple outputs shall be returned as a list.

### FR-10: Provide Circle artifact tools

The `tico.circle` subsystem and `tico-circle` CLI shall support the documented
operations for:

- stable model inspection
- static internal-consistency verification
- operator-index and tensor-boundary extraction
- composable Circle-to-Circle cleanup and optimization passes

Artifact verification shall be described as structural verification, not numerical or
backend validation.

### FR-11: Provide reproducible developer commands

The repository shall expose source installation, build, configuration, test, formatting,
and coverage workflows through `./ccex`.

Documented commands shall match the actual options accepted by the corresponding
scripts under `infra/`.

## 3. Quality requirements

### QR-1: Numerical parity is test-specific and explicit

End-to-end conversion tests shall compare:

- number of outputs
- output shapes
- output dtypes
- output values

The default comparison uses `torch.testing.assert_close()` with explicit `rtol` and
`atol`. Numerically sensitive tests may define tighter or looser tolerances justified by
the operation and dtype.

TICO does not use one project-wide PEIR threshold as the universal correctness
criterion on the current branch.

### QR-2: Structural validity precedes runtime parity

Generated test artifacts shall be checked for Circle validity before runtime output
comparison. The current module harness invokes `circle2circle` for this purpose.

The `tico-circle verify` command provides an additional internal-consistency checker
for serialized artifacts, but it is not a substitute for runtime parity.

### QR-3: Pattern rewrites require positive and negative coverage

A graph rewrite shall be tested with:

- at least one matching graph
- a structurally similar non-matching graph
- relevant rank, shape, dtype, constant, and multi-user edge cases
- output equivalence when the rewrite changes computation
- graph-structure assertions for the property the pass promises

### QR-4: Tests should be deterministic and appropriately scoped

Unit and module tests should use fixed seeds or deterministic inputs where randomness
affects the assertion. They should use the smallest graph that preserves the behavior
under test.

A unit test should not require a remote checkpoint, network access, credentials, or a
GPU when a synthetic CPU graph can cover the contract.

### QR-5: Public behavior changes require documentation

A change to any of the following shall update documentation in the same pull request:

- top-level public APIs
- CLI options
- compile configuration fields or defaults
- source installation or test commands
- package responsibility boundaries
- pass order or major graph invariants
- supported Torch families or CI matrix
- Circle verification or extraction semantics

### QR-6: Errors must not be hidden by semantic fallback

Unsupported conversion, invalid graph structure, or failed verification shall not be
silently swallowed. A fallback is acceptable only when it preserves the documented
semantics and is explicitly tested.

Assertions, tolerances, or skips shall not be weakened merely to make a regression
pass.

## 4. Environment and compatibility

### Python

The package metadata requires Python 3.10 or newer.

### PyTorch

- The package warns when Torch is older than 2.5.
- It also recommends Torch 2.6 or newer for security when an older supported release
  is detected.
- Source installation tooling currently supports stable Torch families 2.5 through
  2.10 and a pinned nightly build.
- The default source-install family is 2.7.
- The current PR CI matrix tests Torch 2.5, 2.6, 2.7, 2.8, and the pinned nightly.

“Install-tool support” and “PR CI coverage” are intentionally separate claims. A
stable family accepted by `./ccex install` is not necessarily run in every pull request.

### Operating system

The supported development and CI workflow is Linux-first. Current PR jobs run on Ubuntu
24.04. Other platforms may work but are not guaranteed by the documented source
workflow.

### Circle runtime and ONE

Circle generation itself does not require `one-compiler`. Local execution through the
bundled Circle interpreter and the end-to-end test harness requires the corresponding
runtime components. The test harness can alternatively use the project-pinned `onert`
Python package.

### External models and data

Core unit and generated module tests shall not require model downloads. Opt-in model and
quantization recipe tests may have dedicated requirements and explicit external model
or dataset dependencies.

## 5. Performance benchmark targets

The repository contains an opt-in benchmark invoked with:

```bash
./ccex test -p
```

The current benchmark constructs one Llama decoder layer locally, measures conversion
three times, scales the mean by the configured number of hidden layers, and compares
serialized Circle size with the layer `state_dict` size.

Current benchmark targets are:

| Synthetic target | Scaled conversion-time threshold | Circle/state-dict size limit |
|---|---:|---:|
| Llama 3.2 1B configuration | 60 seconds | 1.01 |
| Llama 3.2 3B configuration | 180 seconds | 1.01 |

These values are repository benchmark thresholds, not hardware-independent public API
service-level guarantees. The benchmark is not part of the current PR CI matrix. Its
results depend on host hardware, PyTorch version, and the benchmark's single-layer
extrapolation method.

Do not copy one historical benchmark run into this document. The current command and
source are the authoritative way to obtain results.

## 6. Constraints and non-goals

### Current constraints

- The core serializer emits one subgraph.
- Built-in `CircleModel` execution supports one subgraph.
- Every serialized ATen overload must have a registered `NodeVisitor`.
- Tensor metadata must be available and representable in Circle.
- Only supported dense tensor layouts and dtypes can be serialized.
- Dynamic execution depends on runtime support; the test harness uses `onert` for
  dynamic-shape cases.
- The exported-graph pass order is an explicit list, not plugin-discovered.
- Unknown keys in version 1.0 YAML configuration are currently ignored.
- `CompileConfigV1.eliminate_rank_round_trip` is declared but not currently wired; the
  pass is enabled unconditionally in the conversion pipeline.

### Non-goals

TICO does not currently promise:

- training or autograd graph conversion
- universal ATen/operator coverage
- arbitrary Python control-flow support beyond what `torch.export` captures
- automatic NPU compiler invocation or compatibility certification
- automatic floating-point model calibration inside `tico.convert()`
- one universal numerical tolerance for every model and dtype
- full-model performance inference from the synthetic decoder-layer benchmark
- multi-subgraph generation by the core converter

## 7. Requirement-to-test traceability

| Requirement area | Primary validation |
|---|---|
| Public conversion APIs and `.pt2` path | `test/pt2_to_circle_test/test_pt2_to_circle.py`, generated module tests |
| Operator serialization and parity | `test/modules/op/`, `test/pt2_to_circle_test/test_op.py`, `test/unit_test/ops/`, `test/unit_test/serialize/` |
| Network-pattern conversion | `test/modules/net/`, `test/pt2_to_circle_test/test_net.py` |
| PyTorch-IR passes | `test/unit_test/passes/` plus E2E module tests where applicable |
| Dynamic shape signatures/execution | dynamic module tests and `onert` runtime path in the test harness |
| Core quantization graph passes | `test/quantization/passes/`, `test/unit_test/quantization/` |
| Quantization algorithms and WrapQ | `test/quantization/algorithm/`, `test/quantization/wrapq/` |
| Quantization recipes/export/evaluation | `test/quantization/recipes/`, `test/quantization/examples/`, related smoke tests |
| Circle artifact verification/extraction/passes | `test/unit_test/circle/` |
| CLI and configuration | `test/pt2_to_circle_test/`, config tests, quantization example/config tests |
| Performance thresholds | `test/performance/benchmark_perf.py`, invoked by `./ccex test -p` |
| Torch compatibility and package build | `.github/workflows/check-pr.yaml` matrix |
| Style and type checks | `.lintrunner.toml` through `./ccex format --no-apply-patches` |

See [System Test Guide](./system_test.md) for test flow and commands.

## 8. Maintenance rules

When behavior changes:

1. Update implementation and focused tests together.
2. Update the relevant user or contributor document.
3. Keep command examples executable from the repository root.
4. Separate current guarantees from experimental, internal, or planned behavior.
5. Do not claim NPU compatibility from Circle serialization alone.
6. Do not retain dated test counts or logs in the documentation.
7. Link to subsystem documentation instead of duplicating long contracts.

Version-sensitive sources of truth:

| Fact | Source |
|---|---|
| Python requirement and entry points | `pyproject.toml` |
| Torch warning and public exports | `tico/__init__.py` |
| Supported source-install Torch families | `infra/scripts/pytorch_package_utils.sh` |
| PR CI matrix | `.github/workflows/check-pr.yaml` |
| Compile configuration | `tico/config/v1.py` and its use sites |
| Pass order | `tico/utils/convert.py` |
| Performance thresholds | `test/performance/benchmark_perf.py` |
