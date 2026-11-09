# TICO System Test Guide

This document describes how the current TICO repository validates conversion,
serialization, runtime parity, quantization, Circle artifact transformations, package
compatibility, and benchmark targets. It owns test layers, harness behavior,
correctness criteria, and validation traceability. Use the
[Development Guide](development.md) for environment setup, commands, and CI details.

It intentionally does not contain a copied test log or a fixed test count. The latest
GitHub Actions run is the source of truth for current results.

## Contents

- [1. Test objectives](#1-test-objectives)
- [2. Test layers](#2-test-layers)
- [3. Test directory layout](#3-test-directory-layout)
- [4. End-to-end module test flow](#4-end-to-end-module-test-flow)
- [5. Correctness criteria](#5-correctness-criteria)
- [6. Running tests](#6-running-tests)
- [7. Runtime selection and dynamic shapes](#7-runtime-selection-and-dynamic-shapes)
- [8. Quantization testing](#8-quantization-testing)
- [9. Performance testing](#9-performance-testing)
- [10. Circle artifact testing](#10-circle-artifact-testing)
- [11. Continuous integration](#11-continuous-integration)
- [12. Test-result reporting policy](#12-test-result-reporting-policy)
- [13. Adding or changing tests](#13-adding-or-changing-tests)
- [14. Traceability](#14-traceability)

## 1. Test objectives

The test suite is designed to answer distinct questions:

1. **Local implementation correctness**
   - Does one pass, serializer visitor, utility, quantizer, or Circle operation satisfy
     its focused contract?
2. **Conversion correctness**
   - Can a representative PyTorch program be exported, legalized, serialized, and
     structurally validated?
3. **Runtime parity**
   - Does the Circle model produce outputs with the expected count, shape, dtype, and
     values?
4. **Negative-path behavior**
   - Does TICO reject unsupported or invalid patterns with the expected diagnostic?
5. **Version compatibility**
   - Does the package satisfy the tiered PR and scheduled PyTorch compatibility policy?
6. **Performance and scheduler regression detection**
   - Do the opt-in synthetic Llama conversion and size benchmarks remain within their
     repository thresholds?
   - Does the full-model Circle O1 round scheduler reduce pass executions while
     producing the same serialized result as the legacy restart scheduler?
7. **Artifact-tool correctness**
   - Do Circle verification, extraction, semantic optimization, cleanup, and index
     remapping preserve their documented structural contracts?

No single test layer proves every property. In particular, static Circle verification,
runtime parity, and target-NPU compilation are different checks.

## 2. Test layers

| Layer | Primary location | Purpose |
|---|---|---|
| Focused unit tests | `test/unit_test/` | Validate core passes, serializer helpers, operator visitors, Circle tools, quantization helpers, and utilities in isolation. |
| Generated operator/network tests | `test/modules/op/`, `test/modules/net/` | Define small PyTorch modules that are collected by the end-to-end conversion harness. |
| Conversion harness | `test/pt2_to_circle_test/` | Exercise direct and `.pt2` conversion, Circle validation, runtime execution, and PyTorch/Circle comparison. |
| Opt-in model tests | `test/modules/model/` | Validate real model-family integration with per-model dependencies. |
| Quantization tests | `test/quantization/` | Cover algorithms, WrapQ, qparam passes, configs, recipes, evaluation, analysis, and export behavior. |
| Performance tests | `test/performance/` | Measure synthetic Llama conversion/size thresholds and compare full Circle O1 scheduling strategies. |
| Shared test support | `test/support/` | Provide test builders, runtime adapters, tags, and helpers. |
| PyTorch compatibility checks | `.github/workflows/check-pr.yaml`, `.github/workflows/check-pytorch-compatibility.yaml` | Enforce style/DCO checks and validate the default, oldest supported, candidate, and `nightly-latest` tiers. |

## 3. Test directory layout

```text
test/
├── README.md
├── requirements.txt
├── requirements_pre.txt
├── dump_exported_program.py
├── dump_pt2_model.py
├── modules/
│   ├── base.py
│   ├── op/
│   ├── net/
│   └── model/
├── pt2_to_circle_test/
│   ├── builder.py
│   ├── test_op.py
│   ├── test_net.py
│   ├── test_model.py
│   └── test_pt2_to_circle.py
├── unit_test/
│   ├── circle/
│   ├── ops/
│   ├── passes/
│   ├── quantization/
│   ├── serialize/
│   └── utils/
├── quantization/
│   ├── algorithm/
│   ├── analysis/
│   ├── config/
│   ├── evaluation/
│   ├── examples/
│   ├── passes/
│   ├── recipes/
│   └── wrapq/
├── performance/
└── support/
```

Large model tests are selected through `./ccex test -m ...`. The normal discovery path
uses the `pt2_to_circle_test` package loader to collect the operator and network module
suites without automatically loading every model directory.

Use the [test documentation index](../test/README.md) for guides to test modules,
the conversion harness, Circle helpers, and performance benchmarks.

## 4. End-to-end module test flow

`NNModuleTest` in `test/pt2_to_circle_test/builder.py` provides the common conversion
flow.

### 4.1 Discover a test module

A module derives from the repository test base and provides at least
`get_example_inputs()`. It may also provide:

- `get_dynamic_shapes()`
- `get_compile_config()`
- `get_golden_outputs()`
- per-test `rtol` and `atol`
- tags that select direct conversion, skip inference, provide a golden output, mark a
  negative test, or skip the case

Generated test names include the source module namespace, which allows `unittest -k`
filtering through `./ccex test -k ...`.

### 4.2 Run the PyTorch reference first

The harness runs the PyTorch module before export under `torch.no_grad()` and `eval()`.
This ordering is deliberate because some modules may mutate state during export. The
PyTorch output tree is flattened to the tensor/scalar sequence represented by Circle.

### 4.3 Export and convert

The normal path is:

```text
nn.Module
  -> torch.export.export
  -> torch.export.save(.pt2)
  -> torch.export.load(.pt2)
  -> TICO conversion
  -> .circle
```

A test tagged for direct conversion uses:

```text
nn.Module
  -> torch.export.export
  -> TICO conversion
  -> .circle
```

Both paths call the same core `ExportedProgram` conversion implementation.

### 4.4 Validate the Circle artifact

The harness validates the serialized file with TICO's own artifact layer:

1. `CircleDocument.verify()` checks indices, dataflow, constant buffers, interfaces,
   and signatures.
2. Preparing a `CircleReferenceRuntime` checks that every operator is a supported
   builtin with valid operand references and decodable constants.
3. For models with static inputs, `probe_static_contracts()` executes the model on
   zero-valued inputs and validates the shape and dtype computed by every operator
   against the serialized tensor contracts. Dynamic models receive the same
   per-operator contract validation during the real inference step.

This replaces the former external `circle2circle` invocation and catches malformed or
unsupported Circle structures before numerical comparison.

### 4.5 Validate symbolic shape metadata

When a test declares dynamic shapes, the harness reads `ModelInputSpec` from the Circle
file and requires at least one `-1` entry in an input shape signature. Dynamic models
execute with the reference runtime (or `onert`); the legacy `circle-interpreter`
runtime rejects them.

### 4.6 Execute the Circle model

Unless inference is disabled by a test tag, the harness selects:

- `reference` (built-in `tico.circle.runtime`) by default
- `circle-interpreter` or `onert` when selected with `CCEX_RUNTIME` / `-r`; a test
  tagged `use_onert` falls back to `onert` under `circle-interpreter`

The runtime helper binds inputs through the serialized model input specification.
For dynamic `onert` inputs, the adapter replaces unspecified runtime tensor dimensions
with the concrete input shapes before inference.

### 4.7 Compare outputs

The harness compares either:

- the pre-export PyTorch reference outputs, or
- explicit golden outputs supplied by the test module

`None` outputs are removed because Circle exposes only serialized outputs.

### 4.8 Negative tests

A negative test executes the direct conversion path under `assertRaises` and verifies
that the expected diagnostic text appears in the exception. Negative tests should target
a specific unsupported or invalid contract, not merely accept any unrelated failure.

## 5. Correctness criteria

### Output contract

The default result validator checks:

1. Number of outputs
2. Shape and dtype of every Tensor output
3. Exact equality for integer and bool values
4. Floating-point values through `torch.testing.assert_close()`

Default tolerances are currently:

```text
rtol = 1e-5
atol = 1e-5
```

A module can override floating-point tolerances; integer and bool values remain exact.
The override must be justified by the operation, dtype, and expected numerical behavior.
Do not globally relax tolerance because one regression fails.

Python scalars accept scalar or single-element Circle outputs of the same kind: bool,
signed/unsigned integer, or floating point. Integer values are compared without casting
the reference to the output dtype or passing through floating point. Python floats use
the floating-point tolerance, allowing normal float32 rounding without hiding overflow.
Matching NaNs and same-sign infinities are accepted; nonmatching nonfinite values are
rejected.

### Graph rewrite contract

Focused pass tests should assert both:

- semantic equivalence where execution is practical
- the promised graph structure, such as operator removal, replacement, rank, or
  metadata state

Pattern-based passes need a non-matching case that is close enough to catch overly
broad matching.

### Structural contract

Circle-facing tests should validate relevant properties such as:

- tensor/operator indices
- graph inputs and outputs
- buffer ownership and reuse
- shape and shape signature consistency
- qparam dtype, axis, scale, and zero-point shape
- signature and subgraph references
- cleanup and compaction remapping

### Determinism

Use fixed seeds or deterministic data whenever random values influence the assertion.
Keep synthetic tensor sizes small unless the behavior depends on a production-scale
shape.

## 6. Running tests

Choose coverage according to the [change-specific checklist](#13-adding-or-changing-tests).
The Development Guide maintains executable commands and their options; the links below
connect each validation task to that reference.

### Set up the environment

Follow [Create a development environment](development.md#create-a-development-environment)
for installation order and test dependencies, and
[Torch selection](development.md#torch-selection) for version choices.

### Default test suite

Use the [default suite](development.md#default-suite) for model-independent discovery.

### Run a keyword-filtered subset

Use [keyword filtering](development.md#keyword-filtering) for a focused subsystem or
generated operator/network test namespace.

### Include internal-only tests

See [Other test-runner options](development.md#other-test-runner-options) for explicit
internal-test selection.

### Enable verbose TICO diagnostics

See [Log levels](development.md#log-levels) for debug logging and
[Intermediate graph images](development.md#intermediate-graph-images) for pass snapshots.

### Select a runtime

See [Runtime selection](development.md#runtime-selection) for the default reference
runtime and explicit external-runtime installation and selection.

### Run a model test

See [Model tests](development.md#model-tests) for per-model dependencies and quoted
selection patterns.

### Run performance tests

Use the commands in [Performance benchmarks](development.md#performance-benchmarks).
The [performance criteria below](#9-performance-testing) explain what each benchmark
measures and which results are thresholds.

### Option-combination constraints

See [Other test-runner options](development.md#other-test-runner-options) for mutually
exclusive selectors and the separate performance entry point.

## 7. Runtime selection and dynamic shapes

### Reference runtime

The default runtime loads the Circle file through `CircleModel`, binds inputs with
`ModelInputSpec`, and executes the serialized bytes with `tico.circle.runtime`. It
validates every operator result against the serialized shape/dtype contract, supports
dynamic shape signatures, and requires no external package. Its operator coverage,
quantized execution modes, and debugging options are documented in
[Circle artifact tools](../tico/circle/README.md#reference-runtime).

A guard test (`test/unit_test/utils/test_runtime_independence.py`) fails if the
default conversion, execution, or quantized evaluation path imports `onert` or `cffi`,
spawns a subprocess, or loads a native library. CI additionally asserts that the test
environment contains neither ONE nor `onert`.

### Optional external runtimes

`circle-interpreter` (ONE luci-interpreter via CFFI) and `onert` remain available as
explicitly selected compatibility runtimes. They are not installed by
`./ccex configure test`; the Development Guide owns their
[installation and selection commands](development.md#runtime-selection). Comparing
against them is additional validation, not a replacement for the default suite.

### Dynamic input rules

A dynamic-shape test shall:

1. Supply `get_dynamic_shapes()` compatible with `torch.export`.
2. Verify that the generated Circle input contains a `-1` shape-signature dimension.
3. Execute with concrete tensor shapes allowed by the export constraints.
4. Validate outputs with the same count/shape/dtype/value rules as static tests.

After the example-input comparison, the harness re-runs every dynamic model with
inputs whose dynamic dimensions differ from the example sizes and compares the results
with PyTorch again. This proves that the serialized graph is shape-generic (for example,
`repeat`/`expand` sizes derived from a dynamic dimension are read from the runtime
`SHAPE` instead of being frozen at export time). A successful dynamic export does not by
itself prove that every external Circle runtime supports the resulting shape signature.

## 8. Quantization testing

Quantization tests are organized by responsibility rather than forcing all behavior
through one full model:

| Area | Typical location | Expected focus |
|---|---|---|
| Core graph qparam passes | `test/quantization/passes/`, `test/unit_test/quantization/` | Folding, propagation, bias quantization, constant propagation, dtype bridges, placeholder cleanup. |
| Quantizer registry and public lifecycle | `test/quantization/test_quantizer_registry.py`, config tests | Correct config dispatch, one-time `prepare`, required `convert` ordering, inplace behavior. |
| WrapQ | `test/quantization/wrapq/` | Wrappers, observers, fake quantization, module state, export adapters, model-family attention/MLP behavior. |
| Algorithms | `test/quantization/algorithm/` | GPTQ and other algorithm-specific statistics and transformations. |
| Recipes | `test/quantization/recipes/` | Adapter/stage boundaries, config loading, calibration routing, checkpoint and Circle export. |
| Examples/configs | `test/quantization/examples/` | CLI/config behavior without requiring uncontrolled downloads at import time. |
| Analysis/evaluation | `test/quantization/analysis/`, `evaluation/` | Numerical metrics, clipping/sensitivity utilities, and benchmark helpers. |

Quantization tests should make qparam semantics explicit:

- dtype and representable range
- symmetric/asymmetric mapping
- per-tensor/per-channel granularity
- channel axis
- scale and zero-point shapes
- observer/fake-quant enabled state
- behavior for degenerate ranges and empty/incomplete calibration

Use a small synthetic test for the local contract, then add the smallest model-family
smoke test needed to prove integration.

## 9. Performance testing

### Conversion speed and serialized size

`test/performance/benchmark_perf.py` benchmarks synthetic Llama 3.2 decoder layers for
1B and 3B configurations.

The benchmark:

1. Instantiates a local `LlamaDecoderLayer` with sequence length 256.
2. Measures `tico.convert()` three times for one layer.
3. Multiplies the mean by the configured number of hidden layers.
4. Converts once more and compares Circle byte size with a serialized layer
   `state_dict`.

Current thresholds:

| Configuration | Scaled time | Size ratio |
|---|---:|---:|
| Llama 3.2 1B | 60 seconds | Circle <= 1.01 x state dict |
| Llama 3.2 3B | 180 seconds | Circle <= 1.01 x state dict |

Interpret these results as regression indicators for this benchmark implementation, not
as measured full-model end-to-end deployment latency. Results are host- and
version-dependent.

### Full Circle O1 scheduling

`test/performance/benchmark_circle_optimizer.py` accepts a caller-provided full Circle
artifact and runs the same O1 pass sequence with two schedulers:

- legacy `CirclePassStrategy.RESTART`
- O1's round-based `CirclePassStrategy.UNTIL_NO_CHANGE`

Each repetition starts from a fresh clone, verifies the result, and requires the two
scheduler variants to produce byte-identical Circle binaries. The report includes
elapsed time, pass-execution counts, invocation reduction, output size, and SHA-256.
Heavy constant folding and optional O1 transforms can be enabled through explicit
command-line flags.

This comparison has no repository threshold and stores no model artifact. It is a
diagnostic for real graph size and pass-interaction cost, not a claim that the two
schedulers have identical intermediate states.

Both performance workflows are opt-in and are not in the current PR test matrix.

## 10. Circle artifact testing

Tests under `test/unit_test/circle/` validate the post-serialization `tico.circle`
layer. Assert structural validity, preservation of interfaces and observable behavior,
and safe mutation of shared or indexed resources. Rewrite tests also need convergence,
rollback, and scheduler-equivalence coverage when those mechanisms are affected.
The [Circle testing checklist](../tico/circle/README.md#testing) maintains the detailed
artifact-specific cases, including verification, extraction, pass registration, and
reference-runtime oracles.

Circle artifact tests should use in-memory synthetic Circle documents or the smallest
possible fixture. Do not use OCR, graph screenshots, or external visualization as the
source of structural assertions.

## 11. Continuous integration

CI applies the same validation layers to different risk and release boundaries. The
[Development Guide](development.md#pull-request-ci) owns job setup, commands, and the
current matrix; the [PyTorch Version Policy](torch_version_policy.md#ci-tiers) owns
qualification tiers and release gates.

### Pull-request workflow

The pull-request workflow combines contribution checks, package validation, and
version compatibility checks before changes merge.

#### Commit-message check

The contribution gate checks the required DCO trailer. Its exact syntax and activation
rule are in [Pull-request CI](development.md#pull-request-ci).

#### Formatting and static analysis

Formatting and static analysis validate source consistency separately from numerical
tests. See [Formatting and static checks](development.md#formatting-and-static-checks)
for local reproduction and [Pull-request CI](development.md#pull-request-ci) for job
setup and the combined required status.

#### Build package once

Versioned checks exercise the same built package. See
[Build and install a wheel](development.md#build-and-install-a-wheel) for local package
validation and [Pull-request CI](development.md#pull-request-ci) for artifact reuse.

#### Versioned tests

The complete default-family suite checks broad regressions; focused export and
quantization smoke tests detect version-contract breakage on other selected families.
The [CI matrix and smoke scope](development.md#pull-request-ci) distinguish blocking
support checks from candidate signals.

### Scheduled compatibility workflow

Broader compatibility runs provide early warning across supported, candidate, and
nightly families. See the [CI schedule](development.md#pull-request-ci) for cadence,
manual dispatch, and its relationship to PR branch protection.

### Official release workflow

Publication is gated by complete-suite validation across the qualified stable support
window. See the [release CI details](development.md#pull-request-ci) and
[release-branch policy](torch_version_policy.md#release-branches).

## 12. Test-result reporting policy

Do not maintain a “latest test results” table with a date, pass count, or copied console
log in this document. It becomes incorrect as soon as tests are added or CI changes.

Use these sources instead:

- GitHub Actions for the latest PR/main result
- the exact command output for a local reproduction
- attached benchmark artifacts when performance evidence is needed
- a pull-request description for change-specific validation

A change report should list commands actually run, for example:

```text
./ccex test -k eliminate_rank_round_trip
./ccex test -k op
./ccex format --no-apply-patches
```

Do not say an unexecuted command passed. Explain environment limitations or skipped
validation explicitly.

## 13. Adding or changing tests

### For a PyTorch-IR pass

- Add focused coverage under `test/unit_test/passes/test_<pass>.py`.
- Cover a valid match and a close non-match.
- Assert relevant graph structure and metadata.
- Add a generated module parity test when serialization/runtime behavior changes.

### For an operator visitor or serializer change

- Add unit coverage under `test/unit_test/ops/` or `serialize/`.
- Add a small module under `test/modules/op/` or `net/`.
- Validate conversion through the normal `.pt2` path unless the feature specifically
  concerns direct conversion.
- Check shape, dtype, and values, not only successful serialization.

### For a public API or CLI change

- Add an API/CLI test in `test/pt2_to_circle_test/` or the owning subsystem.
- Test invalid input and error messages.
- Update `docs/getting_started.md` and relevant command help/reference text.

### For quantization

- Test lifecycle stages separately where possible.
- Validate save/load or export behavior when state representation changes.
- Avoid remote downloads in unit discovery.
- Add model-family smoke coverage only after local contracts are covered.

### For Circle artifact tools

- Test the pre-pass and post-pass document.
- Run verifier assertions on expected errors/warnings.
- Cover a close non-matching graph for pattern-based rewrites.
- Validate rollback when a rewrite fails after beginning a mutation.
- Validate global resource remapping when retaining multiple subgraphs.
- Add pipeline-level idempotence or scheduler-equivalence coverage when pass order
  or fixed-point behavior changes.

### For a bug fix

First add a regression test that fails for the reported reason. Keep the fixture minimal
and make the assertion distinguish the bug from unrelated failures.

## 14. Traceability

| Requirement | Main test evidence |
|---|---|
| Supported module/ExportedProgram/`.pt2` conversion | `test/pt2_to_circle_test/`, `test/modules/` |
| Unsupported/training error behavior | Negative module tests and focused conversion tests |
| Pass semantics and scheduler behavior | `test/unit_test/passes/` |
| Operator serialization | `test/unit_test/ops/`, `serialize/`, generated operator tests |
| Static and dynamic input contracts | `ModelInputSpec` tests, dynamic module tests, `test/unit_test/circle/runtime/` |
| Reference runtime kernels and execution contracts | `test/unit_test/circle/runtime/`, `test/unit_test/quantization/test_circle_executor.py` |
| Independence from ONE/onert | `test/unit_test/utils/test_runtime_independence.py`, CI environment assertion |
| Quantized graph legalization | `test/quantization/passes/`, quantization unit tests |
| Quantization API and workflows | registry/config/WrapQ/algorithm/recipe tests |
| Circle artifact structural contracts, pass scheduling, and rewrite transactions | `test/unit_test/circle/` |
| Package/Torch compatibility | `tico/utils/compat/torch_version_policy.py`, `.github/workflows/check-pr.yaml`, `.github/workflows/check-pytorch-compatibility.yaml` |
| Performance thresholds | `test/performance/benchmark_perf.py` through `./ccex test -p` |
| Circle O1 scheduler comparison | `test/performance/benchmark_circle_optimizer.py` with a caller-provided artifact |
| Formatting/type quality | `.lintrunner.toml` through `./ccex format --no-apply-patches` |

See [Requirements](./requirements.md) for the supported contract and
[Development Guide](./development.md) for environment setup.
