# TICO Documentation

Start here to find TICO's project guides and the subsystem references maintained
next to their implementation. For a first conversion, read
[Getting Started](./getting_started.md); for a source checkout, use the
[Development Guide](./development.md). Return to the [project overview](../README.md)
for installation highlights and a minimal example.

## Documentation map

| Document | Audience | Purpose |
|---|---|---|
| [Getting Started](./getting_started.md) | TICO users | Install TICO, convert PyTorch modules or `.pt2` files, configure conversion, run Circle models, and inspect artifacts. |
| [Large Circle export](./large_circle_export.md) | Users and contributors | Choose file or bytes APIs, understand appended buffers and memory ownership, and validate large artifacts. |
| [CPU token embedding](./cpu_token_embedding.md) | Quantization and runtime developers | Select explicit embedding quantization boundaries and understand CPU lookup execution. |
| [System Design](./design.md) | Contributors | Understand the actual `main`-branch conversion pipeline, package boundaries, pass execution, serialization, runtime behavior, and extension points. |
| [Development Guide](./development.md) | Contributors | Set up a source checkout, select a Torch build, run tests and model tests, format code, collect coverage, and understand PR CI. |
| [PyTorch Version Policy](./torch_version_policy.md) | Users and contributors | Understand package installation, legacy installable families, the qualified stable window, candidates, CI tiers, and promotion rules. |
| [Requirements](./requirements.md) | Users and contributors | Define the currently supported contract, constraints, quality expectations, benchmark targets, and non-goals. |
| [System Test Guide](./system_test.md) | Contributors | Explain test layers, end-to-end validation, test commands, CI coverage, performance tests, and maintenance rules. |

## Subsystem documentation

The following subsystem documents are maintained next to their implementation:

- [Circle artifact tools](../tico/circle/README.md): inspect, verify, extract,
  transform, and execute serialized Circle files with the reference runtime.
- [Quantization](../tico/quantization/README.md): public quantization APIs and package
  layout.
- [Quantization algorithms](../tico/quantization/algorithm/README.md): algorithm-specific
  implementations.
- [WrapQ](../tico/quantization/wrapq/README.md): wrapper, observer, fake-quant, and PTQ
  infrastructure.
- [Quantization recipes](../tico/quantization/recipes/README.md): model adapters,
  algorithm stages, calibration, evaluation, export, and debug workflows.
- [Quantization examples](../tico/quantization/examples/README.md): config-driven CLI
  entry points and presets.

## Examples and test guides

| Task | Guide |
|---|---|
| Convert, quantize, export, and inspect the MediaPipe hand models | [Hand-model example](../examples/hand_detector/README.md) |
| Find test-module, conversion-harness, Circle-helper, and performance documentation | [Test documentation](../test/README.md) |
| Run model and non-model tests, or inspect the validation strategy | [Development commands](./development.md#run-tests) and [System Test Guide](./system_test.md) |
| Locate logo assets and usage notes | [TICO logo](../assets/logo/README.md) |

## Historical design documents

The project-initiation documents preserve the original plans. Use the current
guides for implementation contracts and development commands.

| Historical document | Current guide |
|---|---|
| [Initial system design](./archive/project-initiation/design.md) | [System Design](./design.md) |
| [Initial requirements](./archive/project-initiation/requirements.md) | [Requirements](./requirements.md) |
| [Initial test plan](./archive/project-initiation/system_test.md) | [System Test Guide](./system_test.md) |

## Source-of-truth policy

The current guides describe repository behavior for the selected branch or tag;
the implementation remains the source of truth for version-sensitive details:

- Public conversion API: [`tico/__init__.py`](../tico/__init__.py) and
  [`tico/utils/convert.py`](../tico/utils/convert.py)
- Compile configuration: [`tico/config/`](../tico/config/)
- Conversion pass order: [`tico/utils/convert.py`](../tico/utils/convert.py)
- Circle serialization: [`tico/serialize/`](../tico/serialize/)
- Source-install and test tooling: [`infra/command/`](../infra/command/) and
  [`infra/scripts/`](../infra/scripts/)
- PyTorch compatibility metadata:
  [`tico/utils/compat/torch_version_policy.py`](../tico/utils/compat/torch_version_policy.py)
- PR CI: [`check-pr.yaml`](../.github/workflows/check-pr.yaml)
- Scheduled PyTorch compatibility CI:
  [`check-pytorch-compatibility.yaml`](../.github/workflows/check-pytorch-compatibility.yaml)

When a change modifies a public API, command, configuration field, package boundary,
pass order, supported version, or test workflow, update the corresponding document in
the same pull request.

Do not store dated test counts or copied CI logs in these documents. GitHub Actions is
the source of truth for the latest run status; this documentation records the stable
workflow and acceptance criteria instead.
