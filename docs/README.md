# TICO Documentation

This directory contains the project-level documentation for TICO. The documents here
cover the public conversion API, the current implementation architecture, development
workflows, supported behavior, and the test strategy.

## Documentation map

| Document | Audience | Purpose |
|---|---|---|
| [Getting Started](./getting_started.md) | TICO users | Install TICO, convert PyTorch modules or `.pt2` files, configure conversion, run Circle models, and inspect artifacts. |
| [System Design](./design.md) | Contributors | Understand the actual `main`-branch conversion pipeline, package boundaries, pass execution, serialization, runtime behavior, and extension points. |
| [Development Guide](./development.md) | Contributors | Set up a source checkout, select a Torch build, run tests and model tests, format code, collect coverage, and understand PR CI. |
| [Requirements](./requirements.md) | Users and contributors | Define the currently supported contract, constraints, quality expectations, benchmark targets, and non-goals. |
| [System Test Guide](./system_test.md) | Contributors | Explain test layers, end-to-end validation, test commands, CI coverage, performance tests, and maintenance rules. |

## Subsystem documentation

The following subsystem documents are maintained next to their implementation:

- [Circle artifact tools](../tico/circle/README.md): inspect, verify, extract, and
  transform serialized Circle files.
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

## Source-of-truth policy

These documents describe released repository behavior; the implementation remains the
source of truth for version-sensitive details:

- Public conversion API: `tico/__init__.py` and `tico/utils/convert.py`
- Compile configuration: `tico/config/`
- Conversion pass order: `tico/utils/convert.py`
- Circle serialization: `tico/serialize/`
- Source-install and test tooling: `infra/command/` and `infra/scripts/`
- PR CI: `.github/workflows/check-pr.yaml`

When a change modifies a public API, command, configuration field, package boundary,
pass order, supported version, or test workflow, update the corresponding document in
the same pull request.

Do not store dated test counts or copied CI logs in these documents. GitHub Actions is
the source of truth for the latest run status; this documentation records the stable
workflow and acceptance criteria instead.
