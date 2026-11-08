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

### Documentation ownership

Maintain each detailed contract in its owning guide below. Entry guides may keep
a short explanation or runnable example, then link to that guide for options,
constraints, and maintenance rules. When consolidating overlapping text, preserve
any unique conditions or examples in the owner before replacing the other copy
with a link.

| Topic | Owning guide |
|---|---|
| Public conversion API and first-use examples | [Getting Started](./getting_started.md) |
| Source setup, test commands, formatting, and CI operations | [Development Guide](./development.md) |
| PyTorch support tiers, version selection policy, and qualification | [PyTorch Version Policy](./torch_version_policy.md) |
| Test layers, harness behavior, correctness criteria, and traceability | [System Test Guide](./system_test.md) |
| Conversion architecture, pass order, and graph invariants | [System Design](./design.md) |
| Circle artifact APIs, passes, and reference-runtime contracts | [Circle artifact tools](../tico/circle/README.md) |
| Large-file layout and ownership; explicit embedding export policy | [Large Circle export](./large_circle_export.md) and [CPU token embedding](./cpu_token_embedding.md) |
| Public quantization API and quantizer registration | [Quantization overview](../tico/quantization/README.md) |
| Algorithm methods and limits; wrapper lifecycle and state | Individual [algorithm guides](../tico/quantization/algorithm/README.md) and [WrapQ](../tico/quantization/wrapq/README.md) |
| Runnable CLI workflows; YAML fields and defaults | [CLI examples](../tico/quantization/examples/README.md) and [configuration reference](../tico/quantization/examples/configs/README.md) |
| Extension routing, shared recipe rules, and adapter/stage boundaries | [Recipes developer guide](../tico/quantization/recipes/README.md) and its linked subsystem guides |
| Trace workflows and APIs; wrapper cases and static profiles | [Debug guide](../tico/quantization/recipes/debug/README.md) and [wrapper smoke reference](../tico/quantization/recipes/debug/wrapper_smoke/README.md) |
| Quality protocols; MMMU parsing and diagnostics | [Recipe evaluation](../tico/quantization/recipes/evaluation/README.md) and [MMMU reference](../tico/quantization/recipes/evaluation/MMMU.md) |
| Export helpers and common artifact rules; Gemma4 artifact contracts | [Export guide](../tico/quantization/recipes/export/README.md) and [Gemma4 artifacts](../tico/quantization/recipes/export/gemma4_artifacts.md) |
| Application-specific commands, deployment profiles, and reports | The application's guide, such as the [hand-model example](../examples/hand_detector/README.md) |

### Implementation references

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
