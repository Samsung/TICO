# Development Guide

This guide describes the current source-development workflow for TICO. Repository
commands are routed through the `./ccex` helper in the project root.

## Contents

- [Prerequisites](#prerequisites)
- [Create a development environment](#create-a-development-environment)
- [Torch selection](#torch-selection)
- [Build and install a wheel](#build-and-install-a-wheel)
- [Run tests](#run-tests)
- [Test organization](#test-organization)
- [Runtime selection](#runtime-selection)
- [Model tests](#model-tests)
- [Debug conversion](#debug-conversion)
- [Formatting and static checks](#formatting-and-static-checks)
- [Coverage](#coverage)
- [Pull-request CI](#pull-request-ci)
- [Change-specific guidance](#change-specific-guidance)

## Prerequisites

- Linux development environment
- Python 3.10 or newer
- `git`
- A Python virtual environment is strongly recommended
- A compatible ONE installation when running `circle-interpreter`-based end-to-end
  tests locally

TICO conversion does not require ONE. The default end-to-end test runtime does because
it executes the generated Circle model.

## Create a development environment

```bash
git clone https://github.com/Samsung/TICO.git
cd TICO

python3 -m venv .venv
source .venv/bin/activate

# Install a supported Torch build and TICO in editable mode.
./ccex install

# Install formatter/static-check dependencies and test-only dependencies.
./ccex configure
```

The order matters. `./ccex configure test` validates an already installed Torch package
and installs the matching TorchVision build; it does not install TICO or Torch. Run
`./ccex install` first.

Set up only one part of the environment when needed:

```bash
./ccex configure format
./ccex configure test
```

## Torch selection

The source installer accepts a stable family, an exact version, a qualification
candidate, a repository-pinned nightly, or the latest published nightly pair:

```bash
./ccex install --torch_ver 2.12
./ccex install --torch_ver 2.7    # legacy best-effort
./ccex install --torch_ver 2.10
./ccex install --torch_ver 2.12.1+cu132
./ccex install --torch_ver 2.13
./ccex install --torch_ver nightly          # repository-pinned Torch/TorchVision
./ccex install --torch_ver nightly-latest   # latest published nightly pair
```

TICO keeps Torch 2.5 through 2.9 as legacy best-effort source-install choices,
qualifies 2.10, 2.11, and 2.12, and uses 2.12 as the default. Torch 2.13 is installable
as a qualification candidate but is not part of the release-support window. Family
requests resolve to a project-pinned patch rather than allowing pip to select an
arbitrary patch release. The package metadata itself keeps a bare `torch` dependency,
so a normal `pip install` does not reject a user-managed older version. `nightly` uses
the versions pinned under `infra/dependency/`; `nightly-latest` resolves Torch and
TorchVision together from one moving nightly index.

Compute-platform options:

```bash
# Force a CPU wheel.
./ccex install --cpu_only

# Override detected host CUDA capability when selecting a compatible wheel.
./ccex install --cuda_ver 12.8
```

`--cpu_only` and `--cuda_ver` are mutually exclusive.

When no Torch version is explicitly requested and a configured stable Torch package is
already installed, `./ccex install` preserves that installation if its compute platform
is compatible. This includes legacy best-effort families. `./ccex configure test` then
installs the matching TorchVision package
and verifies the final package pair with `pip check`. Nightly selectors are deliberately
re-resolved: `nightly` restores the repository pin, while `nightly-latest` upgrades the
Torch/TorchVision pair together before test configuration validates it.

The source of truth for families, exact patches, CUDA wheel variants, and CI matrices is
[`tico/utils/compat/torch_version_policy.py`](../tico/utils/compat/torch_version_policy.py).
See the [PyTorch Version Policy](./torch_version_policy.md) for qualification,
promotion, and release-branch rules.

## Build and install a wheel

Build the distribution artifacts:

```bash
./ccex build
```

Install from `dist/` instead of editable source:

```bash
./ccex install --dist
```

A clean wheel workflow similar to CI is:

```bash
./ccex build
./ccex install --dist --torch_ver 2.12
./ccex configure test --torch_ver 2.12
pt2-to-circle -h
```

## Run tests

### Default suite

```bash
./ccex test
```

With no filter, `ccex` runs `unittest` discovery under `test/`. Large model tests are
selected separately with `-m` and are not part of the normal model-independent suite.

### Keyword filtering

```bash
./ccex test -k add
./ccex test -k ConvertMatmulToLinear
./ccex test -k quantization
```

The shorthand keywords `op` and `net` are expanded to the corresponding generated
module-test namespaces:

```bash
./ccex test -k op
./ccex test -k net
```

Use the narrowest relevant test first, then expand to the owning subsystem and finally
to the complete suite for cross-cutting changes.

### Other test-runner options

```bash
./ccex test --all                 # Explicit full suite
./ccex test -i                    # Include internal-only tests
./ccex test -v                    # Set TICO_LOG=4 for this run
./ccex test -r onert              # Select onert for runtime parity checks
./ccex test -p                    # Run performance benchmarks
```

`--all` and `--keyword` cannot be combined. `--model` also cannot be combined with
`--all` or `--keyword`.

## Test organization

```text
test/
├── modules/                 # Small PyTorch modules used by generated E2E tests
│   ├── op/
│   ├── net/
│   └── model/               # Opt-in dependency-isolated model tests
├── unit_test/               # Focused tests for core conversion and Circle utilities
│   ├── circle/
│   ├── ops/
│   ├── passes/
│   ├── quantization/
│   ├── serialize/
│   └── utils/
├── quantization/            # Algorithms, WrapQ, recipes, configs, export, and analysis
├── support/                 # Shared runtime and test-builder utilities
├── performance/             # Llama decoder-layer conversion/size benchmarks
└── pt2_to_circle_test/      # CLI/API and end-to-end conversion tests
```

The module test harness normally:

1. Creates the PyTorch reference outputs before export.
2. Exports directly or saves and reloads a `.pt2` file.
3. Converts the `ExportedProgram` to Circle.
4. Runs `circle2circle` to validate the serialized model.
5. Executes with `circle-interpreter` or `onert`, unless inference is disabled by a
   test tag.
6. Compares output count, shape, dtype, and values with explicit tolerances.

See [System Test Guide](./system_test.md) for the complete test strategy.

## Runtime selection

The default runtime for module parity tests is `circle-interpreter`:

```bash
./ccex test -k add
```

Select `onert` from the command line or environment:

```bash
./ccex test -r onert -k add
CCEX_RUNTIME=onert ./ccex test -k add
```

A test module may explicitly require `onert`; this is used for dynamic-shape execution.
The test setup installs the project-pinned pre-release `onert` package from
`test/requirements_pre.txt`.

## Model tests

Model tests are selected by directory name or shell-style pattern:

```bash
pip install -r test/modules/model/<model_name>/requirements.txt
./ccex test -m <model_name>

# Quote wildcard patterns so the shell does not expand them.
./ccex test -m "Llama*"
```

The `-m` option sets `CCEX_TEST_MODEL` and runs the model-test loader. Model tests may
have additional dependencies and can be significantly more expensive than synthetic
unit or module tests. Do not use a full model test when a small deterministic graph can
cover the behavior.

## Debug conversion

### Log levels

Set `TICO_LOG` before the Python process imports TICO:

```bash
TICO_LOG=4 ./ccex test -k add
TICO_LOG=4 python examples/my_conversion.py
```

| Value | Level |
|---:|---|
| `1` | fatal |
| `2` | warning |
| `3` | info |
| `4` | debug |

Debug mode includes instrumented graph and constant-size diffs around passes and
conversion phases.

### Intermediate graph images

```bash
TICO_GRAPH_DUMP=1 ./ccex test -k add
```

The main conversion pipeline writes these stage snapshots when applicable:

```text
.tico_tmp/session_<timestamp>/1_after_decompose.png
.tico_tmp/session_<timestamp>/2_after_legalize.png
.tico_tmp/session_<timestamp>/3_after_quantfold.png
```

Graph rendering uses `pydot` and Graphviz. The output directory is session-scoped; do
not commit generated images.

### Circle artifact inspection

```bash
tico-circle inspect model.circle --tensors --operators
tico-circle verify model.circle
tico-circle extract model.circle --ops 20-64 -o region.circle
```

Use `tico-circle verify` for static Circle consistency. Use an end-to-end runtime test
for numerical parity and backend execution.

## Formatting and static checks

Install the tools:

```bash
./ccex configure format
```

Apply formatter-generated patches:

```bash
./ccex format
```

Check all files without applying changes, as CI does:

```bash
./ccex format --no-apply-patches
```

Check only files changed from the local `main` branch:

```bash
./ccex format --diff-only --no-apply-patches
```

The current lintrunner configuration executes:

- Pylint
- ufmt (Black-compatible formatting plus import sorting)
- mypy

## Coverage

Run the discovery suite under `coverage` and print a terminal report:

```bash
./ccex coverage
```

Write a report under `test/reports/cov/`:

```bash
./ccex coverage -f txt
./ccex coverage -f xml
```

If `coverage` is not installed, the current helper requests version 7.6.1.

## Pull-request CI

The PR workflow targets `main` and `rel/*`. PyTorch versions are resolved from the
central policy module rather than duplicated in workflow YAML.

1. **Commit-message check**
   - Runs when the pull request is ready for review.
   - Requires at least one commit-body line in the form:

     ```text
     TICO-DCO-1.0-Signed-off-by: <NAME> <<EMAIL>>
     ```

2. **Style check**
   - Runs on Ubuntu 24.04 with Python 3.12.
   - Executes `./ccex configure format` and
     `./ccex format --no-apply-patches`.

3. **Package build**
   - Builds the TICO wheel once.
   - Uploads one short-lived artifact reused by all versioned test jobs.

4. **Versioned tests**
   - Runs the complete suite on the default qualified family, currently 2.12.
   - Runs blocking export and quantization smoke tests on the oldest supported family,
     currently 2.10.
   - Runs the same smoke tests non-blockingly on the qualification candidate, currently
     2.13.

A separate compatibility workflow runs `nightly-latest` smoke tests daily and the
complete supported/candidate/`nightly-latest` matrix weekly. Official package
publication runs the full suite on every qualified stable family before publishing. See the
[PyTorch Version Policy](./torch_version_policy.md) for the exact tiers.

Performance tests are available through `./ccex test -p`, but are not part of the
current PR matrix.

## Change-specific guidance

### Add or change a PyTorch-IR pass

- Implement `PassBase.call()` and return `PassResult(modified=...)` accurately.
- Preserve `ExportedProgram` graph-signature mappings while replacing nodes.
- Update metadata for newly created or rewritten tensors.
- Add focused coverage under `test/unit_test/passes/`.
- Include both matching and structurally similar non-matching cases.
- Add an end-to-end module test when serialization or runtime behavior changes.

### Add a serialized operator

- Add or update a `NodeVisitor` under `tico/serialize/operators/`.
- Register every supported ATen overload with `register_node_visitor`.
- Add serializer/operator unit coverage and a module conversion/parity test.
- Do not leave an unsupported operator in the final graph and rely on a runtime to
  reject it; TICO validates supported targets before serialization.

### Change quantization

- Preserve the `prepare -> calibration/statistics -> convert` lifecycle.
- Put generic algorithms and infrastructure in their owning quantization packages.
- Put model-family behavior in recipe adapters rather than generic stages.
- Prefer deterministic synthetic tests before full model evaluation.
- Update the relevant quantization README or configuration reference when behavior is
  user visible.

### Change Circle artifact tools

- Keep PyTorch-IR passes in `tico/passes/` and serialized Circle-to-Circle passes in
  `tico/circle/passes/`.
- Verify index remapping, graph boundaries, buffers, signatures, multi-subgraph
  behavior, and cleanup contracts as applicable.
- Follow the dedicated pass guidance in
  [Circle artifact tools](../tico/circle/README.md#writing-a-new-circle-pass).

## See also

- [System Design](./design.md)
- [Requirements](./requirements.md)
- [System Test Guide](./system_test.md)
- [Quantization recipes developer guide](../tico/quantization/recipes/README.md)
