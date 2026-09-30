# TICO Agent Guide

## Scope

These instructions apply to the entire repository.

A more specific `AGENTS.md` in a subdirectory may add to or override these rules for
that subsystem.

## Project purpose

TICO converts PyTorch modules and `torch.export.ExportedProgram` graphs into Circle
models. It also provides Circle artifact tooling and post-training quantization
workflows for neural networks, LLMs, and VLMs.

## Read before editing

Read only the documents relevant to the requested change:

- Development, testing, and formatting: `docs/development.md`
- Torch support tiers, installation, and CI matrices: `docs/torch_version_policy.md`
- Conversion architecture, pass ordering, and graph invariants: `docs/design.md`
- Functional and non-functional requirements: `docs/requirements.md`
- Circle inspection, verification, extraction, optimization, and reference execution:
  `tico/circle/README.md` and `tico/circle/AGENTS.md`
- Large Circle serialization, buffer ownership, and memory limits:
  `docs/large_circle_export.md`
- Quantization architecture: `tico/quantization/README.md`
- Quantization recipe responsibilities and import rules:
  `tico/quantization/recipes/README.md`

Do not duplicate those documents in this file, implementation comments, or new README
files. Link to the source of truth instead.

## Change routing

Use the narrowest subsystem that owns the requested behavior.

| Change type | Primary location |
|---|---|
| ExportedProgram graph legalization or optimization | `tico/passes/` |
| ATen or Circle operator conversion | `tico/ops/`, `tico/serialize/` |
| Circle inspection, verification, extraction, or optimization | `tico/circle/` |
| Circle reference execution and kernels | `tico/circle/runtime/` |
| Runtime selection and optional backend adapters | `tico/interpreter/` |
| Shared Circle binary packing and appended payloads | `tico/serialize/circle_binary.py` |
| Circle document loading and saving | `tico/circle/io.py` |
| Quantization algorithm | `tico/quantization/algorithm/` |
| Quantization configuration | `tico/quantization/config/` |
| Generic quantization infrastructure | `tico/quantization/wrapq/` |
| Model-family-specific quantization behavior | `tico/quantization/recipes/adapters/` |
| Algorithm pipeline behavior | `tico/quantization/recipes/stages/` |
| Calibration, evaluation, export, or debug helper | Matching package under `tico/quantization/recipes/` |
| Out-of-tree recipe activation and adapter registration | `tico/quantization/recipes/extensions.py`, `tico/quantization/recipes/adapters/` |
| CLI workflow combination | `tico/quantization/examples/configs/` |
| Torch version selection and CI matrix policy | `tico/utils/compat/torch_version_policy.py` |
| Tests | Corresponding directory under `test/` |

Before creating a new module, script, or abstraction, inspect nearby implementations
and confirm that an existing owner cannot be extended cleanly.

## Global implementation rules

- Make the smallest coherent change that solves the requested problem.
- Do not refactor unrelated code unless the requested change requires it.
- Preserve public APIs and default behavior unless the task explicitly changes them.
- Preserve the graph invariants and pipeline boundaries documented in
  `docs/design.md`.
- Keep generic passes and generic quantization stages model-agnostic.
- Put model-family-specific behavior in adapters, wrappers, registrations, or
  configuration instead of generic infrastructure.
- Follow existing naming, typing, error-handling, and file-layout conventions in the
  nearest comparable implementation.
- Do not add a production dependency without an explicit requirement and a clear
  justification.
- Resolve supported Torch families and CI matrices through the central version
  policy, and keep nightly pins in the existing `infra/dependency/` files. Do not
  duplicate version pins or support tiers in workflows or agent guides.
- Do not modify vendored or submodule content unless the task explicitly targets it.
- Do not silently catch conversion, export, verification, or quantization failures.
- Do not introduce a semantically different fallback merely to avoid an exception.
- Do not weaken an assertion, broaden a skip, or increase a numerical tolerance merely
  to make a failing test pass.
- Update tests and user-facing documentation when observable behavior changes.
- Never commit checkpoints, generated `.circle` or `.pt2` artifacts, graph dumps,
  credentials, tokens, or user-specific absolute paths.

## Runtime and serialization contracts

- The built-in `reference` runtime is the default. Do not reintroduce ONE/ONERT
  dependencies into default conversion, execution, testing, or quantization evaluation.
  External runtimes require explicit selection, never installation-based discovery or
  silent fallback after a reference-runtime failure.
- Keep structural verification, reference numerical parity, and target-backend
  validation distinct. `CircleDocument.verify()` is not a numerical or backend check;
  fake-quantize reference execution is not bit-exact integer-backend validation.
- Use the shared `tico/serialize/circle_binary.py` serializer for conversion and
  document repacking. Preserve complete-`bytes` return contracts; do not introduce
  model-specific packing paths or silently replace bytes with paths or lazy objects.
- Preserve the configured Circle optimization behavior, including default O1, for
  large exports. Only the documented size-limit condition selects appended storage;
  schema errors, invalid values, and `MemoryError` must not trigger a format fallback.
- Review constant ownership and peak memory when changing serialization or artifact
  transformations. Avoid unnecessary whole-model or payload copies, and do not describe
  the current in-memory implementation as streaming or out-of-core.

## Validation workflow

Start with the narrowest relevant validation and expand only as needed.

For a fresh development environment, activate a virtual environment and install TICO
and Torch before configuring test dependencies:

```bash
./ccex install
./ccex configure test
./ccex configure format
```

`configure test` validates an installed Torch and installs matching test dependencies;
it does not install TICO or Torch. Do not reinstall an already configured environment
for every change. See `docs/development.md` for installation options.

For day-to-day validation:

```bash
# Closest unit or integration tests
./ccex test -k <relevant-keyword>

# Apply repository formatting
./ccex format

# Verify formatting without applying patches
./ccex format --no-apply-patches
```

Run the full non-model test suite for cross-cutting changes:

```bash
./ccex test
```

Run model tests only when the change affects that model family or cannot be covered by
small synthetic tests:

```bash
./ccex test -m <model-name-or-pattern>
```

Do not claim that an unexecuted test passed. In the final report, list exactly which
commands were run, their outcomes, and any validation that was not run.

## Test expectations by area

- `tico/passes/**`: add or update the matching tests under
  `test/unit_test/passes/`.
- `tico/ops/**` or `tico/serialize/**`: add unit coverage and an end-to-end module
  conversion test when applicable.
- `tico/circle/**`: add or update tests under `test/unit_test/circle/`; reference
  kernels and execution contracts belong under its `runtime/` directory.
- Default runtime selection or evaluation plumbing: also run
  `test/unit_test/utils/test_runtime_independence.py`.
- `tico/quantization/**`: normally add deterministic synthetic tests under the matching
  `test/quantization/` directory for algorithms, WrapQ, recipes, configs, export, and
  analysis. Extend `test/unit_test/quantization/` when that is the existing owner of
  the behavior; do not relocate tests merely to match a generic category.
- Public CLI or configuration behavior: update the corresponding README or
  configuration reference.

Prefer a small synthetic regression test over a remote model download whenever both
cover the same behavior.

## Review priorities

When reviewing a change, prioritize behavioral correctness over formatting. Flag the
following as defects:

- A graph rewrite changes shape, dtype, layout, aliasing, state bindings, or output
  semantics without targeted validation.
- A pattern-based pass broadens its match conditions without a structurally similar
  non-matching test.
- A pass-order change is hidden inside an unrelated implementation change.
- Quantization code changes qparam dtype, range, granularity, channel axis, or
  lifecycle ordering without targeted tests.
- Model-family-specific branches are added to a generic pass or generic recipe stage.
- Unit tests require a remote model, internet access, credentials, or a GPU when a
  synthetic CPU test can cover the behavior.
- An exception is swallowed or an unsupported path silently changes semantics.
- A test passes only because its tolerance or skip condition was made unnecessarily
  permissive.

Leave purely mechanical formatting decisions to the repository formatter and CI.

## Commits and pull requests

Only create commits, push branches, or open pull requests when explicitly requested.

When creating a commit for a pull request, follow the repository DCO convention
required by CI and include the required sign-off line in the commit body:

```text
TICO-DCO-1.0-Signed-off-by: <NAME> <<EMAIL>>
```

Do not force-push, rewrite existing commits, discard unrelated working-tree changes,
or modify another contributor's changes unless explicitly requested.

## Definition of done

A change is complete when:

1. The implementation follows subsystem responsibility boundaries.
2. Focused tests cover changed behavior and relevant failure or non-matching cases.
3. Formatting checks pass for the modified files.
4. Public behavior changes are documented.
5. The final report states changed files, validation performed, and remaining risks.
