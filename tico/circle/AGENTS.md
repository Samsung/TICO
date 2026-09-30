# Circle Artifact Agent Rules

## Scope

These rules apply to changes under `tico/circle/` in addition to the repository root
`AGENTS.md`.

This package owns serialized Circle artifacts, Circle-to-Circle transformations, and
reference execution. PyTorch `ExportedProgram` rewrites belong in `tico/passes/`;
do not mix their IR types, pass interfaces, or cleanup utilities.

## Required context and ownership

Read the relevant sections of `tico/circle/README.md` before changing document I/O,
extraction, verification, passes, or runtime behavior. For packing, appended payloads,
or large constants, also read `docs/large_circle_export.md`. Inspect the nearest
implementation and tests under `test/unit_test/circle/`.

- Put semantic rewrites in the owning pass package: `canonicalize`, `simplify`,
  `fold`, `fuse`, `legalize`, `compatibility`, or `cleanup`. Do not recreate removed
  forwarding packages or deprecated CLI names.
- Keep workflow-level extraction in `operations/`, reference execution in `runtime/`,
  and common serialization in `tico/serialize/circle_binary.py`.
- CLI parsing must delegate to library behavior rather than duplicate graph logic.
  Update the README for public options and register one canonical name for each
  user-selectable pass.

## Graph and semantic contracts

- Preserve complete tensor contracts: shape/signature, dtype, quantization parameters,
  and relevant layout and storage semantics. Real-number algebra alone does not prove
  a quantized identity or floating-point reassociation safe.
- Preserve graph I/O order and identities, names, signatures, and observable effects
  unless the workflow explicitly changes them. Use existing extraction boundary and
  signature policies rather than silently synthesizing incompatible signatures.
- Use `rewrite.py` helpers when deleting or remapping indexed objects. Account for
  model-global buffers, operator codes, subgraphs, metadata, and subgraph-local tensor
  and signature mappings across every retained subgraph; preserve buffer 0.
- Use shared purity/effect analysis for DCE and CSE. Unknown/custom, stateful, variable,
  random, or subgraph-referencing operators are not automatically removable merely
  because no graph output consumes their results.
- Do not mutate shared constant weights in place when another tensor or subgraph still
  uses them. Use the existing builders and constant pool for replacement constants.
- Match only proven contracts. Treat unsupported dynamic, sparse, quantized, or
  multi-output patterns conservatively rather than guessing their semantics.

## Sessions and atomic rewrites

Prefer `CircleRewriteRule` with `CircleRulePass` for local patterns so the shared
worklist, optimization session, and mutation transaction handle scheduling and
rollback. Use `CircleGraph` or the session graph cache instead of independently
rebuilding producer/consumer indexes.

- Mutate only the supplied document and report `modified`/`changes` accurately.
- Make a local rewrite atomic. Register existing buffer payloads with
  `current_mutation().watch_buffer(...)` before directly modifying those payloads
  inside a transaction; do not assume an unregistered write can be rolled back.
- Preserve the shared session's revision and invalidation rules. Direct Object API
  mutation by existing passes is supported, not forbidden: report changes to the pass
  manager. Standalone mutations outside the manager must mark the session modified
  before cached analyses are reused.
- Keep document invariants valid at pass boundaries. Do not disable verification to
  hide an invalid rewrite; an explicitly requested temporary invalid state requires
  verification after the complete transformation.

## Pipeline scheduling

Preserve O1's ordered complete-round fixed point (`UNTIL_NO_CHANGE`) and one final
index compaction. Do not compact inside a local rule or replace round scheduling with
restart scheduling implicitly. Keep optional legalization and compatibility behavior
explicit, and change `presets.py` only when the built-in pipeline should change.

For pass-order or scheduler changes, test non-empty fixed points, idempotence, and
relevant interactions between passes. Use the existing scheduler-equivalence benchmark
when relevant and an artifact is available; do not claim full-model benchmark results
from synthetic tests alone. Preserve constant-folding profiles and storage/compute
budgets rather than adding unbounded evaluation or a second folding implementation.

## Binary I/O and constant ownership

Load complete binaries through `CircleDocument` or `tico.circle.io`; use the shared
serializer to save/repack them. Do not directly repack a header whose appended buffers
have not been resolved, retain stale file offsets after a rewrite, or silently discard
unsupported payloads. Preserve complete-`bytes` API contracts and configured O1 behavior
regardless of model size.

Appended payloads may be read-only views that retain the source bytes. Copy explicitly
before an in-place edit, but review whole-document clones, `deepcopy`, and `tobytes()`
for unnecessary payload duplication. Header parsing should not copy multi-GiB payloads
merely to make metadata writable. This is an in-memory path, not a streaming guarantee.

Distinguish absent storage from a valid zero-element constant using the existing
ownership helpers; do not classify a tensor only by whether its payload is non-empty.
Preserve CLI binary stdout versus diagnostic stderr, stream support, and atomic file
writes when changing I/O.

## Reference runtime

The built-in runtime executes the serialized Circle model, not the originating
PyTorch graph. It is a correctness reference, not a performance runtime.

- Do not read the source model or introduce ONE/ONERT imports, subprocesses, or native
  compatibility libraries into the default path. External backends are explicit
  adapters, never silent fallbacks for unsupported reference execution.
- Validate serialized input/output contracts, including graph order, dtype, static
  dimensions, and shape signatures. Do not reshape or cast computed results merely
  to satisfy metadata. Keep unsupported behavior explicit and diagnostic.
- Preserve execution-mode boundaries. `NATIVE` must not silently become
  `FAKE_QUANTIZE`; fake-quantize results do not establish bit-exact integer backend
  behavior. Follow the README for the supported modes and operators.
- Keep caller inputs unmodified, repeated-run behavior deterministic, and constant
  ownership explicit. Preserve last-consumer intermediate release unless tracing is
  requested; trace mode may retain substantially more memory.

## Testing and validation

Add focused positive and nearby non-matching tests. For rewrites, assert structural
properties and numerical equivalence where applicable, not just successful execution.
Cover shared buffers, multiple subgraphs, signatures, effects, index remapping, atomic
rollback, and session invalidation when affected. Use generated-schema round trips to
cover serialization contracts in addition to lightweight Object API fixtures.

For runtime kernels, use hand-computed expectations or independent formulas rather
than the same kernel/helper under test, then add the closest PyTorch-to-Circle module
parity test. Cover applicable options, dtypes, dynamic dimensions, repeated runs, input
non-mutation, quantization modes, and diagnostic failures.

Large-buffer CI tests should use small payloads with reduced private size limits;
real multi-GiB or external-backend checks stay opt-in as documented in
`docs/large_circle_export.md`. Keep structural verification, reference parity, and
actual target-backend validation distinct in test names and result reports.

```bash
./ccex test -k <specific-circle-test-or-keyword>
./ccex test -k circle
./ccex test -k runtime_independence  # When default execution plumbing changes
```

Expand to the full non-model suite for shared infrastructure or pipeline changes.
Follow `test/AGENTS.md` and report exactly which checks ran, including skipped or
unexecuted resource-dependent validation.
