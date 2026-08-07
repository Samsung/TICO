# ExportedProgram Pass Rules

## Scope

These rules apply to changes under `tico/passes/` in addition to the repository root
`AGENTS.md`.

## Required context

Before changing a pass, read the relevant sections of `docs/design.md`, especially the
working IR, graph invariants, pipeline phases, pass ordering, and validation behavior.
Inspect at least one nearby pass and its matching unit test before introducing a new
pattern or utility.

## Pass contract

Every pass must make the following properties clear through its implementation,
docstring, and tests:

- **Preconditions:** exact operator pattern, rank, dtype, shape, constant, metadata, and
  use-count requirements.
- **Transformation:** which nodes, arguments, attributes, state bindings, and metadata
  are replaced or preserved.
- **Postconditions:** the graph property guaranteed after the pass.
- **Semantic assumptions:** any assumption required to preserve observable behavior.

Do not match a broader graph pattern than the transformation can prove safe. Prefer a
conservative non-match over an unsafe rewrite.

## Graph maintenance

- Preserve `ExportedProgram` placeholders, outputs, parameters, buffers, constants,
  and state-dict bindings unless the current pipeline phase explicitly allows changing
  them.
- Follow existing repository patterns for dead-code elimination, graph linting,
  recompilation, metadata refresh, and signature replacement.
- Do not leave stale shape, dtype, layout, value, or alias metadata after changing
  graph structure.
- Do not depend on incidental node names when operator target, arguments, metadata, or
  graph relationships provide a stable match.
- Do not depend on execution order between unrelated users of a value.
- Keep pass-scheduling changes explicit. Avoid hiding a pipeline-order change inside a
  local pattern rewrite.
- Preserve graph outputs and their order unless the task explicitly changes the public
  contract.
- When a rewrite introduces constants, use the representation expected by the current
  invariant phase rather than bypassing repository helpers.
- Reuse existing pass utilities when they encode the same invariant or cleanup
  behavior. Do not create a second subtly different implementation.

## Numerical and structural correctness

A successful rewrite must prove both:

1. The transformed graph computes the intended result for all supported inputs.
2. The transformed structure satisfies the property required by later passes or Circle
   serialization.

Do not test only that a pass returns without raising. Assert the operator targets,
arguments, removed nodes, preserved users, shapes, and dtypes relevant to the pass
contract.

## Tests

Add or update `test/unit_test/passes/test_<pass_name>.py`.

At minimum, cover:

1. A positive matching case.
2. A structurally similar non-matching case.
3. Relevant rank, shape, dtype, constant, and multi-user edge cases.
4. Numerical equivalence when the rewrite changes computation.
5. Structural assertions for the exact property the pass promises.
6. Idempotence when the pass is intended to be safe on an already-transformed graph.

For a bug fix, add a regression test that fails before the fix and identifies the
specific unsafe match or missing rewrite.

For changes that affect Circle serialization or runtime behavior, also add the closest
end-to-end module conversion and parity test.

## Common review failures

Reject or revise changes that:

- read tensor values without proving they are compile-time constants;
- assume a dimension is static without checking symbolic-shape metadata;
- mutate arguments in place while other users still depend on them;
- erase nodes before rewiring all users;
- preserve output values but change dtype, rank, layout, or quantization semantics;
- call broad cleanup routines to hide an invalid intermediate graph;
- broaden matching logic without adding negative tests;
- solve one model-specific pattern by adding model-family checks to a generic pass.

## Validation

Run the closest pass test first:

```bash
./ccex test -k <pass-name-or-keyword>
```

Then run the owning pass unit-test group or full test suite when the change affects
shared helpers, pass ordering, graph invariants, or multiple operators.
