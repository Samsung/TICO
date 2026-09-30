# Quantization Agent Rules

## Scope

These rules apply to changes under `tico/quantization/` in addition to the repository
root `AGENTS.md`.

## Required context

Read the documents relevant to the change:

- Quantization architecture and public API: `tico/quantization/README.md`
- Wrapper, observer, and fake-quant infrastructure:
  `tico/quantization/wrapq/README.md`
- Recipe ownership, stages, adapters, import rules, configs, and debug workflows:
  `tico/quantization/recipes/README.md`
- Dataset roles, provenance, and benchmark safety:
  `tico/quantization/recipes/data/README.md`
- Artifact boundaries, static profiles, and staged export:
  `tico/quantization/recipes/export/README.md` and the relevant model-specific guide
- Public configuration and extension schema:
  `tico/quantization/examples/configs/README.md`
- Algorithm-specific README files under `tico/quantization/algorithm/` when available;
  otherwise inspect the implementation and its focused tests.

Inspect a nearby implementation and its tests before adding a new quantizer, wrapper,
adapter, stage, export path, or configuration field.

## Architectural boundaries

- Quantization algorithms belong in `algorithm/`.
- Generic wrapper, observer, fake-quant, and module infrastructure belongs in
  `wrapq/`.
- Algorithm and policy configuration belongs in `config/`.
- Model-family-specific behavior belongs in `recipes/adapters/` or a registered
  model-family wrapper.
- Algorithm pipeline orchestration belongs in `recipes/stages/`.
- Reusable calibration, evaluation, export, and debugging code belongs in the matching
  `recipes/` package.
- New workflow combinations should normally be YAML presets under
  `examples/configs/`, not new Python scripts.
- Example scripts must remain thin and must not import other example scripts.

Do not add a model-family conditional to generic infrastructure when the behavior can
be expressed through an adapter, wrapper, registration, protocol method, or
configuration.

## Out-of-tree extensions

Use the public adapter/quantizer registries and opt-in recipe extensions for separately
installed packages instead of patching built-in registries or adding private imports.
Follow `recipes/README.md` for the API and collision rules.

- `model.adapter` selects an adapter; it does not redefine `model.family`. Preserve
  family-consistency validation and the family keys used by downstream helpers.
- Preserve idempotent registration of the same object/class and reject a different
  registration under an occupied key. Do not silently replace built-ins.
- Load configured extensions before adapter resolution in every affected entrypoint.
  Loading executes trusted Python code; never discover or activate packages implicitly,
  and do not swallow import or activation failures.
- Test repeated loading, collisions, missing modules/callables, and family mismatches
  with synthetic extensions. Restore registry and loader state after each test.

## Quantization lifecycle

Preserve the expected lifecycle:

```text
prepare -> calibration/statistics collection -> convert
```

- `prepare` may install wrappers, observers, hooks, or algorithm state.
- Calibration or statistics collection must happen while the prepared state is valid.
- `convert` must consume that state deterministically and produce the documented
  quantized representation.
- Do not collect statistics after conversion or silently mutate an already-converted
  model unless the API explicitly defines that behavior.
- Keep observer-enabled, fake-quant-enabled, and quantization-mode transitions
  explicit. Do not rely on an unrelated caller to leave global state in the expected
  mode.
- State-dict save and load behavior must preserve the documented lifecycle stage and
  qparams.
- For caching quantizers, distinguish collecting calibration inputs from collecting
  algorithm statistics during conversion. Preserve cache ownership, collection order,
  and state transitions; test nested or repeated module calls when affected. Verify
  hook/forward restoration and cache cleanup at the documented lifecycle boundaries,
  including failure paths.

## Qparam correctness

Whenever a change affects quantization parameters, make the following explicit in code
and tests:

- storage and computation dtype;
- quantization range and bit width;
- symmetric or asymmetric mapping;
- signed or unsigned representation;
- per-tensor, per-channel, per-group, or other granularity;
- channel or group axis;
- scale and zero-point dtype and shape;
- observer and fake-quant enabled state;
- rounding and clamping behavior;
- behavior for zero ranges, empty calibration, non-finite values, and degenerate
  tensors.

Do not infer a channel axis solely from tensor rank when module semantics provide the
correct axis. Do not transfer or reuse qparams across tensors unless their semantic
mapping is proven compatible.

A change to qparam propagation, folding, sharing, or transfer must include tests for
both the intended propagation path and a nearby path that must not propagate.

### GPTQ-to-PTQ handoff

When GPTQ qparam reuse is enabled, preserve the stage ordering:

```text
PTQ prepare -> inject and lock compatible GPTQ weight qparams -> calibrate -> convert
```

Use `recipes/qparams.py` and the existing PTQ-stage flow rather than a parallel
handoff implementation. Preserve wrapper `fp_name` mappings and folding-aware scale
adjustments. Loaded weight qparams must stay locked through calibration without
preventing unrelated activation observers from collecting statistics.

Keep failure modes distinct: enabled GPTQ followed by requested reuse must not silently
fall back when quantizers are missing or no observer matches. Pure PTQ and explicitly
disabled reuse remain valid workflows. Test mappings, folded weights, lock persistence,
non-target observers, and those separate workflow paths when the handoff changes.

## Numerical behavior

- Use numerically stable accumulation and dtype conversions appropriate to the
  algorithm.
- Preserve device placement unless the API explicitly moves state or tensors.
- Avoid hidden host-device transfers in generic code.
- Keep random sampling deterministic through explicit seeds.
- Do not silently replace NaN, infinity, or invalid qparams unless the policy is
  documented and tested.
- Do not improve one benchmark by hard-coding model names, layer indices, tensor
  shapes, or checkpoint-specific values into generic code.

## Recipes and configurations

- Keep adapters deterministic with respect to configured seeds when practical.
- Stages should be model-agnostic and delegate model-specific operations to adapters.
- Do not silently mutate unrelated configuration fields in `RecipeContext`.
- Save effective configuration when the workflow contract requires it.
- Do not commit secrets, local absolute paths, checkpoints, or private dataset
  locations in YAML files.
- Prefer a small `*_smoke.yaml` or `*_ptq_only.yaml` preset for CI and regression
  testing.
- A new configuration field requires validation, a default or migration strategy, and
  documentation in the relevant config reference.

## Dataset roles and benchmark safety

Use the centralized policy in `recipes/data/dataset_usage.py`; a split named `train`
is not by itself proof that a dataset is calibration-safe. Preserve role validation
before model loading or data downloads, and preserve resolved source provenance in
`effective_config.yaml`.

Do not enable `calibration.allow_benchmark_overlap` or
`calibration.allow_unregistered_dataset` automatically to bypass errors or improve
scores. Explicit experimental opt-ins must retain their warnings and provenance;
overlap-enabled results must not be described as strictly held out. Keep gold targets
out of calibration rendering by default. Add policy and routing tests for new dataset
sources without downloading those datasets in unit tests.

## Staged export and runtime contracts

Treat each exported stage boundary as an interface, not just a filename. When input or
output order, names, shapes, dtypes, qparams, static profiles, or stage composition
changes, review the wrapper, exporter, runtime consumer, manifests, and contract tests
together. Use `recipes/export/README.md` and the relevant model/profile implementation
as the source of truth instead of copying model dimensions into these rules.

- Reuse the intended frozen observer/qparams at producer-consumer boundaries; do not
  independently recalibrate the two sides. Test that split artifacts can be chained.
- Keep host/NPU responsibilities, prefill/decode, KV-cache capacity/update behavior,
  and RoPE/profile conventions explicit. Reject incompatible profiles rather than
  silently casting, reshaping, or substituting another stage.
- Keep artifact-format policy in recipes and inline/appended Circle storage selection
  in the shared serializer. Large-Circle support does not implicitly change a recipe's
  existing host-artifact `auto` policy; see `docs/large_circle_export.md`.
- Use the default reference runtime for ordinary Circle evaluation. Report structural,
  reference-numerical, and target-backend validation separately, especially for
  fake-quantize execution.

## Testing

Prefer tiny deterministic modules and synthetic inputs under the matching
`test/quantization/` directory. Extend `test/unit_test/quantization/` only when it
already owns the relevant behavior; see `test/AGENTS.md`.

Unit tests must not:

- download Hugging Face models or datasets;
- require credentials or network access;
- require CUDA unless the behavior is inherently CUDA-specific;
- depend on user-specific absolute paths or pre-existing output directories;
- allocate full-size LLM or VLM checkpoints when a small module can prove the
  behavior.

Cover the relevant lifecycle stages separately:

1. preparation and wrapping;
2. statistics collection or calibration;
3. conversion;
4. qparam values, shapes, dtypes, axes, and enabled states;
5. state-dict save and load when affected;
6. export when affected;
7. a non-applicable module or tensor path that must remain unchanged.

For model-family integration, run the smallest existing smoke configuration before a
full-size model workflow. A full model run does not replace a focused synthetic
regression test.

## Common review failures

Reject or revise changes that:

- change granularity or axis implicitly;
- share an observer or fake quantizer across semantically different tensors without a
  documented reason;
- call `convert` before required statistics exist;
- treat disabled fake quantization as equivalent to disabled observation;
- preserve integer values while changing scale, zero-point, or dequantized semantics;
- place model-family logic in a generic stage;
- add another example script for a workflow that a YAML preset can express;
- make tests pass by loosening tolerances without a numerical justification;
- make unit tests depend on remote models, private data, credentials, or GPUs.

## Validation

Run the narrowest relevant tests first:

```bash
./ccex test -k <quantizer-wrapper-observer-or-recipe-keyword>
```

Expand to the owning quantization test group, recipe smoke test, export test, or full
suite when shared infrastructure or lifecycle behavior changes.
