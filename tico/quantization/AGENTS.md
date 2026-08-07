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
- Algorithm-specific README files under `tico/quantization/algorithm/` when modifying
  an existing algorithm.

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

## Testing

Prefer tiny deterministic modules and synthetic inputs.

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
