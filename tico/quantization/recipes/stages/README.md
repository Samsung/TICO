# Stage Developer Guide

Stages represent algorithmic steps in a quantization pipeline.

Examples:

```yaml
pipeline:
  - name: gptq
    enabled: true
    weight_bits: 4

  - name: ptq
    enabled: true
    activation: int16
```

Each item in `pipeline` is dispatched to a `Stage` implementation under
`recipes/stages/`.

## Stage contract

```python
class Stage(ABC):
    name: str

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        ...
```

A stage receives the current `RecipeContext` and its own config block. It should
return the updated context.

Most stages update `ctx.model`:

```python
prepared = prepare(ctx.model, config)
# collect statistics / run calibration / run optimization
ctx.model = convert(prepared)
return ctx
```

## When to add a new stage

Add a new stage for a reusable algorithmic operation, such as:

- a new weight-only quantization algorithm
- a new activation calibration algorithm
- a graph/module preprocessing pass
- a post-training optimization pass
- a quantization-aware wrapper transformation

Do **not** add a stage for:

| Case | Better location |
|---|---|
| Model loading | `recipes/adapters/` |
| Dataset loading | `recipes/data/` |
| VQA/MMLU/perplexity evaluation | `recipes/evaluation/` |
| Checkpoint/Circle saving | `recipes/export/` |
| Tensor tracing or breakpoint debugging | `recipes/debug/` |
| One specific demo command | Usually a config preset |

## Step-by-step: adding a new algorithm stage

Assume the new stage is `awq`.

### 1. Create the stage file

```text
tico/quantization/recipes/stages/awq.py
```

Skeleton:

```python
from typing import Any, Mapping

from tico.quantization import convert, prepare
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.base import Stage
from tico.quantization.recipes.utils import stage_payload


class AWQStage(Stage):
    name = "awq"

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        payload = stage_payload(stage_cfg)
        # Build the algorithm config from payload.
        config = AWQConfig(**payload)

        prepared = prepare(ctx.require_model(), config, inplace=bool(payload.get("inplace", True)))

        ctx.adapter.forward_calibration(
            ctx,
            prepared,
            ctx.calibration_inputs,
            desc="AWQ calibration",
        )

        ctx.model = convert(prepared, inplace=True)
        return ctx
```

### 2. Register the stage

Edit `recipes/stages/__init__.py`:

```python
from tico.quantization.recipes.stages.awq import AWQStage

_STAGE_REGISTRY = {
    "gptq": GPTQStage(),
    "ptq": PTQStage(),
    "awq": AWQStage(),
}
```

### 3. Add config usage

```yaml
pipeline:
  - name: awq
    enabled: true
    weight_bits: 4
    group_size: 128

  - name: ptq
    enabled: true
    activation: int16
```

### 4. Add tests

Recommended tests:

```text
test/quantization/recipes/test_stage_registry.py
test/quantization/recipes/test_awq_config.py
test/quantization/wrapq/integration/test_awq_ptq_smoke.py
```

## Model-family-specific behavior

Stages should avoid branching on model family when possible.

Prefer this:

```python
ctx.adapter.forward_calibration(ctx, prepared, ctx.calibration_inputs, desc="...")
```

instead of this:

```python
if ctx.adapter.family == "llama":
    prepared(input_ids)
elif ctx.adapter.family == "qwen3_vl":
    prepared(**batch)
```

If the algorithm truly needs model-family-specific metadata, add a clearly named
adapter hook. Example:

```python
ctx.adapter.get_awq_target_modules(ctx, stage_cfg)
```

## Config handling

Use `stage_payload(stage_cfg)` to remove generic keys such as `name` and
`enabled` before passing values to an algorithm config class.

```python
payload = stage_payload(stage_cfg)
config = SomeAlgorithmConfig(**payload)
```

For dataclass configs, use `filter_dataclass_kwargs` to ignore YAML values that
are intended for other layers of the recipe:

```python
config = SomeConfig(**filter_dataclass_kwargs(SomeConfig, payload))
```

Fail early on unsupported values instead of silently ignoring them when the user
is likely to be surprised.

## In-place behavior

Be explicit about in-place behavior.

Recommended pattern:

```python
prepared = prepare(ctx.require_model(), config, inplace=True)
ctx.model = convert(prepared, inplace=True)
```

If a stage needs to keep the original model for comparison or debugging, store it
in `ctx.artifacts` with a clear name:

```python
ctx.artifacts["fp_model_before_awq"] = original_model
```

Avoid hidden deep copies in production stages unless the config explicitly asks
for them.

## Stage ordering

The order in the YAML file is the execution order.

Common patterns:

```yaml
# Weight preprocessing, then weight quantization, then PTQ wrapping
pipeline:
  - name: spinquant
    enabled: true
  - name: gptq
    enabled: true
  - name: ptq
    enabled: true
```

```yaml
# PTQ-only smoke test
pipeline:
  - name: ptq
    enabled: true
```

A stage should not reorder the pipeline by itself.

## Qparam transfer

If an algorithm produces weight qparams that PTQ wrappers should reuse, keep the
transfer logic in a shared helper such as `recipes/qparams.py`.

Do not duplicate module traversal and observer injection logic in every stage.

## Stage checklist

Before merging a new stage:

- Stage file exists under `recipes/stages/`.
- Stage is registered in `recipes/stages/__init__.py`.
- Config block uses lower snake case `name`.
- Stage delegates model-family forward passes to the adapter.
- Stage respects `enabled: false` through the runner.
- Stage does not import from `examples/`.
- A small config can exercise the stage with `n_samples=1`.
- Errors are explicit for unsupported configs or missing model metadata.
