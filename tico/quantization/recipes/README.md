# Quantization Recipes Developer Guide

`tico.quantization.recipes` is the reusable implementation layer behind the thin
example CLIs.

The goal of this package is to keep example scripts small while still supporting
many combinations of model families, quantization algorithms, calibration data,
evaluation tasks, and export targets.

## Design goals

1. **Examples are thin entrypoints.**
   `tico/quantization/examples/*.py` should parse CLI arguments, load a
   config, and call recipe code. They should not own quantization logic.

2. **Configurations describe workflows.**
   A new LLaMA GPTQ preset, a Qwen3-VL PTQ-only preset, or a small smoke-test
   preset should usually be added as a YAML file, not as a new Python script.

3. **Model-specific behavior lives in adapters.**
   Tokenization, processors, calibration input format, model-family PTQ config
   construction, and model-family evaluation belong in `recipes/adapters/`.

4. **Algorithm-specific behavior lives in stages.**
   GPTQ, PTQ, SmoothQuant, SpinQuant, CLE, or future algorithms belong in
   `recipes/stages/`.

5. **Examples must not import other examples.**
   Shared functions must be moved into `recipes`, `evaluation`, `export`,
   `data`, or `debug` modules.

6. **The pipeline follows the WrapQ lifecycle.**
   Stages should preserve the expected flow:

   ```text
   prepare -> calibrate/statistics collection -> convert
   ```

7. **Developer workflows and public examples are separated.**
   Layer-level parity checks, tensor tracing, breakpoint-heavy debugging, and
   synthetic smoke tools should live under `recipes/debug` and be exposed through
   `examples/inspector.py`, not as one-off public example scripts.

## Directory layout

```text
tico/quantization/recipes/
├── README.md
├── __init__.py
├── config.py              # Config loading, dotted overrides, effective config saving
├── context.py             # RecipeContext shared by adapters and stages
├── runner.py              # Pipeline runner used by examples/quantize.py
├── utils.py               # Small reusable helpers
├── qparams.py             # GPTQ -> PTQ qparam transfer helpers
├── adapters/              # Model-family-specific behavior
├── stages/                # Algorithm/pipeline-stage-specific behavior
├── data/                  # Calibration data builders
├── evaluation/            # Reusable evaluation helpers
├── export/                # Checkpoint / Circle / other export helpers
└── debug/                 # Trace, parity, and inspection helpers
```

## Control flow

The default quantization flow is:

```text
examples/quantize.py
  └─ load_recipe_config(...)
  └─ QuantizationRunner.run(cfg)
       ├─ get_adapter(cfg["model"]["family"])
       ├─ adapter.load_model(ctx)
       ├─ adapter.build_calibration_inputs(ctx)
       ├─ for each enabled pipeline stage:
       │    └─ get_stage(stage_cfg["name"]).run(ctx, stage_cfg)
       ├─ adapter.evaluate(ctx)
       ├─ adapter.export(ctx)
       └─ save effective_config.yaml
```

`evaluate.py` and `inspector.py` reuse the same adapters and config format, but do
not run the full quantization pipeline unless the debug mode explicitly needs a
prepared/converted model.

## Responsibility boundaries

| Need | Add or modify |
|---|---|
| New model family, tokenizer, processor, calibration input shape, model-specific PTQ config | `recipes/adapters/<family>.py` |
| New quantization algorithm or preprocessing pass | `recipes/stages/<algorithm>.py` |
| New calibration dataset or synthetic data generator | `recipes/data/*.py` |
| New benchmark or metric | `recipes/evaluation/*.py` |
| New checkpoint, Circle, or artifact output | `recipes/export/*.py` |
| New tensor trace, parity, or inspection mode | `recipes/debug/*.py` and `examples/inspector.py` |
| New common command-line workflow | Usually a config file under `examples/configs/` |
| New public top-level user action not covered by quantize/evaluate/inspect | A new script under `examples/`, only after design review |

## Core abstractions

### `RecipeContext`

`RecipeContext` is a mutable object passed through the pipeline. It stores the
loaded config, selected adapter, model, tokenizer or processor, calibration
inputs, runtime device/dtype, output directory, and optional artifacts.

Stages may update `ctx.model` and attach intermediate results in `ctx.artifacts`,
but they should not silently change unrelated config fields.

### `ModelAdapter`

A model adapter owns model-family-specific behavior:

```python
class ModelAdapter(ABC):
    family: str

    def load_model(self, ctx: RecipeContext) -> RecipeContext: ...
    def build_calibration_inputs(self, ctx: RecipeContext) -> list[Any]: ...
    def forward_calibration(self, ctx, model, calibration_inputs, *, desc: str) -> None: ...
    def calibrate_prepared_model(self, ctx, prepared_model, stage_cfg) -> None: ...
    def build_ptq_config(self, ctx, stage_cfg): ...
    def evaluate(self, ctx: RecipeContext) -> None: ...
    def export(self, ctx: RecipeContext) -> None: ...
```

Adapters should be deterministic with respect to `runtime.seed` when possible.

### `Stage`

A stage owns one algorithm or preprocessing pass:

```python
class Stage(ABC):
    name: str

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        ...
```

Stages should be as model-agnostic as practical. If a stage needs a
model-family-specific operation, delegate it to the adapter rather than adding
family-specific branches inside the stage.

## Import rules

Allowed:

```python
# examples -> recipes
from tico.quantization.recipes.runner import QuantizationRunner

# stages -> algorithm configs/utilities
from tico.quantization.config.gptq import GPTQConfig

# adapters -> model-family config builders/evaluation/data/export helpers
from tico.quantization.recipes.data.llm import build_wikitext_calibration_inputs
```

Avoid:

```python
# Do not import one example from another.
from tico.quantization.examples.quantize_qwen3_vl_with_gptq import evaluate_model

# Do not put model-family-specific calibration inside generic stages.
if ctx.adapter.family == "some_new_family":
    ...
```

## Naming conventions

Use lower snake case for model families, stages, config files, and modes:

```text
model.family: qwen3_vl
stage name: smoothquant
config file: qwen3_vl_gptq_ptq.yaml
inspect mode: layer_parity
```

Config presets should follow this pattern:

```text
<family>_<pipeline>[_purpose].yaml
```

Examples:

```text
llama_gptq_ptq.yaml
llama_ptq_only.yaml
qwen3_vl_gptq_ptq.yaml
qwen3_vl_synthetic_smoke.yaml
```

## Runtime and artifact conventions

- Always save `effective_config.yaml` when `export.output_dir` is set.
- Do not store secrets or personal paths in committed configs.
- Prefer explicit `runtime.seed` in every config.
- Prefer small `*_ptq_only.yaml` or `*_smoke.yaml` configs for CI.
- Put large benchmark/evaluation presets in clearly named files.
- Do not require CUDA for the smallest smoke config unless the model itself
  cannot run on CPU.

## Testing checklist

Before merging a new adapter, stage, or public config:

- The new code imports without downloading a model.
- A small config can run with `calibration.n_samples=1`.
- The full pipeline saves `effective_config.yaml` when export is enabled.
- No example imports another example.
- New logic is covered by at least one unit test or integration smoke test.
- Legacy scripts, if kept, are deprecation wrappers only.
