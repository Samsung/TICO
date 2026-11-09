# Quantization Recipes Developer Guide

`tico.quantization.recipes` is the reusable implementation layer behind the thin
example CLIs.

The goal of this package is to keep example scripts small while still supporting
many combinations of model families, quantization algorithms, calibration data,
evaluation tasks, and export targets.

For the public Python API, start with the [quantization overview](../README.md).
This guide owns extension routing and shared recipe development rules. For
runnable commands and YAML settings, use the
[CLI examples](../examples/README.md) and
[configuration reference](../examples/configs/README.md).

## Recipe guides

| Guide | Scope |
|---|---|
| [Model adapters](adapters/README.md) | Model loading, input preparation, family-specific configuration, and adapter registration. |
| [Pipeline stages](stages/README.md) | Stage contracts, algorithm orchestration, and stage registration. |
| [Data builders](data/README.md) | Calibration inputs, dataset roles, provenance, and synthetic data. |
| [Evaluation helpers](evaluation/README.md) | FP/Quant quality protocols, task selection, and benchmark integration. |
| [Export helpers](export/README.md) | Artifact writers, static profiles, staged Circle contracts, and saved bundles. |
| [Debug and inspection](debug/README.md) | Tensor tracing, wrapper smoke checks, and export regression workflows. |

## Design goals

1. **Examples are thin entrypoints.**
   `tico/quantization/examples/*.py` should parse CLI arguments, load a
   config where needed, and call recipe code. They should not own quantization
   or artifact-execution logic.

2. **Configurations describe workflows.**
   A new LLaMA quantization preset, a Qwen3-VL PTQ-only preset, or a small
   smoke-test preset should usually be added as a YAML file, not as a new Python
   script.

3. **Model-specific behavior lives in adapters.**
   Tokenization, processors, calibration input format, model-family PTQ config
   construction, and model-family evaluation belong in `recipes/adapters/`.

4. **Stages orchestrate algorithms within a recipe.**
   GPTQ, PTQ, SmoothQuant, SpinQuant, and CLE pipeline integration belongs in
   `recipes/stages/`. Numerical algorithm implementations belong in `algorithm/`,
   and generic wrapper-based PTQ infrastructure belongs in `wrapq/`.

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
├── extensions.py          # Opt-in loading of out-of-tree extensions listed in configs
├── utils.py               # Small reusable helpers
├── qparams.py             # GPTQ -> PTQ qparam transfer helpers
├── adapters/              # Model-family-specific behavior
├── stages/                # Pipeline orchestration for algorithms and preprocessing
├── data/                  # Calibration data builders
├── evaluation/            # Reusable evaluation helpers
├── export/                # Checkpoint / Circle / other export helpers
├── runtime/               # Generation and cache composition from saved Circle bundles
└── debug/                 # Trace, parity, and inspection helpers
```

## Control flow

The default quantization flow is:

```text
examples/quantize.py
  └─ load_recipe_config(...)
  └─ QuantizationRunner.run(cfg)
       ├─ load_recipe_extensions(cfg)        # only when cfg["extensions"] is set
       ├─ resolve_adapter(cfg)               # model.adapter or model.family
       ├─ validate_adapter_evaluation_config(adapter, cfg)
       ├─ validate pipeline stages and determine calibration requirements
       ├─ adapter.load_model(ctx)
       ├─ adapter.build_calibration_inputs(ctx)  # only when required
       ├─ for each enabled pipeline stage:
       │    └─ get_stage(stage_cfg["name"]).run(ctx, stage_cfg)
       ├─ adapter.evaluate(ctx)
       ├─ adapter.export(ctx)
       └─ save effective_config.yaml
```

Before loading the model, the runner checks every pipeline entry and resolves
every enabled stage name, even when calibration is already required by an earlier
stage or the adapter. Invalid entries and unknown enabled stages therefore fail
before model loading or calibration. Disabled stage names are not resolved, and
legacy stages without calibration metadata still require calibration inputs.

`evaluate.py` and `inspector.py` reuse the same adapters and config format, but do
not run the full quantization pipeline unless the debug mode explicitly needs a
prepared/converted model. `export.py` also reuses adapters and config format, but
loads a saved checkpoint and calls the adapter export path without building
calibration inputs or running quantization stages.

`generate.py` consumes a saved bundle through `recipes/runtime/`; it does not
load a recipe config, adapter, source model, or checkpoint. The LLaMA runtime
executes staged Circle artifacts and owns mask, RoPE, and KV-cache composition.
See the [bundle contract](export/llama_artifacts.md#llama-circle-bundles) for the reusable
API and the [generation CLI](../examples/README.md#generate-from-a-llama-circle-bundle)
for prompt and numeric-input examples.

## Responsibility boundaries

| Need | Add or modify | Implementation guide |
|---|---|---|
| New model family, tokenizer, processor, calibration input shape, model-specific PTQ config | `recipes/adapters/<family>.py` | [Model adapters](adapters/README.md) |
| New quantization algorithm or preprocessing implementation | `algorithm/<algorithm>/`, with configuration under `config/` | [Quantization contribution guide](../README.md#contributing-adding-a-new-algorithm) |
| Integrating an algorithm or preprocessing step into a recipe pipeline | `recipes/stages/<algorithm>.py` | [Pipeline stages](stages/README.md) |
| New calibration dataset or synthetic data generator | `recipes/data/*.py` | [Data builders](data/README.md) |
| New benchmark or metric | `recipes/evaluation/*.py` | [Evaluation helpers](evaluation/README.md) |
| New checkpoint, Circle, or artifact output | `recipes/export/*.py` | [Export helpers](export/README.md) |
| Generation and cache composition from saved model-family artifacts | `recipes/runtime/<family>.py` | [Saved LLaMA bundle contract](export/llama_artifacts.md#llama-circle-bundles) |
| New tensor trace, parity, or inspection mode | `recipes/debug/*.py` and `examples/inspector.py` | [Debug and inspection](debug/README.md) |
| New common command-line workflow | Usually a config file under `examples/configs/` | [Configuration reference](../examples/configs/README.md) |
| New public top-level user action not covered by an existing command | A new script under `examples/`, only after design review | [Command selection](../examples/README.md#which-command-should-i-use) |

### Choosing a preset or entrypoint

Add a YAML preset when the model family already has an adapter and the algorithm
already has a stage. This includes new combinations of existing stages and changes
to calibration size, dtype, qscheme, evaluation, or export. Run the preset through
the existing `quantize`, `evaluate`, `export`, or `inspector` command; document new
config fields in the [configuration reference](../examples/configs/README.md).

Avoid model/algorithm-specific scripts such as `quantize_llama_with_gptq.py`,
`quantize_qwen3_vl_ptq_only.py`, or `quantize_vision_attention_debug.py`. Extend the
owner in the table above first. A new Python entrypoint is appropriate only for a
new user-facing top-level action that does not fit `quantize.py`, `evaluate.py`,
`export.py`, `generate.py`, or `inspector.py`.

## Core abstractions

### `RecipeContext`

`RecipeContext` is a mutable object passed through the pipeline. It stores the
loaded config, selected adapter, model, tokenizer or processor, calibration
inputs, runtime device/dtype, output directory, and optional artifacts.

Stages may update `ctx.model` and attach intermediate results in `ctx.artifacts`,
but they should not silently change unrelated config fields.

### `ModelAdapter`

A model adapter owns loading, calibration inputs and invocation, model-family
PTQ configuration, evaluation, and export. Adapters should be deterministic with
respect to `runtime.seed` when possible. The
[adapter contract](adapters/README.md#adapter-contract) defines the methods;
[GPTQ config class selection](adapters/README.md#gptq-config-class-selection) and
the [evaluation target registry](adapters/README.md#evaluation-target-registry)
define the optional GPTQ hook and evaluation declarations. User-facing target
selection is documented in the
[configuration reference](../examples/configs/README.md#selecting-top-level-evaluation-targets).

#### Adapter registry and out-of-tree adapters

Built-in adapters are registered in `recipes/adapters/__init__.py`. Separately
installed packages can add adapters without editing TICO:

```python
from tico.quantization.recipes.adapters import register_adapter

register_adapter("gemma4_variant", MyGemma4Variant())  # family stays "gemma4"
```

Registry rules:

- Keys are normalized (stripped, lower-cased) like `get_adapter()` lookups.
- Registering the same adapter object under the same key again is a no-op.
- Registering a different adapter under an occupied key raises `ValueError`.
  Built-in keys are therefore never replaced; pick a distinct key.
- `resolve_adapter(cfg)` selects `model.adapter` when set, otherwise
  `model.family`, and requires the selected adapter's `family` to equal
  `model.family` so family-keyed helpers (dataset defaults, the GPTQ config
  family fallback, the PTQ selector structure fallback, export) keep working.
- PTQ selector `override_policies` resolve model structure through the selected
  adapter's `build_quant_target_context()` hook, falling back to the registered
  family adapter; see "PTQ selector override structure" in
  [`adapters/README.md`](adapters/README.md).

#### Extension loading

Registration must happen in the process that runs the recipe. Configs opt in
with a top-level `extensions` list of `package.module[:callable]` entries,
which `load_recipe_extensions(cfg)` imports and invokes before adapter
resolution in `quantize`, `evaluate`, `export`, `compare_side_by_side`, and
`inspector --mode trace`. The other inspector smoke/static-runtime modes do not
load extensions or resolve a recipe adapter. Loading runs trusted Python code
from the environment; missing modules or callables raise instead of being
skipped by these entrypoints. See the
[extension schema](../examples/configs/README.md#extensions-optional) for the config
schema. Python callers can equivalently call the extension's activation
function themselves before using the recipe APIs.

### `Stage`

A stage orchestrates an algorithm or preprocessing step within the recipe pipeline.
It receives the current `RecipeContext` and its own config block and returns the
updated context. See the [stage contract](stages/README.md#stage-contract) for the
interface and lifecycle example.

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
config file: qwen3_vl_quantize.yaml
inspect mode: layer_parity
```

Follow the [config file naming convention](../examples/configs/README.md#file-naming)
for preset names and examples.

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
