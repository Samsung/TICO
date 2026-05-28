# Recipe Config Guide

This directory contains reusable YAML presets for the thin example CLIs.

A config describes **what workflow to run**. Python code describes **how the
workflow is implemented**.

Most new examples should be added as config files in this directory.

## File naming

Use:

```text
<model_family>_<pipeline>[_purpose].yaml
```

Examples:

```text
llama_eval_suite.yaml
llama_gptq_ptq.yaml
llama_ptq_only.yaml
qwen3_vl_eval_suite.yaml
qwen3_vl_gptq_ptq.yaml
qwen3_vl_ptq_only.yaml
```

Use lower snake case. Keep names descriptive but short.

## Top-level schema

```yaml
model:        # Model family and checkpoint loading
runtime:      # Device, dtype, seed, progress behavior
calibration:  # Dataset and calibration sample settings
model_args:   # Optional model-family-specific wrapper/export args
pipeline:     # Ordered quantization/preprocessing stages
evaluation:   # Optional evaluation settings
export:       # Optional artifact output settings
```

Not every section is required for every model family, but committed configs
should be explicit enough to reproduce the intended workflow.

## `model`

```yaml
model:
  family: llama
  name_or_path: Maykeye/TinyLLama-v0
  trust_remote_code: false
  hf_token: null
  cache_dir: null
```

Rules:

- `family` must match a registered adapter in `recipes/adapters/__init__.py`.
- `name_or_path` can be overridden by `--model`.
- Do not commit personal paths or access tokens.
- Prefer `hf_token: null`; pass secrets through the environment or runtime
  tooling instead of YAML.

## `runtime`

```yaml
runtime:
  device: cuda
  dtype: float32
  seed: 42
  show_progress: true
```

Rules:

- Include `seed` in every committed config.
- Use `cpu` for the smallest smoke configs when feasible.
- Use `cuda` for representative full quantization configs when CPU execution is
  unrealistic.
- Keep dtype names simple: `float32`, `float16`, or `bfloat16`.

## `calibration`

LLM example:

```yaml
calibration:
  dataset: wikitext
  dataset_config: wikitext-2-raw-v1
  split: train
  n_samples: 128
  seq_len: 2048
  decode_steps: 0
```

VLM example:

```yaml
calibration:
  dataset: vqav2
  n_samples: 128
  seq_len: 2048
```

Rules:

- `n_samples` should be small in smoke configs and representative in benchmark
  configs.
- `seq_len` should match the intended export/evaluation path when static shapes
  matter.
- Dataset-specific logic belongs in `recipes/data/`, not in config parsing.

## `model_args`

Use `model_args` only for model-family-specific details needed by wrappers,
export, or debug modes.

Example:

```yaml
model_args:
  vision:
    grid_thw: [8, 24, 24]
    visual_start_idx: 0
    spatial_merge_size: 2
```

Rules:

- Keep generic quantization parameters under `pipeline`, not `model_args`.
- Document any new model-family-specific fields in this README.
- The adapter owns interpretation of `model_args`.

## `pipeline`

`pipeline` is an ordered list. The runner executes enabled stages in order.

```yaml
pipeline:
  - name: gptq
    enabled: true
    weight_bits: 4
    perchannel: true

  - name: ptq
    enabled: true
    activation_dtype: int16
    default_qscheme: per_tensor_symm
```

Rules:

- `name` must match a registered stage in `recipes/stages/__init__.py`.
- `enabled: false` keeps a stage documented but skips it.
- Stage ordering is meaningful; do not rely on a stage reordering the pipeline.
- Prefer a new config file when changing the stage list.

### Common stage patterns

GPTQ + PTQ:

```yaml
pipeline:
  - name: gptq
    enabled: true
    weight_bits: 4
    perchannel: true
    symmetric: false

  - name: ptq
    enabled: true
    activation_dtype: int16
    linear_weight_bits: 4
```

PTQ-only smoke:

```yaml
pipeline:
  - name: ptq
    enabled: true
    activation_dtype: int16
    linear_weight_bits: 8
```

Preprocessing + GPTQ + PTQ:

```yaml
pipeline:
  - name: spinquant
    enabled: true

  - name: gptq
    enabled: true
    weight_bits: 4

  - name: ptq
    enabled: true
```

## `evaluation`

LLM example:

```yaml
evaluation:
  enabled: true
  perplexity:
    dataset: wikitext
    dataset_config: wikitext-2-raw-v1
    split: test
  lm_eval_tasks: null
  max_seq_len: 2048
```

VLM example:

```yaml
evaluation:
  enabled: false
  vlm_tasks:
    - vqav2
    - textvqa
  coco: false
  llava_bench: false
  mmlu:
    enabled: false
    subjects: null
    n_shots: 5
    n_samples: -1
    batch_size: 1
  n_samples: 50
  max_seq_len: 2048
```

Rules:

- Expensive evaluation should default to `enabled: false` in smoke configs.
- Evaluation helpers belong in `recipes/evaluation/`.
- The model adapter decides which evaluation fields are meaningful.

## `export`

```yaml
export:
  enabled: false
  output_dir: ./out/llama
  artifacts:
    - ptq_checkpoint
```

Rules:

- `effective_config.yaml` should be saved under `output_dir` when export is
  enabled or an output directory is provided.
- Use relative paths in committed configs.
- Artifact names should be stable and lower snake case.
- Export implementation belongs in `recipes/export/` and adapter orchestration.

## CLI overrides

Simple values can be overridden with `--set`:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_ptq_only.yaml \
  --set runtime.device=cpu \
  --set calibration.n_samples=1 \
  --set evaluation.enabled=false
```

Use dedicated config files for changes to `pipeline` stage ordering or large
nested model-specific structures.

## Adding a new config preset

1. Pick an existing config that is closest to the intended workflow.
2. Copy it and rename it using the naming convention.
3. Keep `model.family` consistent with an existing adapter.
4. Keep every stage name consistent with the stage registry.
5. Set `evaluation.enabled` and `export.enabled` intentionally.
6. Run a one-sample smoke test.
7. Add the new preset to documentation if it is a public workflow.

## Config checklist

Before committing a config:

- No secrets, personal paths, or machine-specific cache directories.
- `runtime.seed` is set.
- `model.family` is registered.
- Every `pipeline[*].name` is registered.
- The config has a clear purpose: smoke, PTQ-only, GPTQ+PTQ, benchmark, debug,
  or export.
- Expensive evaluation/export is disabled unless the config is explicitly a
  benchmark/export preset.
- A minimal smoke command has been tested.
