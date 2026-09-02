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
llama_quantize.yaml
llama_eval_suite.yaml
llama_export.yaml
qwen3_vl_quantize.yaml
qwen3_vl_eval_suite.yaml
qwen3_vl_eval_suite_mx_override_polices.yaml
qwen3_vl_export.yaml
gemma4_quantize.yaml
gemma4_eval_suite.yaml
gemma4_export.yaml
gemma4_e2b_assistant_quantize.yaml
gemma4_e2b_assistant_export.yaml
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
  dataset: Salesforce/wikitext
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

Qwen3-VL example:

```yaml
model_args:
  profile: npu_export
  vision:
    grid_thw: [8, 24, 24]
    visual_start_idx: 0
    spatial_merge_size: 2
```

Gemma4 static image-text example:

```yaml
model_args:
  vision:
    visual_start_idx: 0
    num_visual_tokens: 256
    image_height: 896
    image_width: 896
    validate_static_layout: false
```

Rules:

- Keep generic quantization parameters under `pipeline`, not `model_args`.
- Set both Gemma4 `image_height` and `image_width` when forcing a static
  calibration image size; setting only one is invalid.
- Enable Gemma4 `validate_static_layout` only for calibration inputs whose image
  placeholders follow the deployment layout.
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
    activation: int16
    linear_weight: uint4
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
    activation: int16
    linear_weight: uint4
```

PTQ-only smoke:

```yaml
pipeline:
  - name: ptq
    enabled: true
    activation: int16
    linear_weight: uint8
```

PTQ activation can also use an explicit spec mapping, for example MX:

```yaml
pipeline:
  - name: ptq
    enabled: true
    activation:
      kind: mx
      elem_format: fp8_e4m3
      axis: -1
    linear_weight: uint4
```

Weight-only PTQ keeps activations in the floating-point runtime dtype while
quantizing weights:

```yaml
runtime:
  dtype: float32

pipeline:
  - name: ptq
    enabled: true
    activation:
      kind: no_quant
    linear_weight: uint4
```

`no_quant` disables observer statistics and fake-quantization for the selected
role. It does not choose the floating-point execution dtype; `runtime.dtype`
does that. `activation: null` retains the normal default-resolution behavior and
must not be used as an alias for disabling activation quantization.

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

### GPTQ-to-PTQ weight qparam reuse

When GPTQ runs before PTQ, it updates the model weights and attaches the
per-module weight qparams used for that update. By default, PTQ reuses those
qparams in matching PTQ weight observers:

```yaml
pipeline:
  - name: gptq
    enabled: true
    weight_bits: 4
    weight_bits_overrides:
      model.visual.patch_embed.proj: 8

  - name: ptq
    enabled: true
    reuse_gptq_qparams: true  # default: true
    linear_weight: uint4
    vision_patch_embed_weight: uint8
```

Keep the GPTQ and PTQ weight policies compatible for every reused module,
including bit-width, signedness, qscheme, and channel granularity. The example
uses 8-bit asymmetric per-channel quantization for the Qwen3-VL vision patch
projection in both stages.

Set `reuse_gptq_qparams: false` to let PTQ weight observers compute qparams
from the current model weights. If GPTQ already ran, these are the GPTQ-updated
weights, not the original pre-GPTQ floating-point weights.

When reuse is enabled after an enabled GPTQ stage, PTQ fails if GPTQ metadata is
missing or if no PTQ weight observer accepts the attached qparams. Successful
reuse reports the number of matched PTQ weight observers.

## `evaluation`

LLM example:

```yaml
evaluation:
  enabled: true
  perplexity:
    dataset: Salesforce/wikitext
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

### Selecting top-level evaluation targets

`evaluation.selected_tasks` is an optional exclusive allow-list used by the
evaluation adapters. It is normally populated by the `evaluate.py --tasks`
option.

```yaml
evaluation:
  enabled: true
  selected_tasks:
    - mmmu
    - ppl
```

Selection rules:

- When `selected_tasks` is absent or `null`, existing per-target `enabled`
  values and truthy fields determine which evaluations run.
- When `selected_tasks` is a list, only the listed top-level targets run.
- A selected target runs even when its nested `enabled` value is `false`.
- Dataset names, subjects, sample counts, and other benchmark details still
  come from their existing YAML fields or `--set` overrides.
- An explicit empty list runs no evaluation targets.
- Target names are canonical and have no aliases.

Canonical targets:

```text
LLaMA:
  lm_eval
  ppl

Qwen3-VL / Gemma4:
  vqa
  coco
  llava_bench
  videomme
  mmlu
  hellaswag
  mmmu
  ppl
```

`lm_eval` requires a non-empty `evaluation.lm_eval_tasks` value. `vqa`
requires a non-empty `evaluation.vlm_tasks` value. Other selected targets may
use their evaluator defaults when their detail mapping is omitted.

CLI examples:

```bash
# Run only the configured LM-Eval benchmark list.
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/llama_eval_suite.yaml \
  --tasks lm_eval
```

```bash
# Run only MMMU and PPL; preserve their YAML detail settings.
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu,ppl
```

```bash
# Select the VQA evaluator and override its dataset list separately.
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks vqa \
  --set 'evaluation.vlm_tasks=[vqav2,textvqa]'
```

## `evaluation.llava_bench` for judge-based LLaVA-Bench scoring

`qwen3_vl_eval_suite.yaml` uses the standard example entry point:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml
```

It uses this schema under `evaluation`:

```yaml
evaluation:
  enabled: true
  llava_bench:
    enabled: true
    mode: judge
    dataset: lmms-lab/llava-bench-in-the-wild
    split: train
    n_samples: 50
    start_index: 0
    max_seq_len: 2048
    max_new_tokens: 512
    temperature: 0.0
    image_max_pixels: 802816  # 1024 * 28 * 28
    image_min_pixels: null
    resized_height: null
    resized_width: null
    visual_token_margin: 256
    skip_too_long: true
    candidate_label: Qwen/Qwen3-VL-4B-Instruct
    baseline_label: reference
    candidate_answers: null
    baseline_answers: null
    regenerate: false
    output:
      dir: ./out/llava_bench
      answers: ./out/llava_bench/qwen3_vl_4b.answers.jsonl
      reviews: ./out/llava_bench/qwen3_vl_4b.llama3_2_3b.reviews.jsonl
      summary: ./out/llava_bench/qwen3_vl_4b.llama3_2_3b.summary.json
    judge:
      enabled: true
      model_id: meta-llama/Llama-3.2-3B-Instruct
      device: cuda
      dtype: float16
      max_new_tokens: 256
      temperature: 0.0
      swap_order: true
```

Rules:

- Use the original LLaVA-Bench question for generation; do not add short-answer
  constraints.
- Use greedy decoding for reproducible generation and judging unless an
  experiment explicitly requires sampling.
- For `max_seq_len=2048`, keep `image_max_pixels` set to a bounded value such
  as `802816` (`1024 * 28 * 28`). Otherwise high-resolution images can expand
  beyond the static context before answer generation starts.
- Report the judge model ID with every result. Llama 3.2 3B is useful for
  inexpensive regression checks, not for official-score claims.
- Use `judge.swap_order: true` for FP-vs-quant or baseline-vs-candidate
  comparisons to reduce judge position bias.
- Set `candidate_answers` to reuse an existing answer JSONL instead of
  regenerating answers.
- Set `baseline_answers` when comparing a candidate answer file against an FP or
  runtime baseline. If `baseline_answers` is unset, the dataset reference answer
  is used as the baseline.

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
- `circle_per_layer` is a generic artifact key; the selected model adapter
  chooses the model-family-specific stage layout.
- Qwen3-VL Circle export requires `model_args.profile: npu_export`, and its
  fixed vision grid/start/merge values must match the wrapped checkpoint.

## CLI overrides

Simple values can be overridden with `--set`:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/llama_quantize.yaml \
  --set runtime.device=cpu \
  --set calibration.n_samples=1 \
  --set evaluation.enabled=false
```

Use dedicated config files for changes to `pipeline` stage ordering or large
nested model-specific structures.

## Adding a new config preset

1. Pick an existing config that is closest to the intended workflow.
2. Copy it and rename it using the naming convention.
3. Keep `model.family` consistent with an existing adapter unless the config is
   for a standalone example script.
4. Keep every stage name consistent with the stage registry.
5. Set `evaluation.enabled` and `export.enabled` intentionally.
6. Run a one-sample smoke test.
7. Add the new preset to documentation if it is a public workflow.

## Config checklist

Before committing a config:

- No secrets, personal paths, or machine-specific cache directories.
- `runtime.seed` is set when the config uses recipe runners.
- `model.family` is registered when the config uses recipe runners.
- Every `pipeline[*].name` is registered when the config has a pipeline.
- The config has a clear purpose: smoke, PTQ-only, GPTQ+PTQ, benchmark, debug,
  or export.
- Expensive evaluation/export is disabled unless the config is explicitly a
  benchmark/export preset.
- A minimal smoke command has been tested.
