# Evaluation Helpers Developer Guide

`recipes/evaluation/` contains reusable evaluation helpers used by
`examples/evaluate.py`, optional evaluation inside `examples/quantize.py`, and
debug workflows.

Evaluation helpers are kept outside examples so that metrics can be shared
across FP models, fake-quant models, checkpoint evaluation, and regression tests.

## Standard quality protocol: FP eager vs Quant eager

LLaMA, Qwen3-VL, and Gemma4 use the same eager evaluation policy through
`eager.py`. The standard score measures the quality change after quantization.
Static graph parity, target NPU latency, memory, and throughput are separate
measurements with their own deployment profiles.

The three sequence settings have independent owners:

| Setting | Meaning | Default when omitted or null |
|---|---|---|
| `calibration.seq_len` | Calibration input budget | The calibration data builder's existing default |
| `evaluation.max_seq_len` | Common eager evaluation context budget | The source model's recorded native context |
| `export.max_seq_len` | Static prefill/decode capacity | `min(2048, model context)` |

Calibration never rewrites `config.max_position_embeddings`, the VLM text
configuration, or RoPE scaling. Eager quantized wrappers construct masks and
position embeddings for the actual request. LLaMA and Qwen3-VL do not allocate
native-context square masks at construction. Gemma4 retains bounded export
templates but builds full/sliding-window eager masks independently, including
the offsets of sliding KV caches.

`evaluation.backend` defaults to `eager`; other values fail validation. The
execution backend and a quantizer's attention `profile` are different settings.
An `npu_export` quantization profile can still run in eager mode, and evaluation
does not change its fused weights, arithmetic, observer placement, or qparams.

`evaluation.use_cache` defaults to `true` for all three adapters and controls
the inference default on both the model and generation configuration. A
task implementation can still explicitly disable unused cache for a scoring
forward. `evaluation.videomme.use_cache` remains an optional **results-cache
directory**, not the KV-cache setting.

Benchmark data must never update quantization statistics. Evaluation rejects
wrappers still in calibration mode or forced into the static export path.
Convert and save the calibrated model before using it as the Quant baseline.

### Compare the same protocol

Use one evaluation config for the FP and Quant runs, with the same model and
processor, task selection, dataset/split, sample selection, chat template,
few-shot count, image/video preprocessing, generation budget, stopping rules,
seed, and dtype. Change only the checkpoint and output directory:

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu,ppl \
  --output-dir ./out/comparison/fp

python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu,ppl \
  --checkpoint ./out/qwen3_vl_quantized/quantized_model.pt \
  --output-dir ./out/comparison/quant
```

The same command pattern applies to `llama_eval_suite.yaml` (for example,
`--tasks lm_eval`) and `gemma4_eval_suite.yaml`. `evaluate.py` never runs the
quantization pipeline in the config. The saved checkpoint must correspond to
the configured model/tokenizer/processor.

`--output-dir` controls the protocol record. Existing task-specific output and
results-cache paths remain explicit task settings. When evaluating LLaVA-Bench,
use separate candidate-answer/review/summary paths for FP and Quant, or enable
its `regenerate` option (enabled in the standard VLM evaluation presets). Do
not reuse FP candidate answers or an lmms results
cache as a Quant measurement.

Use an explicit common context cap when the native context is too expensive,
for example `--set evaluation.max_seq_len=8192` in **both** commands. Changing
this cap defines a different evaluation protocol; it does not change
calibration length or static artifact shapes. Report the cap with the score.
For VLM generation the budget includes visual/prompt tokens and the reserved
generation tokens. Video preprocessing uses this evaluation budget for FP and
Quant alike, never a quantized wrapper's static template capacity.

PPL windows can be configured separately with
`evaluation.perplexity.max_seq_len` for LLaMA or `evaluation.ppl.max_seq_len`
for VLMs. They default to the common evaluation context and must not exceed
it. The VLM evaluation-suite presets explicitly keep a 2048-token PPL window
with stride 512 while leaving generation tasks at native context. LLaMA
supports `evaluation.perplexity.stride`; omission retains its existing
non-overlapping window behavior. VLM `ppl.stride`, Gemma4's `chat-prefix` PPL,
and the existing task-specific prompt/scoring protocols are preserved. Scores
with different PPL windows, strides, or prompt modes are not the same protocol.

### Protocol record and checkpoint migration

Set `evaluation.output_dir`, or use `evaluate.py --output-dir`, to save
`evaluation_protocol.json`. The adapters also expose the same dictionary as
`ctx.artifacts["evaluation_protocol"]`. It records the execution purpose,
resolved context and independent calibration/export settings, cache policy,
configured/parameter dtypes, checkpoint path, tokenizer/processor identity and
chat-template digest, evaluation settings, and installed evaluator versions.
It is a record of the protocol prepared for execution, not a metrics file or
proof that a benchmark completed. Use separate directories for FP and Quant.

Older checkpoints may have a calibration-clamped positional configuration.
The evaluator retains the source model's native context before replacing it
with a checkpoint and rejects a requested/default context beyond the
checkpoint's recorded capacity. Regenerate such a checkpoint with this version,
or apply the same explicit smaller evaluation cap to FP and Quant. Editing a
checkpoint's capacity field is not a migration: it cannot establish that its
original positional semantics were preserved.

### Deployment validation

Keep small synthetic prefill/decode, RoPE/mask, padding, and KV-cache parity
tests alongside the eager quality evaluation. Choose static text capacity,
vision shape/frame profile, batch size, and cache policy when a deployment use
case is selected, then use the existing staged export/runtime workflows.
Export validation and fake-quant reference parity do not establish target NPU
latency or bit-exact integer execution. Benchmark those on the selected target
runtime separately; no external benchmark package changes are needed for the
eager protocol.

## Responsibilities

Evaluation helpers should:

- run benchmark-specific evaluation loops;
- compute and print metrics;
- hide benchmark-specific adapters or dataset plumbing;
- return structured results when practical.

Evaluation helpers should not:

- load the main model from Hugging Face;
- decide which model family is being evaluated;
- run quantization stages;
- save quantized model artifacts;
- parse CLI arguments directly.

The model adapter decides **which** evaluation helpers to call. The helper
decides **how** to evaluate a task.

## Top-level target selection

`recipes/evaluation/selection.py` defines the common selection contract used by
all model adapters and dataset preflight. Both use the same selector validation
and diagnostics.

`evaluation.selected_tasks` is either:

- absent or `null`, which preserves each adapter's existing enabled behavior;
- a list of canonical top-level target names, which becomes an exclusive
  allow-list;
- an empty list, which intentionally runs no evaluation targets.

The `evaluate.py --tasks` option populates this field. It selects evaluators,
not their benchmark-specific details. For example:

```text
lm_eval -> evaluation.lm_eval_tasks
vqa     -> evaluation.vlm_tasks
mmmu    -> evaluation.mmmu.*
ppl     -> evaluation.perplexity.* or evaluation.ppl.*
```

Each adapter declares `evaluation_targets`. A selected name that is not in that
registry must fail before model loading. Do not add aliases for config key names
or legacy spellings.

## Current layout

```text
recipes/evaluation/
├── README.md
├── eager.py             # common eager policy and protocol metadata
├── selection.py         # top-level target parsing and allow-list helpers
├── llm.py               # perplexity and text-only LM benchmark helpers
├── vlm.py               # VQA / COCO-style VLM benchmark helpers
├── llava_bench_judge.py # judge-based LLaVA-Bench evaluator wrapper
├── llava_bench.py       # shared llava_bench target selection and mode routing
└── mmlu.py              # MMLU wrapper helpers
```

The split is by benchmark/input type, not by model name.

These target-level entry points are exceptions to "the adapter decides which
helper to call", so that a shared selection policy is edited once rather than
once per model family:

- `llava_bench.py`: `evaluation.llava_bench` accepts a mapping, boolean, or
  null and routes to either the judge evaluator or the legacy COCO-style
  evaluator. VLM adapters call `run_llava_bench_evaluation()` at their
  LLaVA-Bench position.
- `vlm.py`: `run_vqa_evaluation()` decides whether the `vqa` target runs,
  normalizes and validates `evaluation.vlm_tasks`, and calls
  `evaluate_vqa_tasks()` plus `print_vqa_results()`. VLM adapters call it at
  their VQA position with the already-resolved evaluation arguments.
- `mmmu.py`: `run_mmmu_evaluation()` selects the `mmmu` target, normalizes its
  subject sentinel, and translates recipe settings into evaluator arguments.
  VLM adapters call it at their MMMU position; adapter preflight still validates
  the selected dataset before evaluation.

## Result conventions

Prefer returning structured values in addition to printing readable summaries.

Examples:

```python
float                         # perplexity
dict[str, tuple[int, int]]    # task -> (correct, total)
dict[str, float]              # metric -> score
```

Printing is acceptable for user-facing examples, but the returned value should
remain useful for tests.

## Function shape

Use explicit keyword-only arguments:

```python
def evaluate_perplexity(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    cache_dir: str | None,
    max_seq_len: int,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> float:
    ...
```

This avoids coupling the evaluation helper to the full recipe config. The
adapter should translate config fields into function arguments.

## Adding a new benchmark

1. Decide whether the benchmark belongs in an existing file or a new file.
2. Write a helper with explicit keyword-only arguments.
3. Keep dataset loading and benchmark adapters inside the helper.
4. Return structured results.
5. Add a compact print helper if the metric table is non-trivial.
6. Call the helper from the relevant model adapter’s `evaluate()` method.
7. Add config fields under the `evaluation:` section.
8. Add a smoke test with a tiny sample count.

Example config section:

```yaml
evaluation:
  enabled: true
  hellaswag:
    enabled: true
    n_samples: 32
```

Example adapter call:

```python
from tico.quantization.recipes.evaluation.selection import (
    get_mapping_evaluation_config,
    should_run_mapping_evaluation,
)

if should_run_mapping_evaluation(eval_cfg, "hellaswag"):
    hellaswag_cfg = get_mapping_evaluation_config(eval_cfg, "hellaswag")
    results = evaluate_hellaswag(
        model=ctx.model,
        tokenizer=ctx.tokenizer,
        device=str(ctx.device),
        n_samples=int(hellaswag_cfg.get("n_samples", -1)),
        max_seq_len=max_seq_len,
    )
```

## Config naming rules

Use task names that match common benchmark names:

```text
perplexity
lm_eval_tasks
vlm_tasks
coco
mmlu
hellaswag
arc
gsm8k
```

Avoid model-family names inside benchmark keys. The adapter already knows the
model family.

Good:

```yaml
evaluation:
  mmlu:
    enabled: true
```

Bad:

```yaml
evaluation:
  qwen3_vl_mmlu:
    enabled: true
```

## Side effects and logging

Evaluation helpers may print compact metric summaries. They should not print
large model outputs, full prompts, or dataset examples by default.

For verbose inspection, add an explicit argument:

```python
verbose: bool = False
```

## Device and dtype

Evaluation helpers should not move the full model between devices unless the
function contract explicitly says so. The adapter or caller owns model placement.

It is fine to move batches or tokenized inputs to `device` inside the helper.

## Dealing with external benchmark packages

Some benchmarks require optional packages. Keep those imports local and fail with
a clear message:

```python
def evaluate_some_task(...):
    try:
        import some_benchmark
    except ImportError as exc:
        raise RuntimeError(
            "some_benchmark is required for evaluation.some_task. "
            "Install the optional evaluation dependencies."
        ) from exc
```

This keeps basic quantization recipes importable without all benchmark
dependencies installed.

## Testing checklist

For a new evaluation helper, test:

- `n_samples=1` or the smallest supported sample count;
- CPU execution when possible;
- checkpoint evaluation through `examples/evaluate.py`;
- optional evaluation through `examples/quantize.py`;
- clear behavior when the benchmark dependency is missing;
- structured results that can be asserted in tests.
