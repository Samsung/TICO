# MMMU evaluation protocol and diagnostics

The Qwen3-VL and Gemma4 recipe adapters share the MMMU evaluator. These options
apply to both floating-point evaluation and checkpoint evaluation. They do not
change quantization algorithms or run an additional quantization stage.

## Defaults and migration

For `dataset: MMMU/MMMU_Pro`, `subjects: [vision]`, the default prompt is
`official_direct`, with 256 generated tokens when `max_new_tokens` is omitted or
null. An explicitly configured token count wins. Non-vision subjects retain the
16-token default and their existing few-shot prompt path.

The shipped Qwen3-VL/Gemma4 evaluation presets explicitly set direct mode, 256
output tokens, and zero shots for vision (the image-only path ignores shots).
The old strict-single-letter prompt is replaced, so do not compare an old
baseline score directly against a newly evaluated quantized model. Re-evaluate
both models under the same settings.

`official_cot` uses the corresponding MMMU prompt but requires a non-null,
positive `max_new_tokens`. It does **not** automatically allocate 16,384 output
tokens. Choose a budget that fits the profile and specify it when changing
modes. Changing only `prompt_mode` in a YAML file with `max_new_tokens: 256`
retains that explicit 256-token value; it does not select a long CoT preset.

The prompt strings are from `MMMU-Benchmark/MMMU`, `mmmu-pro/prompts.yaml`.
The word `official` describes the prompt, not equivalence of preprocessing,
model implementation, generation configuration, sample selection, or scoring to
an external benchmark. The parser version is `tico-mmmu-v2`.

## Configuration

```yaml
evaluation:
  max_seq_len: 2048  # Total input-plus-output capacity.
  mmmu:
    enabled: true
    dataset: MMMU/MMMU_Pro
    subjects: [vision]
    n_shots: 0
    n_samples: 1000
    prompt_mode: official_direct
    max_new_tokens: 256
    input_max_seq_len: null
    output_jsonl: null
    temperature: 0.0
    verbose: false
```

| Option | Meaning |
|---|---|
| `prompt_mode` | `official_direct` or `official_cot`, for MMMU-Pro vision only. |
| `max_new_tokens` | Positive integer; explicit values override defaults. |
| `input_max_seq_len` | Optional positive, fixed input cap, including text and image tokens. |
| `output_jsonl` | Optional new output file. Existing files raise `FileExistsError`. |

With a null input cap, existing automatic budgeting is preserved:
`input_cap = max_seq_len - max_new_tokens`. The Qwen-style processor uses that
cap to compute its image pixel budget, so increasing output capacity can reduce
image resolution. Other processors retain their model-specific image settings;
the final processed input length is checked rather than forcibly truncating
multimodal tokens.

A fixed input cap is never silently shrunk. When the total capacity is specified,
`input_max_seq_len + max_new_tokens <= max_seq_len` is required. Incompatible
settings fail before dataset loading or creation of a diagnostic log. A null
total capacity means there is no evaluator-imposed total limit; the caller must
still respect the model/runtime's actual context limit.

## Fixed-input output-budget comparison

Run from the repository root. Use the same model, processor, dataset revision,
sample selection, prompt, seed, dtype and image settings for both runs. For a
2048-token profile, a 1536-token input cap accommodates both 256 and 512 output
tokens.

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu \
  --set evaluation.mmmu.n_samples=50 \
  --set evaluation.max_seq_len=2048 \
  --set evaluation.mmmu.input_max_seq_len=1536 \
  --set evaluation.mmmu.max_new_tokens=256 \
  --set evaluation.mmmu.output_jsonl=./out/mmmu_fp_256.jsonl

python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu \
  --set evaluation.mmmu.n_samples=50 \
  --set evaluation.max_seq_len=2048 \
  --set evaluation.mmmu.input_max_seq_len=1536 \
  --set evaluation.mmmu.max_new_tokens=512 \
  --set evaluation.mmmu.output_jsonl=./out/mmmu_fp_512.jsonl
```

For Gemma4, replace the config path with `gemma4_eval_suite.yaml`. No family name
is hard-coded in the budget or diagnostic helpers. A fixed cap is a necessary
control, not by itself proof that inputs match: compare per-sample tensor hashes
in the JSONL output. Preserve the resolved recipe configuration alongside logs;
logs do not embed model weights or a complete environment manifest.

## Answer extraction

The last explicit answer declaration (`Answer: C`, `The answer is C`, etc.)
outranks general option mentions. Markdown such as `**Answer:** C` is accepted.
Otherwise the parser accepts a bare answer on the final non-empty line, an
unambiguous choice-only final line, or a leading capitalized option with
answer-like punctuation. Prose such as `I think ...`, `To answer a question ...`
and a trailing `Option A is incorrect.` is not treated as a final answer.

Candidates are restricted to the actual number of choices, up to A-J. Explicitly
ambiguous or unparseable output returns `None`. It counts as an incorrect answer,
not a skipped sample. Parsing remains heuristic; inspect saved outputs when
changing parsing policy and re-score both compared models consistently. There
is no automatic official-score claim or random-guess fallback.

## Diagnostic record contract

`output_jsonl: null` disables diagnostic tensor hashing, generated-token copies
and log I/O. Enabling it has additional CPU-copy and hashing overhead; do not use
those timings as a clean throughput benchmark.

Records are flushed after every write. A file contains `run_start`, a
`subject_start` for each subject, `sample` records, `subject_summary` records,
and a final `run_summary`. Unexpected exceptions emit `run_error` and propagate.
An interrupted or failed file can be inspected but is not automatically resumed.
A file without `run_summary` must not be treated as a completed run. Choose a
fresh path for each run; existing files are not overwritten or appended to.

A successfully evaluated sample records:

- Sample ID, dataset/subject/split/index, prompt and raw generated answer.
- Gold, predicted answer, correctness, parser method and parse-failure status.
- Processed input length/cap, image/video grid when provided, per-tensor shape,
  dtype and SHA-256, and a combined `tensor_inputs_sha256` fingerprint.
- Generated token IDs/count, configured EOS IDs, observed EOS, budget-limit hit,
  inferred `stop_reason` and `length_limited` status.

Tensor fingerprints include exact bytes, tensor names, shapes and dtypes, before
generation. They exclude model weights and non-tensor processor outputs. This
supports equality checks for the numeric inputs, not complete semantic equality
of different model implementations.

EOS is inferred from `model.generation_config` (or the model config when no
generation config exists), not the tokenizer's potentially different EOS. EOS
at the last allowed token is not classified as truncation. Padding after an
observed EOS is excluded from the count. Without usable EOS configuration,
`stop_reason` is `unknown` and `length_limited` is null. For configured EOS,
a limit hit without EOS is reported as length-limited; this is an observation,
not the generation engine's authoritative stopping-criteria trace. Custom stop
rules can require further investigation.

The evaluator preserves the previous skip policy: multi-image samples,
image-token-count mismatch errors, and runtime errors are skipped. It does not
silently convert other errors into skips. Each skip has an ID and reason in the
log. Compare the evaluated ID sets before interpreting FP-versus-quantized
accuracy. If coverage differs, report it and compare the common evaluated set
rather than treating different denominators as a clean quantization delta.

Subject summaries report evaluated/skipped counts, parse failures/rate, EOS
count, length-limited count and unknown-stop count. Accuracy is null in the
JSONL summary for an empty evaluated set; the legacy console table displays
0.0000 without dividing by zero. The public return value remains
`{subject: (correct, total, skipped)}`.

## Validation

Focused repository tests:

```bash
python -m unittest discover -s test/quantization/recipes -p 'test_mmmu_protocol.py'
python -m unittest discover -s test/quantization/recipes -p 'test_vlm_generation_diagnostics.py'
```

Also run the existing VLM evaluation/adapter tests and repository formatting
checks in the configured development environment. These synthetic tests do not
validate a full model, GPU execution, task accuracy, speed, or official benchmark
reproduction.
