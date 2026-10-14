# MMMU evaluation protocol and diagnostics

The Qwen3-VL and Gemma4 recipe adapters share the MMMU evaluator. These options
apply to both floating-point evaluation and checkpoint evaluation. They do not
change quantization algorithms or run an additional quantization stage.

## Supported evaluation scope

This evaluator supports **MMMU-Pro multiple-choice evaluation only**:
`dataset: MMMU/MMMU_Pro` with `vision`, `standard (4 options)`, or
`standard (10 options)`. The current strict/16 vision protocol is unchanged.

The original `MMMU/MMMU` mixes multiple-choice and open-ended questions and
is no longer an evaluation target here. Its request is rejected before dataset
loading, generation, or log creation. Qwen3-VL/Gemma4 recipe validation also
rejects it before model loading when MMMU is enabled or explicitly selected.
Set `evaluation.mmmu.dataset` explicitly; an omitted dataset is not silently
changed to another benchmark. A disabled/unselected MMMU target does not block
other evaluators.

There is no legacy open-ended scoring, automatic filtering, or implicit
multiple-choice subset. Open-ended rows in an otherwise supported configuration
(including few-shot examples) fail with their sample ID before generation;
they are not converted into wrong answers or skips. Low-level data-loading
helpers are not an assertion of scoring support for a dataset.

Use the separate LLaVA-Bench evaluation path for free-form descriptions or
conversational answers. Removing MMMU compatibility does not change that path,
VQAv2, COCO, Video-MME, or any quantization algorithm. It also does not add a
new free-form judge or claim official MMMU open-ended scoring support.

## Defaults and migration

For `dataset: MMMU/MMMU_Pro`, `subjects: [vision]`, the default prompt mode is
`strict_single_letter` with 16 generated tokens when `max_new_tokens` is
omitted or null. The two MMMU-Pro standard configurations retain the 16-token
default and their existing few-shot prompt path.

Per-mode defaults live in one place, `MMMU_PRO_VISION_DEFAULT_MAX_NEW_TOKENS`
in `tico/quantization/evaluation/mmmu_eval_utils.py`; the adapters only forward
the YAML values and never substitute their own budget:

| `prompt_mode` | Prompt | Default `max_new_tokens` |
|---|---|---|
| `strict_single_letter` (default) | `You are a strict single-letter answering machine. Answer the multiple-choice question shown in the image. Return only one letter from A to <last label>.` | 16 |
| `official_direct` | MMMU-Pro direct prompt (unchanged text) | 256 |
| `official_cot` | MMMU-Pro CoT prompt (unchanged text) | none; an explicit value is required |

An explicitly configured `max_new_tokens` always wins over the mode default.
Invalid values (booleans, fractions, strings, zero, negative numbers) are
rejected rather than converted. The shipped Qwen3-VL/Gemma4 evaluation presets
set `prompt_mode: strict_single_letter`, `max_new_tokens: 16` and zero shots for
vision (the image-only path ignores shots).

**Changing only `prompt_mode` in a preset that states `max_new_tokens: 16`
keeps the 16-token budget.** To evaluate the direct prompt with its own
default, change both values:

```bash
# official_direct with its 256-token default
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu \
  --set evaluation.mmmu.prompt_mode=official_direct \
  --set evaluation.mmmu.max_new_tokens=256

# equivalently, null selects the mode default
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu \
  --set evaluation.mmmu.prompt_mode=official_direct \
  --set evaluation.mmmu.max_new_tokens=null
```

`official_cot` uses the corresponding MMMU prompt but requires a non-null,
positive `max_new_tokens`. It does **not** automatically allocate 16,384 output
tokens. Choose a budget that fits the profile and specify it when changing
modes.

The official prompt strings are from `MMMU-Benchmark/MMMU`,
`mmmu-pro/prompts.yaml` and are not modified. The word `official` describes the
prompt only; it does not claim equivalence of preprocessing, model
implementation, generation configuration, sample selection or scoring to an
external benchmark, and using it is not a reproduction of the official
benchmark. The strict prompt is the historical TICO protocol restored as the
default; the measurements behind that choice cover two models and do not show
that strict/16 is optimal for every model.

### Choice-count correction versus the historical prompt

The strict prompt derives its last allowed label from the sample's actual
number of options. Ten options reproduce the historical `Return only one letter
from A to J.` wording exactly; four options give `A to D`, twelve options give
`A to L`, and one option gives `Return only the letter A.`. Earlier TICO
versions always wrote `A to J`, so scores on samples with a different option
count are not a bit-exact reproduction of the old default: the change is
"strict protocol restored plus allowed-label range corrected", not "old default
reproduced". Reproduce the exact historical prompt from the older revision when
that is required.

Only the prompt and the parser depend on the option count. The question,
option text and gold answer are never injected into the image-only prompt.

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
    prompt_mode: strict_single_letter
    max_new_tokens: 16
    input_max_seq_len: null
    output_jsonl: null
    temperature: 0.0
    verbose: false
```

| Option | Meaning |
|---|---|
| `prompt_mode` | `strict_single_letter` (default), `official_direct` or `official_cot`, for MMMU-Pro vision only. |
| `max_new_tokens` | Positive integer; explicit values override the mode default. `null` selects the mode default. |
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
tokens. The example sweeps the direct prompt, so `prompt_mode` is set
explicitly; without it the default strict prompt would be used.

```bash
python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu \
  --set evaluation.mmmu.n_samples=50 \
  --set evaluation.max_seq_len=2048 \
  --set evaluation.mmmu.prompt_mode=official_direct \
  --set evaluation.mmmu.input_max_seq_len=1536 \
  --set evaluation.mmmu.max_new_tokens=256 \
  --set evaluation.mmmu.output_jsonl=./out/mmmu_fp_256.jsonl

python -m tico.quantization.examples.evaluate \
  --config tico/quantization/examples/configs/qwen3_vl_eval_suite.yaml \
  --tasks mmmu \
  --set evaluation.mmmu.n_samples=50 \
  --set evaluation.max_seq_len=2048 \
  --set evaluation.mmmu.prompt_mode=official_direct \
  --set evaluation.mmmu.input_max_seq_len=1536 \
  --set evaluation.mmmu.max_new_tokens=512 \
  --set evaluation.mmmu.output_jsonl=./out/mmmu_fp_512.jsonl
```

For Gemma4, replace the config path with `gemma4_eval_suite.yaml`. No family name
is hard-coded in the budget or diagnostic helpers. A fixed cap is a necessary
control, not by itself proof that inputs match: compare per-sample tensor hashes
in the JSONL output. Preserve the resolved recipe configuration alongside logs;
logs do not embed model weights or a complete environment manifest.

## Multiple-choice scoring

Every evaluated sample has `scoring_policy: multiple_choice` and uses parser
`tico-mmmu-v4`. The dataset row's `question_type` may be absent or
`multiple-choice`. An explicit `open` or any unsupported type raises a
`ValueError` naming the sample, including before the multi-image skip check.
There is no fallback scorer. The multiple-choice parser grammar itself has not
changed as part of removing open-ended compatibility.

### Multiple-choice validation

Options are labelled `A`, `B`, ... in order. Samples with 1 to 26 options are
supported; the evaluator validates every multiple-choice sample's option list
and gold answer **before generation** and raises a `ValueError` naming the
sample ID for an empty option list, more than 26 options, a non-list option
structure, or an invalid gold. Such samples are neither skipped nor clamped to
a supported count, and the run ends with a `run_error` record. The
`mmmu_choice_labels(num_choices)` helper is the single source of the label set
for prompts, parsing and logging.

The gold contract is a single label: the gold must be a string whose
`strip().upper()` value is exactly one character inside the sample's labels.
`"c"` and `" C "` are both `C`; `""`, `" "`, `"AB"`, `"A,B"`, labels outside
the range, and non-string values (`None`, `1`, `True`) are rejected, never
converted. Validation, scoring and the `gold` log field all use the same
normalized value (`normalize_mmmu_gold_label`); the dataset string is kept
separately as `gold_raw`.

## Answer extraction

The multiple-choice parser version is `tico-mmmu-v4`. It is a versioned TICO
heuristic, not a claim of official scoring parity. Supported grammar, in
priority order:

1. The last explicit declaration `[final|correct] answer (:|=|is) X` outranks
   every other option mention. `X` is either a parenthesised label whose
   closing parenthesis ends the label (`Answer: (H)HENRI DE TOULOUSE-LAUTREC`
   gives `H`) or a bare letter followed by end of text, whitespace,
   punctuation, `)`, `/` or `-` (`Answer: HENRI` gives nothing). Markdown such
   as `**Answer:** C` is accepted.
2. A bare `X`, `(X)` or `X.` on the last non-empty line.
3. An unambiguous choice-only last line such as `I would choose B.`.
4. A leading `(X)` followed by anything (`(B)Yes.` gives `B`) or a leading
   `X.`/`X)`/`X:` followed by a space.

Rules:

- Candidates are restricted to the sample's actual option count. A label
  outside it (`J` with nine options, `Answer: Z` with twelve options) returns
  `None` with method `out_of_range`.
- A candidate followed by `or`, `and`, `/` or `-` (`Answer: C or D`,
  `Answer: (C)/(D)`, `Answer: C-D`, `(C) or (D)`) is ambiguous and returns
  `None` with method `ambiguous_declaration` or `ambiguous_leading`. When the
  last declaration is ambiguous or out of range, the parser does not fall back
  to an earlier declaration or to a bare/leading candidate (`Answer: B` then
  `Answer: C-D` is `None`; `Answer: A` then `Answer: Z` with twelve options is
  `None`). In `tico-mmmu-v3` a bare label followed by `-` was not recognised as
  a declaration at all, so `Answer: B` then `Answer: C-D` returned `B`; v4
  closes that gap. Existing v3 result files keep their recorded version.
- Prose is not guessed from: `I think this is ambiguous.`,
  `To answer a question ...`, `The answer is a circle.`, `Option A is
  incorrect.` all return `None`.
- A parenthesised label in the middle of the text is never taken as the answer;
  only the explicit-declaration, last-line and leading contexts apply.
- `None` is counted as an incorrect answer, not a skipped sample. The record
  distinguishes a response whose final answer was rejected
  (`ambiguous_declaration`, `ambiguous_leading`, `out_of_range`) from a
  response in which the parser found no candidate (`parse_method: unparsed`).
  `unparsed` is not a verdict that no final answer exists: the response may
  have been cut off before one (`stop_reason: length`), or it may state an
  answer in a format the grammar does not support. Read it together with
  `stop_reason` and the raw `generated` text before drawing conclusions.

Callers of `extract_answer(text)` that omit `num_choices` get the historical
ten-label default; the evaluator always passes the real count. Inspect saved
outputs when changing parsing policy and re-score both compared models with the
same parser version and the same prompt mode. Comparing an FP model and a
quantized model requires identical protocol settings and parser version; there
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

New logs use `schema_version: 2`: the obsolete `open_parser_version` field
has been removed. The multiple-choice parser version remains `tico-mmmu-v4`
because its scoring rules are unchanged. Existing logs are not rewritten.
`run_start` records the parser version, prompt mode and budgets;
`subject_start` records the resolved budget, input cap, effective prompt mode
and parser version. A successfully evaluated sample records:

- Sample ID, dataset/subject/split/index, prompt mode, the rendered prompt and
  the raw generated answer.
- `question_type` (as read from the row, `null` when absent),
  `scoring_policy: multiple_choice` and `parser_version`.
- `num_choices` and `valid_labels` (for example `ABCDEFGHIJKL`), normalized
  `gold`, original `gold_raw`, predicted answer, correctness, `parse_method`
  and `parse_failed`.
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

For supported multiple-choice rows the previous skip policy is preserved: multi-image samples,
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
