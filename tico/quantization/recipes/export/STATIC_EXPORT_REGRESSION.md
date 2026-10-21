# Static export isolation and regression verification

## Scope

This change isolates fixed-profile export input preparation from the dynamic
mask/RoPE helpers used by eager quality evaluation. It is an incremental change
on PR #1073 at `61f3a05d19c48e41fffc44b5e273a07b303bcaed`.

The regression baseline for the **original, pre-PR export behavior** is
`bb09ca852b0ca1b1a6271cfe1e10c162104c997e`, not `61f3a05`.
Comparing against `61f3a05` would only measure this follow-up change.

The implementation does not add an evaluation backend, a benchmark protocol, or
an NPU runtime. It does not recalibrate, change the weight bit widths, or reload
an FP model to create replacement quantized wrappers.

## Implementation contract

- Per-layer LLaMA, Qwen3-VL, and Gemma4 adapters require explicit host masks and
  `(cos, sin)` tensors. Their stage I/O and delta-KV contracts remain unchanged.
  Existing boolean keep-masks remain accepted where the wrapped model supports
  them. No eager fallback is used for missing host inputs.
- Full LLaMA and Qwen3-VL text export uses bounded tables created **before**
  `torch.export`. `prepared_static_export()` attaches them temporarily and
  restores the previous state on normal exit and exceptions. A full-model owner
  prepares shared tables once rather than allocating an S-by-S mask per layer.
- New LLaMA and Qwen3-VL wrappers keep separate rotary snapshots. Neither allocates native-context
  S-by-S tables at construction. Export never calls the mutable eager rotary
  instance to obtain its static tables.
- LLaMA per-layer export and `llama_host.npz` use the static preparation helper,
  not the eager `_slice_rope()` helper.
- Gemma4 keeps its existing static template path and explicit shared-KV/PLE
  adapters. Custom full-text static `position_ids` that would invoke live eager
  rotary generation are rejected; use explicit per-layer RoPE tensors instead.
- LLaMA and Qwen3-VL decoder export adapters request tuple results per call. They
  no longer modify the eager wrapper's persistent `return_type`.
- Quantized weights, observers, and frozen qparams are shared. Export preparation
  rejects active calibration. Do not run eager/export concurrently on the same
  model object while temporary static preparation is active.

### LLaMA RoPE unification is an intentional numerical correction

The old standalone decoder fallback ignored scaled RoPE when the rotary module
was owned by `LlamaModel`, not `LlamaDecoderLayer`. LLaMA export no longer
preserves that bug. Full-model and layer wrappers now use the same model-level
rotary source, including all scaling parameters. An independently wrapped layer
constructs the HF rotary implementation from its complete config; an optional
`rotary_emb=` constructor argument preserves customized model-level state.

Static preparation calls a fresh copy of the immutable source before tracing.
It does not run the live eager rotary module, and old tables cannot override the
unified source. Length-dependent rotary implementations use the selected export
profile. This does not make a fixed-capacity graph equivalent to every possible
history-dependent eager execution outside that profile.

Read [LLAMA_ROPE_UNIFICATION.md](LLAMA_ROPE_UNIFICATION.md) for checkpoint
migration, focused numerical verification, and how to interpret intentional
old/new differences. The old/new comparator still reports all mismatches as
FAIL; the correction is not an exception that disables regression checking.

### Full-model API

The recipe full-model exporter now accepts an explicit `max_seq_len`. For direct
full-model exports, use the preparation context and keep it active during
conversion:

```python
from tico.quantization.wrapq.utils.static_export import prepared_static_export

adapter = quantized_model.wrapped.as_export_module("prefill").eval()
with prepared_static_export(adapter, max_seq_len=2048):
    exported = tico.convert(adapter, (input_ids,), strict=False)
exported.save("model.q.circle")
```

An export adapter with external host masks/RoPE does not need this context. Direct
full-model tracing without preparation is rejected rather than tracing eager
mask/rotary construction. Qwen3-VL full-text preparation supports cache-free
prefill; external-KV deployment uses the existing per-layer adapters.

## Checkpoint requirements

Use a **trusted, frozen, full TICO `nn.Module` checkpoint** saved with
`torch.save(model, ...)`, not a `state_dict`. The worker uses
`torch.load(..., weights_only=False)`, which can execute pickle code. Do not use
untrusted checkpoints. Parameters must already be float32 for this CPU regression
workflow. The script refuses implicit casting and does not download models.

| Checkpoint provenance | Old/new `run` | Candidate-only `isolation` |
| --- | --- | --- |
| Pre-PR LLaMA with original tables | Use `--migrate-llama-rope` for the candidate; baseline remains unchanged | Same explicit migration required |
| LLaMA created with unified RoPE code | Old exporter may not load it; use a pre-PR checkpoint for old/new comparison | Supported directly |
| Intermediate LLaMA / previous isolation snapshot | Baseline compatibility depends on that checkout; do not fabricate old tables | Use `--migrate-llama-rope` |
| Pre-PR Qwen3-VL/Gemma4 | Existing supported legacy-table behavior is unchanged | Supported as before |
| Intermediate full-Qwen without tables or a snapshot | Not supported | Explicit provenance error |

A new checkpoint is not numerically interchangeable with an old one merely
because the recipe is the same. Do not independently recalibrate each checkout
and call that an export-only regression comparison. Use the same frozen
checkpoint file in both workers.

An old checkpoint's recorded context capacity remains authoritative. For example,
a checkpoint clamped to 2048 cannot be stressed with 4096 tokens. The script
rejects that request; it does not silently change `max_position_embeddings`.

## Dependencies and local tests

Run inside the existing TICO development environment with the appropriate
`transformers` version for the checkpoint, Torch, NumPy, YAML, `circle-schema`,
and the other project dependencies. The coordinator and both workers use the
same Python interpreter. Do not install one checkout over the other; each worker
imports source directly from its specified directory and verifies the location.

```bash
python -m unittest \
  test.quantization.wrapq.utils.test_static_export_inputs \
  test.quantization.recipes.test_export_regression
```

These tests include real Torch export of small helper/adapter models, immutable
rotary snapshots, cleanup/error paths, preserved eager return policy, frozen
qparam fingerprints, and negative tests for the report comparator. The Circle
structure unit tests use schema-shaped fixtures, not a FlatBuffer compiler.
The integration CLI below is what actually compiles and executes Circle.

## 1. Compare the original exporter and the patched exporter

Create a detached baseline checkout. Use a new results directory on every run:

```bash
git worktree add --detach ../TICO-export-baseline \
  bb09ca852b0ca1b1a6271cfe1e10c162104c997e

python tico/quantization/recipes/debug/export_regression.py run \
  --baseline-repo ../TICO-export-baseline \
  --candidate-repo . \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --checkpoint /absolute/path/to/pre_pr_frozen_llama.pt \
  --migrate-llama-rope \
  --seq-len 2048 \
  --eager-seq-len 1536 \
  --decode-steps 4 \
  --output /absolute/path/to/new-results/llama-old-new
```

The same checkpoint is loaded independently in each process. With
`--migrate-llama-rope`, only the candidate rebuilds canonical RoPE, and the runner
checks that all frozen weights/qparams remain unchanged. Older unscaled decoder
host inputs may now differ intentionally; those comparisons still return FAIL.
That result must be inspected alongside the HF parity tests and candidate-only
before/after-eager isolation checks. The candidate model
is then reused for eager execution and a second export. All three phases reset
the same random seed for export input generation. No calibration runs. The
script sets both the old `calibration.seq_len` and new `export.max_seq_len` to the
requested export length solely because the old exporter uses the former field.

`--artifacts auto` selects `circle_bundle` for LLaMA, ensuring the host RoPE arrays
are included, and `circle_per_layer` for Qwen3-VL/Gemma4. To also exercise the
full LLaMA model, add:

```bash
--artifacts circle_bundle,circle_full
```

For Qwen3-VL or Gemma4, omit `--migrate-llama-rope`, replace the config with `qwen3_vl_export.yaml` or
`gemma4_export.yaml` and provide that family's frozen checkpoint. Vision profile,
visual placement, attention profile, and PLE/shared-KV configuration must match
the checkpoint. The YAML does not rewrite a frozen checkpoint's quantization
profile. Use `--set key=value` to reproduce the actual deployment settings.

For an additional **text-only full-wrapper** probe on Qwen3-VL/Gemma4, add
`--include-full-text`. This extra probe is not a full multimodal product graph or
a new supported video-export workflow. Its failure is reported as an error;
it is never silently skipped.

## 2. Stress a native-context candidate checkpoint

For a new checkpoint with at least 4100 tokens of recorded capacity, test that a
4096-token eager prefill plus four decode steps does not affect a 2048-token
export:

```bash
python tico/quantization/recipes/debug/export_regression.py isolation \
  --candidate-repo . \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --checkpoint /absolute/path/to/new_frozen_llama.pt \
  --seq-len 2048 \
  --eager-seq-len 4096 \
  --decode-steps 4 \
  --output /absolute/path/to/new-results/llama-isolation
```

This mode explicitly records the baseline comparison as `NOT_RUN`. A PASS in
this mode proves only the requested before/after-eager checks, not equivalence
to the pre-PR exporter.

The default eager stress is text-only teacher forcing. For actual multimodal
inputs, save a tensor-only kwargs dictionary, or a list of such dictionaries,
using `torch.save` and pass `--eager-inputs /path/to/model_kwargs.pt`. Each dict is
called on the same candidate model in order. In that mode the file's calls
replace the automatic prefill/decode stress; `eager.json` records the mode and
input-file checksum. It does not automatically construct a cross-call KV cache.

## What the script compares

1. **Frozen state:** weights, quantization modes, observer buffers, cached scales,
   zero points, and related qparam state before/after export and eager execution.
2. **Circle structure:** graph inputs/outputs, shapes, dtypes, topology, builtin
   options, opcode versions, constants, quantization parameters, and signature
   definitions. Buffer-table reordering and internal tensor names are ignored;
   graph I/O names remain significant. This is conservative structural equality,
   not general graph-isomorphism or semantic-equivalence proof.
3. **Actual stage execution:** real `tico.convert` and real
   `CircleReferenceRuntime`, using native or fake-quantize execution as required.
   Export inputs are compared exactly. Circle outputs are compared with explicit
   tolerances (default `atol=1e-5`, `rtol=1e-4`). NaN/Inf is always a failure.
4. **Stage probes:** the real conversion sample plus additional decode
   empty/full-past and padded-prefill mask probes where applicable. All output
   tensors, including delta-KV, are compared. These are individual stage probes,
   not an end-to-end prefill/decode NPU generation test.
5. **Host artifacts:** actual saved NPZ/NPY arrays, tensor-only PT artifacts, JSON
   contracts, and other host files. Arrays and constants are compared exactly;
   NPZ timestamps are not treated as numerical changes.
6. **Torch diagnostics:** Torch Export operator lists and local Torch-vs-Circle
   output differences are recorded. Torch operator-list changes are diagnostic
   when final Circle structure and values match. Add `--require-torch-parity` to
   make local Torch-vs-Circle parity an additional hard requirement.

The runner intercepts real conversion/save functions only to record evidence; it
does not substitute a mock compiler, fake graph, or synthetic runtime result.
Every saved Circle must be captured and every compared stage must have numerical
output evidence. Missing dependencies, unsupported runtime operators, malformed
snapshots, or export failures are not counted as passing comparisons.

The CPU reference workflow can use substantial RAM and disk space: it records
weights' fingerprints, full stage input/output arrays, and three sets of model
artifacts. It does not measure latency, NPU memory, quality benchmark scores, or
bit-exact agreement with the target integer NPU.

## Results and exit codes

- Exit **0 / PASS**: all requested comparisons completed and passed.
- Exit **1 / FAIL**: completed comparisons found structural, numerical, host, or
  state changes. Inspect the detailed stage results; a mismatch is not by itself
  proof that the candidate is incorrect.
- Exit **2 / ERROR**: the requested verification could not be completed. Inspect
  the worker log and `error.json`; do not interpret missing evidence as a pass.

Argument errors also exit with code 2. Existing output directories are never
overwritten. `run` and `isolation` produce `summary.json`; worker failures include
`comparisons_complete: false`. If the coordinator itself fails before producing
that summary, inspect its `error.json` and nonzero exit status.

```text
results/
  summary.json
  baseline.log                       # run mode only
  candidate.log
  baseline/                          # run mode only
    environment.json
    quantization_state.json
    before/{artifacts,records,snapshot.json}
  candidate/
    environment.json
    quantization_state.json
    eager.json
    before/{artifacts,records,snapshot.json}
    after_eager/{artifacts,records,snapshot.json}
```

Recompare existing snapshots with different output tolerances without exporting:

```bash
python tico/quantization/recipes/debug/export_regression.py compare \
  /path/to/results/baseline/before \
  /path/to/results/candidate/before \
  --atol 1e-5 --rtol 1e-4 \
  --report /path/to/results/recomparison.json
```

Do not loosen tolerances to hide qparam, host-array, constant, or I/O changes;
those comparisons are intentionally exact and unaffected by numerical tolerances.
