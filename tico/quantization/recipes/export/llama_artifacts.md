# LLaMA artifacts and runtime contracts

This guide owns LLaMA Circle bundle contents, saved-artifact generation, rotary
sources, and checkpoint migration. For the shared exporter lifecycle and execution
preflight, see the [export developer guide](README.md). The
[CLI guide](../../examples/README.md#llama-benchmark-flow) provides the end-to-end
quantize, evaluate, and export workflow.

## Contents

- [LLaMA Circle bundles](#llama-circle-bundles)
- [Rotary source behavior](#rotary-source-behavior)
- [Capturing the rotary source](#capturing-the-rotary-source)
- [Migrating checkpoints without a canonical source](#migrating-checkpoints-without-a-canonical-source)
- [Offline focused verification](#offline-focused-verification)
- [Real-checkpoint export isolation](#real-checkpoint-export-isolation)
- [Two-revision diagnostic comparison](#two-revision-diagnostic-comparison)

## LLaMA Circle bundles

Opt in to `export.artifacts: [circle_bundle]` for a self-contained LLaMA
generation bundle. The LLaMA adapter writes it directly under
`export.output_dir`; the existing `circle_per_layer` artifact keeps its current
behavior. A bundle requires paired prefill/decode export and batch size 1.

```text
llama_manifest.json
llama_host.npz
token_embedding.<tag>.circle
decoder_layer_prefill_<index>.<tag>.circle
decoder_layer_decode_<index>.<tag>.circle
lm_head.<tag>.circle
tokenizer/                                  # when a tokenizer is available
```

`<tag>` is `f32` for float32 model export and `q` for a quantized checkpoint.
The versioned manifest records model geometry, sequence capacity, pad/EOS IDs,
stage input/output roles, the host-context file, and the optional tokenizer
directory. The host context preserves the prepared decoder layers' RoPE tables
and attention-mask fill values. The consumer uses these frozen values instead
of reconstructing them from a model name or a Transformers config.

`LlamaCircleRuntime` loads the saved Circle stages through the built-in reference
runtime. It owns causal masks, right-padding for static prefill, logical token
positions, and per-layer KV cache updates:

```python
import numpy as np

from tico.quantization.recipes.runtime.llama import LlamaCircleRuntime

runtime = LlamaCircleRuntime("./out/llama_bundle")
tokens = runtime.generate(
    np.asarray([[1, 2, 3]], dtype=np.int64),
    max_new_tokens=4,
)
```

Input IDs are one unpadded sequence with shape `[1, prompt_length]`. The result
has shape `[1, prompt_length + generated_length]`, including EOS when emitted.
Generation is greedy and uses the manifest's EOS ID or IDs by default. The
requested prompt length plus `max_new_tokens` must not exceed `max_seq_len`;
capacity errors do not silently truncate the prompt or reduce the requested
steps. Generation starts a fresh sequence when at least one token is requested;
a zero-token request returns the prompt without changing the current cache.

`runtime.execution_mode` is `native` for floating-point artifacts or
`fake_quantize` for quantized artifacts. Fake-quantize reference execution is a
semantic check, not bit-exact validation of an integer target backend. The
runtime itself does not import Transformers, load a PyTorch model, or fetch
files. The optional [generation CLI](../../examples/README.md#generate-from-a-llama-circle-bundle)
uses only local tokenizer files for prompt encoding/continuation decoding.

Tiny deterministic random-model tests exercise floating-point saved-artifact
parity, quantized tensor domains, and cache composition. Quantized reference
execution does not promise PyTorch fake-quantization parity: a directly quantized
embedding carries independent table/output qparams, while reference GATHER
forwards table values without applying the output grid. See the
[quantized Circle contract and reference limitation](../../../../docs/design.md#102-conversion-time-quantized-graph-legalization).
Meaningful language output requires pretrained weights; model quality and
target-device correctness and performance need their own evaluation.

## Rotary source behavior

There is one rotary source: the original model-level rotary implementation,
including scaling and any explicitly supplied custom frequency state.

- `QuantLlamaModel` captures this source and propagates independent copies to its
  decoder wrappers. Both eager and export copies use the same source.
- A standalone `QuantLlamaDecoderLayer` without a model-level rotary module uses
  `LlamaRotaryEmbedding(config=...)`, not an unscaled theta-only fallback.
- For custom rotary state not represented in config, explicitly pass the original
  module: `QuantLlamaDecoderLayer(fp_layer, qcfg=qcfg, rotary_emb=fp_model.rotary_emb)`.
- Static preparation evaluates a copy of the immutable rotary snapshot for the
  requested profile before tracing. It never calls mutable eager rotary state.
- The `hf` / `pre_negated_sin` representation remains controlled by the existing
  attention profile. Compare both paths with the same profile.

Migration from a legacy unscaled-decoder checkpoint targets the rotary source;
it does not intentionally alter quantized parameters, observers, frozen qparams,
graph I/O, or delta-KV buffer contracts. Corrected cos/sin inputs and resulting
inference outputs **can differ from the legacy export**.

That migration corrects the legacy numerical behavior and does not promise
byte-equivalent artifacts, pretrained-model accuracy, or final Circle
equivalence. Length-dependent RoPE is evaluated at the selected export capacity;
comparisons must use that same profile rather than an unrelated eager
history/length.

## Capturing the rotary source

Construct and calibrate wrappers from the matching HF model. No additional YAML
option is required. Rotary snapshots are saved in full-model pickle checkpoints.
A normal state-dict load into newly constructed wrappers gets the source from the
constructor; the source checkpoint/config must still match.

## Migrating checkpoints without a canonical source

Legacy full-model pickle checkpoints can retain unscaled standalone layer tables
or a rotary snapshot without canonical model-level provenance. Export refuses
these sources with an explicit migration error instead of silently using them.
Reconstructing and calibrating the model with the matching rotary source is
preferred for final quality measurements.

To reuse a trusted frozen checkpoint for diagnosis without recalibration:

```python
import torch
from tico.quantization.wrapq.wrappers.llama.static_inputs import migrate_llama_rope_state

model = torch.load("frozen.pt", map_location="cpu", weights_only=False)
report = migrate_llama_rope_state(model)
# Optionally save to a NEW path; the original is not overwritten by migration.
torch.save(model, "frozen.unified-rope.pt")
```

Only load trusted files: `weights_only=False` permits pickle code execution.
The migration rebuilds from the recorded complete config, not the live eager
rotary module. Pass `rotary_emb=original_fp_model.rotary_emb` when the original
model has customized, non-configurable rotary state. It does not silently enlarge
`max_position_embeddings`, reload FP weights, or recompute quantization parameters.
Discard external KV caches created before migration. If calibration originally
used the wrong standalone RoPE, preserving those qparams does not guarantee they
remain optimal; re-evaluate and consider recalibration using the corrected path.

## Offline focused verification

No model/data downloads are required. This command uses tiny randomly initialized
HF LLaMA models with default, linear, and llama3 scaling. It compares the HF
source, full wrapper, independent layer, and full model's child layers under both
sine conventions. It also checks padded prefill, two delta-KV decode steps,
source isolation, checkpoint serialization, migration qparams, and Torch Export.

```bash
python tico/quantization/recipes/debug/verify_llama_rope.py \
  --repo . --report ./out/llama-rope-checks.json
```

The script requires the complete TICO development environment and test sources.
Exit codes: `0=PASS`, `1=assertion failure`, `2=missing/incomplete/error`. A skipped
or unimportable test does not count as success. This is not a real-model Circle
or NPU validation command.

## Real-checkpoint export isolation

For a checkpoint that lacks a canonical rotary source, opt in to migration:

```bash
python tico/quantization/recipes/debug/export_regression.py isolation \
  --candidate-repo . \
  --config tico/quantization/examples/configs/llama_export.yaml \
  --checkpoint /absolute/path/to/frozen.pt \
  --migrate-llama-rope \
  --seq-len 2048 --eager-seq-len 1536 --decode-steps 4 \
  --output /absolute/path/to/new-results/llama-unified-isolation
```

For a checkpoint with a valid canonical source, omit `--migrate-llama-rope`. Use
longer eager stress only if the checkpoint's recorded capacity allows it. When
migration is requested, the runner checks frozen state before and after it, then
exports before and after eager stress. `environment.json` records migration
provenance.

## Two-revision diagnostic comparison

Select a baseline revision and a config matching a trusted frozen checkpoint that
both checkouts can load. Record the resolved source revisions with the result.
The following example opts in to candidate-only migration for a legacy rotary
source; omit that option when both workers can use its canonical source directly:

```bash
BASELINE_REF=your-recorded-baseline-ref
EXPORT_CONFIG=/absolute/path/to/matching_llama_export.yaml
FROZEN_CHECKPOINT=/absolute/path/to/frozen_llama.pt
git worktree add --detach ../TICO-export-baseline "$BASELINE_REF"
python tico/quantization/recipes/debug/export_regression.py run \
  --baseline-repo ../TICO-export-baseline --candidate-repo . \
  --config "$EXPORT_CONFIG" \
  --checkpoint "$FROZEN_CHECKPOINT" \
  --migrate-llama-rope \
  --seq-len 2048 --eager-seq-len 1536 --decode-steps 4 \
  --output /absolute/path/to/new-results/llama-comparison
```

If the selected baseline reads static capacity from `calibration.seq_len`, add
`--legacy-calibration-length`. This explicit bridge is unnecessary for exporters
that read `export.max_seq_len`. It does not recalibrate the checkpoint or verify
calibration/export length independence.

The baseline remains unchanged. Candidate-only migration preserves frozen
weights/qparams, but corrected RoPE may change host arrays, conversion probe
inputs, and outputs. Those differences still produce `FAIL` (exit 1); there is
no broad "expected difference" exemption. Review them alongside the focused HF
parity checks and candidate before/after isolation. Structural changes or other
unexpected differences still require investigation. Do not loosen tolerances
simply to hide the correction.

See [static export regression guide](../debug/static_export_regression.md) for detailed
artifact comparison semantics and resource requirements. The migration option
is LLaMA-only; Qwen3-VL and Gemma4 have their own checkpoint and cache-readiness
requirements.
