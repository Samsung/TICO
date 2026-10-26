# Unified LLaMA RoPE for eager and static export

## Behavior

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
- Quantized parameters, observers, frozen qparams, graph I/O, and delta-KV buffer
  contracts are not intentionally changed. Corrected cos/sin inputs and the
  resulting inference outputs **can differ from the old unscaled decoder**.

This corrects a numerical bug; it is not a byte-preserving refactor. It does not
prove pretrained-model accuracy or final Circle equivalence. Length-dependent
RoPE is evaluated at the selected export capacity; comparisons must use that
same profile rather than an unrelated eager history/length.

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

## 1. Offline focused verification

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

## 2. Real-checkpoint export isolation

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

## 3. Two-revision diagnostic comparison

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

See [STATIC_EXPORT_REGRESSION.md](STATIC_EXPORT_REGRESSION.md) for detailed
artifact comparison semantics and resource requirements. The migration option
is LLaMA-only; Qwen3-VL and Gemma4 have their own checkpoint and cache-readiness
requirements.
