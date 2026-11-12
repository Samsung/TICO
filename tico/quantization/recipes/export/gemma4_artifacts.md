# Gemma4 export artifact contracts

This guide owns the common dense Gemma4 E2B/E4B PLE, vision, cache, and
external-embedding export contracts. Both variants use `family: gemma4`, the
same wrappers, and the same artifact names inside separate output directories.
Text widths, layer counts, PLE shapes, and attention geometry come from the
loaded config and attention modules.

| Text geometry | E2B | E4B |
|---|---|---|
| Model | `google/gemma-4-E2B-it` | `google/gemma-4-E4B-it` |
| Hidden / intermediate size | 1536 / 6144 | 2560 / 10240 |
| Decoder layers | 35 | 42 |
| Layer pattern | 4 sliding + 1 full, repeated 7 times | 5 sliding + 1 full, repeated 7 times |
| Query / KV heads | 8 / 1 | 8 / 2 |
| Sliding / full head dimension | 256 / 512 | 256 / 512 |
| Shared-KV layers / first shared layer | 20 / 15 | 18 / 24 |
| Sliding / full KV producer | 13 / 14 | 22 / 23 |
| Double-wide shared MLP | enabled | disabled |
| PLE shape per token | `[35, 256]` | `[42, 256]` |
| Packed PLE width | 8960 | 10752 |
| Vocabulary / PLE vocabulary | 262144 / 262144 | 262144 / 262144 |
| Sliding window | 512 | 512 |

The shared static vision profile `gemma4_66x36_264` has identical geometry,
position IDs, and position-ID hash to the retained `e2b_66x36_264` name. The
legacy default and manifest name remain unchanged. Vision outputs project to
the text hidden width: `[264, 1536]` for E2B and `[264, 2560]` for E4B.

The concrete file tables and runtime flow below use **E2B example values**:
1536, 35, and `[35, 256]` are not universal contracts. E4B emits decoder
indices `0..41`, hidden states `[B, S, 2560]`, PLE `[B, S, 42, 256]`, and
cache tensors `[B, 2, cache_length, head_dim]`; `head_dim` is 256 for sliding
and 512 for full attention. Each shared consumer takes storage KV from the
producer of its own layer type.

The existing E2B quantize and export presets provide the examples below:

```bash
python -m tico.quantization.examples.quantize \
  --config tico/quantization/examples/configs/gemma4_quantize.yaml \
  --model google/gemma-4-e2b-it

python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/gemma4_export.yaml \
  --checkpoint ./out/gemma4_quantized/quantized_model.pt \
  --model google/gemma-4-e2b-it
```

The presets enable GPTQ followed by PTQ, with UINT4 linear weights, INT16
activations, UINT8 embeddings, `max_seq_len=2048`, vision profile
`e2b_66x36_264`, and `export.vision.granularity: both`. The shapes below use
that E2B configuration. File sizes depend on the checkpoint, quantization
policy, and serialization; inspect the generated artifacts for actual sizes.
The artifact tag `q` marks quantized graphs; a floating-point export uses
`f32` instead.

The E4B presets are `gemma4_e4b_quantize.yaml` and `gemma4_e4b_export.yaml`.
They select `google/gemma-4-E4B-it`, the common vision profile, and
`./out/gemma4_e4b_quantized` (with Circle files in its `circle_layers/`
subdirectory), keeping E2B artifacts separate. Eager quality evaluation uses
`gemma4_e4b_eval_suite.yaml`; static debug execution uses
`static_gemma4_e4b_runtime.yaml` through `inspector --mode static-gemma4-runtime`.

## Top level: `out/gemma4_quantized/` (E2B example)

| File | Description |
|---|---|
| `quantized_model.pt` | PTQ-wrapped model checkpoint after GPTQ and PTQ: float32 weights plus frozen observer qparams. Input of `export --checkpoint`. |
| `effective_config.yaml` | Resolved configuration of the quantize run (`gemma4_quantize.yaml` plus CLI overrides). |
| `circle_layers/` | Export output directory, described below. |

## `circle_layers/`: text pipeline (E2B example)

| File | Description |
|---|---|
| `token_embedding.q.circle` | Token embedding (scaled by sqrt(hidden)). Input `input_ids` `[1, S]` with dynamic `1 <= S <= 2048`, output `[1, S, 1536]`. Shared by prefill and decode, CPU. |
| `ple_embedding.q.pt` | Host-side PLE token lookup. UINT8 table `(262144, 8960)` with per-row scale/zero-point, `embed_scale`, and the four frozen observers replayed by the stage. The recipe's `auto` policy selects `.pt` because the estimated table size exceeds its 2 GiB threshold. Load with `Gemma4PLEEmbeddingHostTable.from_artifact`; output `[1, S, 35, 256]`. Shared by prefill and decode. |
| `ple_projection_prefill.q.circle` | PLE projection, norm, and combine stage for the NPU. Inputs `inputs_embeds` `[1, 2048, 1536]` and `per_layer_token_inputs` `[1, 2048, 35, 256]`, output `per_layer_inputs` `[1, 2048, 35, 256]`. |
| `ple_projection_decode.q.circle` | Same graph traced with `S = 1`. |
| `ple_pipeline.json` | PLE manifest: chosen `ple_embedding` format (`pt`) and size estimate, projection input/output shapes, and the qparams of the three boundary observers (`per_layer_token_inputs`, `per_layer_projection`, `per_layer_inputs`). |
| `multimodal_fusion_prefill.q.circle` | Fixed-slot fusion that writes the 264 visual embeddings into the text sequence starting at `visual_start_idx`. No weights. |
| `decoder_layer_prefill_0..34.q.circle` | The 35 text decoder layers at `S = 2048`. Inputs: hidden states, attention mask, RoPE, `per_layer_input` `[1, 2048, 256]` (plus shared K/V for shared-KV consumers); outputs hidden states (plus K/V). |
| `decoder_layer_decode_0..34.q.circle` | Single-token decode graphs of the same layers with 2047 past K/V slots as input. |
| `lm_head.q.circle` | Final RMSNorm plus LM head. Input `[1, 1, 1536]`, output logits `[1, 1, 262144]`. Final logit softcapping is applied on the host. |

## Per-Layer Embedding (PLE) contract

When the text config sets `hidden_size_per_layer_input > 0`, the Gemma4
`circle_per_layer` export adds three PLE stages alongside the existing
`token_embedding`, `multimodal_fusion_*`, decoder-layer, and `lm_head` graphs:

```text
ple_embedding.<tag>.circle            # shared CPU lookup, dynamic 1 <= S <= max_seq_len
ple_embedding.<tag>.pt                # ...or the host table selected by the recipe's auto policy
ple_projection_prefill.<tag>.circle   # NPU projection/combine, S = max_seq_len
ple_projection_decode.<tag>.circle    # NPU projection/combine, S = 1
ple_pipeline.json                     # chosen format, shapes, and observer boundaries
```

- `ple_embedding` takes `input_ids` `[1, S]` (int64) and returns
  `per_layer_token_inputs` `[1, S, num_hidden_layers, hidden_size_per_layer_input]`.
  It follows the same dynamic-sequence contract as `token_embedding`
  (`make_token_embedding_dynamic_shapes`) and is the only artifact that
  contains the large `embed_tokens_per_layer` table. Prefill runs it with
  `S = max_seq_len`, decode with `S = 1`.
- `export.ple_embedding_format` selects the lookup artifact: `auto` (default)
  retains the recipe's 2 GiB estimated-size threshold and writes
  `ple_embedding.<tag>.pt` at or above that threshold; `circle` and `pt`
  force one format. The E2B table (262144 x 35 x 256 elements) is 9.4 GB in
  float32 and 2.35 GB at 8-bit weights. The E4B table has
  262144 x 42 x 256 elements: its UINT8 payload is 2.625 GiB before qparams,
  frozen observers, and ancillary metadata. `auto` selects the host `.pt`
  artifact for both variants and for both `f32` and `q` tags. This format
  policy is separate from the serializer's
  appended-payload support: explicitly selecting `circle` can write larger
  tables through the shared serializer. See
  [recipe format policies](../../../../docs/large_circle_export.md#recipe-format-policies).
- The `.pt` artifact stores the complete stage contract: the float table (or
  the integer table with its per-row weight qparams), `embed_scale`, the packed
  geometry, and the frozen `embedding`, `embed_scale`, `act_out`, and
  `per_layer_token_inputs` observers. `Gemma4PLEEmbeddingHostTable.from_artifact`
  replays the stage on the host bit-exactly against the export adapter. The
  current host loader materializes the complete artifact in memory; this is
  not streaming or out-of-core execution. E4B reuses this external host-table
  contract rather than placing its entire PLE table in a single inline
  Circle FlatBuffer.
- `ple_projection_*` take `inputs_embeds` `[1, S, hidden_size]` (the
  `mm_fusion` output for prefill, the token embedding for decode) and
  `per_layer_token_inputs`, and return packed `per_layer_inputs`
  `[1, S, num_hidden_layers, hidden_size_per_layer_input]`. The runtime feeds
  decoder layer `i` the slice `per_layer_inputs[:, :, i, :]`.
- Observer ownership: `ple_embedding` output and `ple_projection_*` input use
  the text-model `per_layer_token_inputs` observer, the normalized projection
  uses `per_layer_projection`, and the packed output plus every decoder-layer
  `per_layer_input` boundary use `per_layer_inputs`.

With `export.prefill_decode: false` the exporter writes only the unsuffixed
prefill-shaped `ple_projection.<tag>.circle`. Models without PLE emit no PLE
artifacts.

Inside `ple_projection_*`, `per_layer_model_projection_scale` and
`per_layer_input_scale` observe the two multiplication constants, and
`per_layer_sum` observes the projection-plus-token sum before the final scale.
These internal observers follow the configured activation policy; the existing
`per_layer_inputs` observer still owns the final output/decoder boundary. Both
prefill and decode reuse the same frozen observers and scale buffers.

Checkpoints created before these internal PLE observers were added must be
prepared and PTQ-calibrated again before quantization and export. Re-exporting an
older `quantized_model.pt` alone cannot supply the missing sum statistics and is
rejected with a recalibration message. Adding the intermediate fake-quant boundary
can change quantized outputs, so rerun accuracy validation after recalibration.

## External-embedding text logits

`Gemma4TextLogitsPrefillExportAdapter` composes the existing static prefill
adapters for every text decoder layer, the final norm, LM head and the original
top-level logit softcapping. It accepts host-prepared `inputs_embeds`, optional
projected `per_layer_inputs`, and mask/RoPE dictionaries, and returns all token
logits. Token lookup, vision/fusion and PLE lookup/projection remain separate
host or staged operations. Shared-KV models use the existing per-layer staged
export rather than this composite adapter.

The adapter requires the actual external-embedding and cache paths to have been
calibrated before conversion. Its live capability/mode guards preserve frozen
source state; constructing or exporting it does not collect statistics. The two
Gemma4 top-level wrapper smoke cases use this text-logits scope, including the
conditional model's text-only route. They do not claim to export its vision
pipeline in the same artifact.

## KV cache boundaries

Gemma4 text attention has two independent KV quantization domains:

| Domain | Tensor and qparam owner |
|---|---|
| Storage S: `new_k`, `new_v` | Current projected KV written to persistent storage. The producer owns the cache input/output dtype, scale, zero-point, and granularity. |
| Compute C: `present_key`, `present_value` | KV used by this attention layer. Both Concat operands and its output use this grid. |

Current raw KV branches before storage quantization: one branch applies `new_*`
to produce the persistent storage delta, and another applies `present_*`
directly for attention. Current attention KV therefore avoids rounding through
the storage grid. Existing cache inputs are explicitly converted from S to C;
there is no intermediate read grid. `past_key` and `past_value` observer
overrides are rejected with migration guidance. Configure `new_*` for storage
and `present_*` for computation, then reprepare and recalibrate.

The static host copies/slices prefill and decode deltas without requantizing
them. Prefill KV outputs, decode cache inputs and deltas, and shared consumer
inputs therefore use the producer's frozen `new_*` qparams. Supported Circle
cache conversions use per-tensor UINT8/INT16 affine policies, or a consistently
unquantized cache path. An explicitly declared external float source can also
feed an affine compute boundary. MX, per-channel cache grids, and unsupported
affine-to-identity paths are rejected rather than serialized with mismatched
Concat qparams. Disabling fake quantization on an affine cache boundary is also
rejected; use an explicit supported identity policy for a float path.

The full prepared text model binds each shared consumer to the corresponding
KV-owning layer of the same layer type, keeping full and sliding sources
separate. For standalone shared attention or decoder use, call
`bind_cache_source(producer_attention)` before calibration. Export also accepts
`as_export_module(cache_source=producer_attention)` to supply or check the
binding, but first binding a calibrated consumer is rejected: reprepare and
bind before calibration. The source is a reference, not an owned child module,
so consumer calibration does not reset or recalibrate producer observers or
traverse its weights. The consumer converts S directly to its own C grid and
never mutates producer storage.

Gemma4 Assistant receives KV from the FP target used by its existing assisted
generation adapter. Its backbone declares this external float storage contract
with `bind_external_float_cache()` and IdentityObserver storage boundaries;
it does not invent a quantized producer. In PyTorch, external FP KV is
fake-quantized directly to the assistant's `present_*` compute grid. Circle
export uses the normal quantization folding rules, which can fold this
conversion into integer KV inputs carrying the compute qparams. The PyTorch
float storage declaration does not force a float Circle input ABI; callers
must use the serialized input dtypes and qparams.

`cache_output_mode="delta"` returns current storage KV. `"present"`/`"full"`
returns the updated HF cache object or the full **storage** tuple. The shared-KV
dictionary likewise carries storage tensors. Compute-grid tensors are never
written back into an HF cache, and `use_cache=False` never updates it.

The HF runtime path supports non-offloaded `DynamicCache` and `StaticCache`
full/sliding layers using their cache-layer APIs. Reads exclude unoccupied
allocation slots. Offloaded, quantized, and other cache layer implementations
are rejected explicitly.

Prefill-only calibration (`calibration.decode_steps: 0`) establishes both S and
C for ordinary prefill and decode export, including shared consumers observing
their producer's KV during prefill. Optional cached decode calibration remains
available for broader activation-statistics coverage. The quantize preset keeps
`decode_steps: 8`; an active PTQ stage's `decode_calibration_steps` still takes
precedence. Missing storage or compute qparams require calibration; no scales
are synthesized or copied between unrelated observers.

The common WrapQ frozen-state schema persists every observer's exact grid,
policy, provenance, lock and bypass state, together with wrapper mode and
capability readiness. Gemma cache metadata version 3 adds source bindings and
grid fingerprints; it does not duplicate observer grids. Recreate the same
prepared model/configuration before loading a state dict so shared source
bindings are rebuilt. A successful `restored.load_state_dict(...)` restores
quantized mode directly; no recalculation or second freeze is needed. Recipe
whole-module loads validate the same frozen contract before inference/export.
Legacy cache-only version 2 state cannot restore non-cache grids exactly and
is rejected even with `strict=False`, as are old three-domain state dicts and
serialized module checkpoints. Reprepare the original floating-point
model, remove obsolete `past_*` overrides, rerun PTQ calibration and conversion,
and save a new checkpoint before export. Existing past qparams are never silently
discarded or remapped. No automatic checkpoint migration is provided.

## RoPE split/concat domains

With the `npu_export` profile (`rope="pre_negated_sin"`), the half split,
reorder, and concatenation between `q_norm`/`k_norm` and the sine
multiplication preserve the normalized values. Each branch inherits the frozen
norm output grid when the norm output and the branch's `q_x1/q_x2/q_cat`
(`k_x1/k_x2/k_cat`) sites are all per-tensor affine and none of those sites
has an explicit override:

- Calibration collects nothing at the inherited sites; the norm output
  observer is collected once per forward at its original location.
- Quantized execution re-declares the frozen norm grid on the halves and the
  reordered concat. No grid is derived, copied, or modified, and the ordinary
  exporter folds the redundant markers, so the serialized norm output, halves,
  and RoPE concat carry one identical domain with no Quantize between them.
- Q and K are decided independently. Shared-KV consumers inherit only the Q
  branch; the K branch belongs to the producer. Domains are never shared
  between Q and K or across layers.
- cos/sin inputs, `q_cos/q_sin/q_rot` (`k_*`), attention logits, softmax,
  outputs, weights, and the S/C cache grids are unchanged.

`rope="hf"`, per-channel or MX norm outputs or sites, float (`no_quant`) norm
outputs, and explicitly overridden sites keep their independent observers.
Disabling fake quantization on an inheriting norm output is rejected at freeze
time instead of silently floating the branch. The decision is fixed at wrapper
construction and persisted in the attention extra state (`gemma_rope`); a
state dict or whole-module checkpoint produced under the other policy is
rejected with `strict=False` as well. Checkpoints calibrated before this
policy existed load only into wrappers that still resolve to independent
grids (for example `rope="hf"` or explicit site overrides); otherwise
reprepare and recalibrate. No automatic conversion between the two policies
is provided.

## `circle_layers/`: vision pipeline

Configure the layout under `export.vision`:

```yaml
export:
  artifacts:
    - circle_per_layer
  vision:
    granularity: both       # monolithic | layer | both
```

The default `monolithic` writes only the whole-tower `vision_prefill` graph.
`layer` writes the split pipeline; `both` writes both forms for numerical or
compiler comparisons. Each encoder layer is one Circle artifact. The table
below lists the E2B files in split-pipeline execution order after the monolithic
alternative.

| File | Description |
|---|---|
| `vision_prefill.q.circle` | Monolithic vision tower plus `embed_vision` projection. Input `pixel_values` `[1, 2520, 768]`, output visual tokens `[264, 1536]`. |
| `vision_patch_embedder.q.circle` | Split stage 1: patch embedding plus the baked positional embedding, output `[1, 2520, 768]`. |
| `vision_encoder_layer_00..15.q.circle` | Split stage 2: the 16 vision encoder layers. Each takes `hidden_states` and the three tensors stored in `vision_context.pt`. |
| `vision_pooler.q.circle` | Split stage 3: pools 2520 patches into 280 soft tokens. |
| `vision_post_projection.q.circle` | Split stage 4: selects the 264 valid tokens and applies `embed_vision`, output `[264, 1536]`. |
| `vision_context.pt` | Shared inputs of the split encoder layers: additive `attention_mask` `(1, 1, 2520, 2520)` and RoPE `cos`/`sin` `(1, 2520, 64)`. Stored once instead of being embedded in 16 graphs. |
| `vision_pipeline.json` | Split pipeline manifest: stage order, layer indices, tensor shapes, boundary observer/qparam contracts, and the `vision_context.pt` reference. |
| `vision_profile.json` | Static image profile `e2b_66x36_264`: 1056x576 image, 36x66 patch grid, 2520 patches (2376 valid plus 144 padding), 264 visual tokens, and the `image_position_ids` hash. |
| `effective_config.yaml` | Resolved configuration of the export run (`gemma4_export.yaml` plus `--checkpoint` and `--model`). |

If only the monolithic vision graph is used, the 21 split-vision files
(`vision_patch_embedder`, `vision_encoder_layer_*`, `vision_pooler`,
`vision_post_projection`, `vision_context.pt`, `vision_pipeline.json`) are not
needed; set `export.vision.granularity: monolithic` to skip them.

The raw attention mask and RoPE tensors in `vision_context.pt` are runtime
inputs of every encoder Circle. Producer and consumer stages use the same
frozen observer at each split boundary, so the generated quantized artifacts
are directly chainable.

### Static vision profile ownership

Constants generated for a static export profile have separate local observers
finalized through the common validator. Gemma4 pooler profiles own independent
small observers: activation sites install exact source frozen snapshots, and the
generated pooling matrix derives its own grid. Source observers and calibration
reservoirs are not copied or shared with another quantization owner.

All pooler observer execution policies are captured when constructing the profile.
In particular, generating a new matrix preserves the source's enabled/disabled
fake-quant policy without reusing another matrix's grid or lock. Rebuild a profile
to capture subsequent source-policy changes. Capability and captured-mode guards
continue to check the live source owner. Constructing or exporting a profile must
not change source snapshots; prepared vision models remain valid for refreeze and
checkpointing with equivalent export-profile topology.

Exporting another profile, including a failed export, also leaves previously
created profile adapters unchanged.

## Runtime data flow (E2B example)

```text
Prefill:
  token_embedding(input_ids)           -> text_embeds       [1, 2048, 1536]
  vision_prefill(pixel_values)         -> visual_embeds     [264, 1536]
  multimodal_fusion_prefill            -> inputs_embeds     [1, 2048, 1536]
  ple_embedding(input_ids)   (host)    -> per_layer_token_inputs [1, 2048, 35, 256]
  ple_projection_prefill               -> per_layer_inputs  [1, 2048, 35, 256]
  decoder_layer_prefill_0..34         (layer i gets per_layer_inputs[:, :, i, :])
  lm_head(last hidden)                 -> logits

Decode (one token):
  token_embedding(next_token)          -> inputs_embeds     [1, 1, 1536]
  ple_embedding(next_token)  (host)    -> per_layer_token_inputs [1, 1, 35, 256]
  ple_projection_decode                -> per_layer_inputs  [1, 1, 35, 256]
  decoder_layer_decode_0..34          (past K/V managed by the host)
  lm_head                              -> logits
```
