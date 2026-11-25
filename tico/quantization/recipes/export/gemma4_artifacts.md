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
| `text_pipeline.json` | Versioned decoder/LM-head manifest: static dimensions, producer indices, suffix boundary, and the single-token finalize capability/aliases. The loader validates shapes, dtypes and boundary grids from the referenced Circle files. |
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
    granularity: encoder    # monolithic | encoder | layer | both
    spatial_pooling: true   # default; applies to e2b_66x36_264 and its geometry alias
```

With `spatial_pooling: true`, the 36×66 row-major patch grid is grouped into
264 windows of nine patches and exported with a 1×9 AveragePool at stride one.
The pooler retains its 280 output slots, frozen activation boundaries and final
scaling. The default is `true`; set `false` to keep the pooling MatMul. Other
layouts and quantization policies that require the pooling matrix continue to
use MatMul. The option applies to all four granularities, including the separate
pooler graph in `encoder` mode. Integer rounding can differ between target
kernels.

| Granularity | Vision graphs |
|---|---|
| `monolithic` | One graph containing patch embedding, all encoder layers, pooling, and post projection. |
| `encoder` | Four graphs: patch embedder, the complete encoder stack, pooler, and post projection. |
| `layer` | Patch embedder, one graph per encoder layer, pooler, and post projection. |
| `both` | The `monolithic` and `layer` forms together. It does not add the `encoder` graph. |

The Python API default remains `monolithic`; the E2B/E4B export YAML presets
retain `both`. Select `encoder` through the existing generic override:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/gemma4_export.yaml \
  --checkpoint ./out/gemma4_quantized/quantized_model.pt \
  --set export.vision.granularity=encoder \
  --output-dir ./out/gemma4_quantized/encoder_circle_layers
```

For a floating-point export, replace `--checkpoint ...` with `--source model`;
the vision files then use the `f32` tag. Use `gemma4_e4b_export.yaml` with the
E4B checkpoint for the same layout on E4B. Encoder layer count and hidden width
come from the loaded model rather than the E2B example values below.

| File | Description |
|---|---|
| `vision_prefill.q.circle` | Monolithic vision tower plus `embed_vision` projection. Input `pixel_values` `[1, 2520, 768]`, output visual tokens `[264, 1536]`. |
| `vision_patch_embedder.q.circle` | Split stage 1: patch embedding plus the baked positional embedding, output `[1, 2520, 768]`. |
| `vision_encoder.q.circle` | `encoder` stage 2: all encoder layers in model order in one graph. Inputs are `hidden_states`, `attention_mask`, `position_embeddings_cos`, `position_embeddings_sin`; output `hidden_states` `[1, 2520, 768]`. |
| `vision_encoder_layer_00..15.q.circle` | Split stage 2: the 16 vision encoder layers. Each takes `hidden_states` and the three tensors stored in `vision_context.pt`. |
| `vision_pooler.q.circle` | Split stage 3: pools 2520 patches into 280 soft tokens. |
| `vision_post_projection.q.circle` | Split stage 4: selects the 264 valid tokens and applies `embed_vision`, output `[264, 1536]`. |
| `vision_context.pt` | Raw external context for the encoder stack or each split encoder layer: additive `attention_mask` `(1, 1, 2520, 2520)` and RoPE `cos`/`sin` `(1, 2520, 64)`. Stored once instead of embedded in encoder graphs. |
| `vision_pipeline.json` | Split pipeline manifest: granularity, stage order, layer metadata, tensor shapes/dtypes, boundary observer/qparam contracts, and the `vision_context.pt` reference. |
| `vision_profile.json` | Static image profile `e2b_66x36_264`: 1056x576 image, 36x66 patch grid, 2520 patches (2376 valid plus 144 padding), 264 visual tokens, and the `image_position_ids` hash. |
| `effective_config.yaml` | Resolved configuration of the export run (`gemma4_export.yaml` plus `--checkpoint` and `--model`). |

`encoder` writes exactly the four vision Circle files `vision_patch_embedder`,
`vision_encoder`, `vision_pooler`, and `vision_post_projection`, plus the context
and manifest. It does not write `vision_prefill` or `vision_encoder_layer_*`.
`monolithic` skips all split files. Use a separate output directory when
changing granularity so artifacts from an earlier export do not remain beside
the new files.

The four-stage execution order is:

```text
vision_patch_embedder(pixel_values)
  -> hidden_states
vision_encoder(hidden_states, attention_mask, position_embeddings_cos, position_embeddings_sin)
  -> hidden_states
vision_pooler(hidden_states)
  -> pooled_hidden_states
vision_post_projection(pooled_hidden_states)
  -> visual_embeds
```

Load the context with `torch.load("vision_context.pt", weights_only=True)`.
Eager adapters take its three raw tensors in the encoder input order shown
above. For a
static profile with `S` patches, vision hidden width `H`, and attention head
dimension `D`, encoder input/output hidden states have shape `[1, S, H]`, the
mask `[1, 1, S, S]`, and each RoPE tensor `[1, S, D]`. Context dtypes follow the
prepared encoder compute dtype and are recorded in `shared_encoder_inputs`;
use the tensors for the matching `vision_profile.json`. For a serialized Circle,
bind context according to its actual input dtype/qparams: float inputs use the
raw tensors, while affine integer inputs require host quantization
(`clamp(round(value / scale) + zero_point)` to the input integer range), rather
than an integer cast. Quantize context once and reuse it for that profile.

The raw attention mask and RoPE tensors in `vision_context.pt` are runtime
inputs of the encoder Circle. Quantized execution applies the encoder-owned
context observers once before running the complete stack. Each layer retains
its own quantization operations without extra boundaries between internal
layers. The patch stage ends in the encoder input domain; the encoder applies
its input observer, all layers, its output observer, and then the pooler input
observer. The pooler input therefore uses the same frozen domain as the encoder
output. The pooler/post-projection boundary likewise reuses the pooler output
observer. These boundary declarations are bypassed in `NO_QUANT` mode.

The `encoder` manifest has stage IDs `patch_embedder`, `encoder`, `pooler`, and
`post_projection`, and exactly three artifact boundaries connecting adjacent
IDs. `num_layers` describes the entire encoder stack; internal layer edges
are not artifact boundaries. The profile, artifact tag, and context reference
identify the corresponding tensor contract. `static_profile` embeds the profile
geometry and position-ID hash. `compute_dtype` records hidden-state compute
types; `shared_encoder_inputs` records the raw context types, and
`shared_encoder_input_observers` records their frozen grids. `quantized_export`
selects the quantized path; each observer also records `fake_quant_enabled`.
Eager/fake-quant tensor dtypes and
serialized quantized Circle boundary dtypes are separate: use the generated
Circle tensors and their frozen scale/zero-point metadata when passing encoded
values between stages, including context inputs. Matching frozen observers make
neighboring serialized boundaries directly chainable without recalibration.

Pooling, padding removal, valid-token selection, standardization, and the final
projection keep their existing stage responsibilities and model/profile settings.

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

## Last-token-only generation prefill

`StaticGemma4Runtime` optionally finishes the trailing shared-KV decoder layers
with query length 1. Enable `debug.static_gemma4_runtime.last_token_only_prefill`
in the static runtime configuration; its default is `false`. This changes only
the initial generation prefill. Calibration and the eager full-text model remain
full sequence. `runtime.prefill(batch, last_token_only=False)` selects the
existing full-prefill reference path for comparison. Calls needing every prompt
hidden state must keep full prefill.

The supported maximum valid prompt is `S-1`, since the existing static runtime
reserves one cache slot for decode; finalize does not change that capacity.

The boundary comes from the instantiated layers, not E2B layer numbers. The
runtime resolves each consumer's preceding, same-type
`store_full_length_kv` producer and checks its explicit cache-source binding and
KV geometry. Only the contiguous terminal run of read-only consumers is eligible;
a model without that run executes full prefill. Malformed sharing raises an
error. Dense decoder RMSNorm, MLP, residual/scalar and PLE fusion operate per token;
only attention mixes tokens, and these consumers read prefix-produced KV.

For the current batch-size-1 right-padding contract, valid decoder positions are
the leading `L` slots after multimodal fusion, so `p = L - 1`. The runtime checks
the decoder validity mask against `valid_length`. It slices hidden state, the
already computed full `ple_projection_prefill` output, RoPE and the original
prefill mask's query row at this same `p`. Logical RoPE position and multimodal,
causal and sliding visibility are preserved. Without `ple_prefill_split`, PLE projection still processes all
`S` positions; it never switches to `ple_projection_decode` during finalize.

The shared-consumer **decode Circle is reused** for finalize. Its ABI is hidden
`[1,1,D]`, PLE `[1,1,P]`, RoPE `[1,1,head_dim]`, mask `[1,1,S]`, and full shared
KV `[1,kv_heads,S,head_dim]`. It uses the same wrapper and frozen observers as
prefill, has no K/V projection or append, and receives the selected prefill mask
row rather than a newly constructed decode mask. Invocation-local shared KV
retains all `S` producer output slots, including the final valid prompt token;
the persistent decode cache separately keeps `S-1` slots. Finalize does not write
cache or increase `past_len`. Normal decode starts at logical position `L` with
the first generated token and runs every decoder layer.

Quantized finalize supports identity/per-tensor affine activation observers,
including the PLE boundary. Per-channel activation and MX activation policies
are explicitly unsupported because their axes/groups need a separate proof for
query slicing; disable this option for those checkpoints. Per-channel and MX
weight policies remain independent. Existing frozen observer grids are reused
without calibration, grid conversion, or an added slice observer. The manifest
records `finalize_supported`; unsupported policies still permit ordinary full
prefill/decode export. Circle reference execution validates fake-quant semantics,
not bit-exact integer NPU behavior.

Export still writes full-prefill and normal decode artifacts for every layer.
`text_pipeline.json` records `finalize_contract:
shared_consumer_decode_alias_v1` and aliases eligible finalize stages to existing
decode files. No duplicate suffix weights or finalize Circle files are created.
The suffix's Q/projection/MLP activations change from `S` rows to 1 row and its
attention query dimension changes from `S` to 1 while the KV dimension remains
`S`. Without `ple_prefill_split`, prefix graphs, PLE projection and ordinary decode
shapes are unchanged.
This structural reduction does not establish an NPU latency or TTFT speedup.

To export and run the floating-point artifact path with a matching model-backed
frontend:

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/gemma4_export.yaml \
  --source model --model google/gemma-4-e2b-it \
  --output-dir ./out/gemma4_fp/circle_layers

python -m tico.quantization.examples.inspector \
  --mode static-gemma4-runtime \
  --config tico/quantization/examples/configs/static_gemma4_runtime.yaml \
  --set debug.static_gemma4_runtime.text_artifacts=./out/gemma4_fp/circle_layers \
  --set debug.static_gemma4_runtime.last_token_only_prefill=true
```

Repeat the inspector command with `last_token_only_prefill=false` for the full
reference. `text_artifacts: null` uses model-backed static adapters instead of
saved Circle decoders. Saved stages use the built-in CPU `reference` runtime.
The inspector frontend is model-backed and does not load a calibrated checkpoint;
for checkpoint-exported quantized decoders, use `runtime.prefill_decoder(...)`
with fused embeddings and full PLE from the matching calibrated frontend. Do not
interpret a mixed floating frontend/quantized decoder run as full checkpoint
parity. The export command near the top of this guide writes those quantized
decoder artifacts from the existing checkpoint.

For an existing runtime with the same model/profile, this shows the calibrated
frontend connection and explicit decoder entry. `fused_hidden`, `full_ple`,
`prefill_masks`, and `prefill_rope` must describe the same right-padded prompt;
`full_ple` retains all `S` positions. Here `qtext` is the frozen text wrapper from
the checkpoint used for export (for a top-level saved PTQ wrapper:
`checkpoint.wrapped.model.wrapped.language_model.wrapped`).

```python
from tico.quantization.wrapq.wrappers.gemma4.export_adapters import (
    Gemma4PLEEmbeddingExportAdapter,
    Gemma4PLEProjectionExportAdapter,
    Gemma4TokenEmbeddingExportAdapter,
)

# Match token/PLE stages for subsequent normal decode as well as prefill inputs.
runtime.text_model = qtext
runtime.token_embedding = Gemma4TokenEmbeddingExportAdapter(qtext).eval()
runtime.ple_embedding = Gemma4PLEEmbeddingExportAdapter(qtext).eval()
runtime.ple_projection = Gemma4PLEProjectionExportAdapter(qtext).eval()
runtime.use_text_pipeline("./out/gemma4_quantized/circle_layers")
logits = runtime.prefill_decoder(
    fused_hidden, full_ple, prefill_masks, prefill_rope, valid_len,
    last_token_only=True,
)
for token_ids in continuation_tokens:  # Each tensor has shape [1, 1].
    logits = runtime.decode_one(token_ids)
```

Replace all four frontend fields together, as above, with stages from the same
calibrated checkpoint and static profile. The runtime tracks the identities of
`text_model`, `ple_embedding`, and `ple_projection`. When any changes, it drops
only its automatically generated split PLE and selective lookup adapters. This
synchronization runs in `use_text_pipeline()` and before subsequent PLE calls,
so field replacement also works without a saved decoder pipeline. Frontend
replacement also resets automatic selective lookup activation. Without an explicit
decode-lookup override, ordinary decode uses the installed public lookup until
supported model-backed split stages are requested and built for the new frontend.
This permits full prefill/decode with a full-only lookup policy even after the old
frontend used split PLE. Current-source stages are reused while the frontend is
unchanged. Existing mode/capability guards still apply; changing quantization mode
in place requires rebuilding the public export adapters. Start a new prefill after
replacement; `use_text_pipeline()` resets the generation cache.

Explicit host lookup and Circle projection connections remain caller-owned. A
caller can replace `ple_embedding`/`ple_projection` with the full-range host and
projection stages and install the corresponding phase stages:

```python
runtime._selective_ple_embedding = shared_host_lookup
runtime._split_ple_stages = (
    prefix_lookup, prefix_projection_circle,
    suffix_lookup, suffix_projection_circle,
)
```

The runtime preserves these explicitly replaced slots independently, including
when it invalidates another slot it generated itself. Keep these stages matched
to the installed frontend and checkpoint; replace explicit stages together when
switching artifact sets. Opaque host/Circle frontends need explicit split stages
for split prefill: the runtime cannot derive their layer ranges from a full-range
artifact and reports a missing split-stage connection instead of substituting
model-backed producers. The single host table remains shared by all lookup views.

For a PLE-disabled model leave `ple_embedding`, `ple_projection`, and `full_ple`
as `None`. This entry does not rerun embedding or layer 0 during finalize.

The loader is
`tico.quantization.recipes.export.gemma4_text_pipeline.load_gemma4_text_pipeline`;
`runtime.use_text_pipeline(directory)` validates its static profile and topology
before dispatch. Saved artifacts without `text_pipeline.json` require re-export
for this loader. The legacy model-backed full-prefill path remains available;
the runtime does not guess a finalize contract from old filenames.

Checkpoint-free regression coverage:

```bash
python -m unittest -v \
  test.quantization.wrapq.wrappers.gemma4.test_shared_kv_suffix \
  test.quantization.recipes.debug.test_static_gemma4_finalize \
  test.quantization.recipes.export.test_gemma4_finalize_circle \
  test.quantization.recipes.export.test_gemma4_finalize_saved_stage \
  test.quantization.recipes.export.test_gemma4_text_pipeline
```

## Selective PLE initial prefill

Enable `export.ple_prefill_split: true` together with `export.prefill_decode: true`
and `export.ple_embedding_format: pt`. The existing `auto` policy also works when
it selects `.pt` (including the dense E2B/E4B tables). A Circle lookup still returns
all layers and therefore rejects this combination before stage export. Set
`ple_prefill_split: false` to retain the previous full PLE producer. Calibration
and the eager model always retain full-sequence PLE computation.

Let `S=max_seq`, `L=num_hidden_layers`, `P=hidden_size_per_layer_input`, and `K`
be the boundary returned by `resolve_gemma4_shared_kv_suffix`. No variant-specific
layer number selects the split. The replacement artifacts are:

| Artifact stem | Inputs | Output | Original layers |
|---|---|---|---|
| `ple_projection_prefill_prefix` | `[1,S,H]`, `[1,S,K,P]` | `[1,S,K,P]` | `[0,K)` |
| `ple_projection_prefill_suffix` | `[1,1,H]`, `[1,1,L-K,P]` | `[1,1,L-K,P]` | `[K,L)` |
| `ple_projection_decode` | `[1,1,H]`, `[1,1,L,P]` | `[1,1,L,P]` | `[0,L)` |

Each stem receives the existing `.q.circle` or `.f32.circle` suffix. The full
`ple_projection_prefill` is not emitted in split mode. Use a separate directory
for full-reference exports; reusing an existing directory can retain obsolete
unreferenced artifacts. Decoder artifacts and `text_pipeline.json` retain their
existing ABI and finalize aliases.

The host selects prefix token rows and columns `[0:K*P]` directly from the one
shared lookup table. It selects suffix columns `[K*P:L*P]` only for original
`input_ids[:,p:p+1]`, where `p=valid_length-1` under right padding. Integer storage
is dequantized after this selection with the original weight qparams; vocabulary
row qparams are gathered by token ID, including repetitions. The complete table
remains resident in memory. There is no full-table float reconstruction during
range lookup, no separate phase tables, and no new storage format or mmap path.

Projection selects weight rows before its Linear operation. Its input for both
phases is **H0**, the original layer-0 input (multimodal fusion output), with the
same `p` selection for suffix. Prefix decoder output is not a PLE projection
input. The P-axis RMSNorm is independent for each token/layer and the original
scale, norm, projection, sum and final observers retain their operation order.
The `per_layer_token_inputs=None` helper branch still omits token addition and
final input scaling. Normal decode always computes all layers for one token.

Version 2 of `ple_pipeline.json` identifies split producers by phase, half-open
layer range, original decoder indices, static shapes, shared lookup reference,
H0/token-selection contract and observer qparams. Tensor `dtype`/`example_dtype`
describe floating fake-quant examples; `abi_dtype` records the serialized affine
boundary storage dtype (or float for identity/bypass). The observer reference
records the source grid and enabled/frozen flags. Version 1 remains the legacy
full producer contract; consumers of version 1 must reject version 2 rather than
assume a full packed output. Unsplit export continues to write version 1 with
additive contract metadata. Host-table schemas 1 (unfused scale) and 2 (fused
scale) retain their existing storage and scale semantics.

Frozen per-tensor affine and identity activation policies support slicing.
Shape-dependent per-channel or MX activations reject explicitly. Weight policies
are validated separately: supported affine channel weights keep their original
scale/zero-point and channel mapping; unsupported policies reject instead of
re-estimating qparams. Adapter creation and tracing do not recalibrate or change
source modes, and detached stages retain the live source capability guard.
For the selected `.pt` lookup, QUANT weights require an enabled affine grid
(per-tensor, vocabulary-row axis 0, or column axis 1, including negative-axis
equivalents); Identity/bypass-only/MX lookup weights reject explicitly. For
model-backed lookup, disabling weight fake quantization after freezing preserves
the original selected weights and retained grid, matching full lookup, while the
activation observers still run. This does not extend the existing integer `.pt`
artifact contract to storing off-grid original weights with weight FQ disabled.
Projection
weights support identity, per-tensor affine, or output-row affine grids. Norm
weights remain on the unchanged P axis. Identity and disabled activation payloads
reuse the existing host schemas and replay no fake-quant operation.

Runtime comparison settings under `debug.static_gemma4_runtime` are:

| Mode | `last_token_only_prefill` | `ple_prefill_split` |
|---|---|---|
| Full reference | `false` | `false` |
| Decoder suffix only (#1141) | `true` | `false` |
| Decoder and selective PLE | `true` | `true` |

Models without PLE or an eligible terminal suffix naturally retain their
existing paths. Invalid topology is an error. The explicit per-call full-prefill
reference override remains available. The runtime creates prefix PLE, releases
its token-lookup temporary, then creates the one-token suffix PLE from H0.
Both output buffers coexist during prefix execution; the prefix buffer and its
last layer view are released at the decoder boundary before suffix execution.
No packed full PLE is reassembled or retained as a decode cache.

```bash
python -m tico.quantization.examples.export \
  --config tico/quantization/examples/configs/gemma4_export.yaml \
  --checkpoint ./out/gemma4_quantized/quantized_model.pt \
  --set export.output_dir=./out/gemma4_split/circle_layers \
  --set export.ple_prefill_split=true \
  --set export.ple_embedding_format=pt

python -m tico.quantization.examples.inspector \
  --mode static-gemma4-runtime \
  --config tico/quantization/examples/configs/static_gemma4_runtime.yaml \
  --set debug.static_gemma4_runtime.last_token_only_prefill=true \
  --set debug.static_gemma4_runtime.ple_prefill_split=true
```

The inspector uses a floating model-backed frontend; checkpoint parity requires
the matching calibrated frontend described above. Run the same runtime command
with the other two setting combinations for comparison.

The produced PLE element count changes from `S*L*P` to `(S*K+L-K)*P`.
Projection MACs are those counts multiplied by `H` (one multiply-accumulate is
one MAC, batch size 1). For `S=2048,L=35,K=15,P=256`, a 2-byte output changes
from 35 MiB to 15.0098 MiB, a 57.1% reduction in that output payload. Lookup
activation and selected dequantization follow the same token/layer counts.
This is not a reduction of the table payload or total model RAM by 57.1%.
The 2-byte example describes an output storage representation; PyTorch floating
and fake-quantized tensors can use 4-byte float32 elements instead. Actual peak
activation also includes lookup/projection/norm/fake-quant intermediates, decoder
state and allocator behavior. Exporting the shared host artifact still performs
the existing full-table weight quantization; the selected-row reduction applies
to inference lookup, not to this one-time artifact construction.

The one lookup table payload is unchanged. Prefix and suffix projection weights
together equal one full projection matrix; normal decode stores another full
matrix, as with the previous prefill/decode pair. Frozen qparam/norm metadata and
Circle headers add per-artifact overhead. Exact artifact bytes, peak host RSS,
host latency and NPU latency/TTFT must be measured on the intended checkpoint and
backend; the element and MAC formulas alone establish none of those timings.

Checkpoint-free selective-producer and real Circle composition checks:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest -v \
  test.quantization.recipes.debug.test_static_gemma4_split_ple \
  test.quantization.wrapq.wrappers.gemma4.test_ple_embedding_selection \
  test.quantization.wrapq.wrappers.gemma4.test_ple_projection_range \
  test.quantization.recipes.export.test_export_gemma4

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest -v \
  test.quantization.recipes.export.test_gemma4_split_ple_circle

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ./ccex test -k gemma4
```

These tests cover FP and frozen UINT8/INT16 PLE producers, original per-channel
weight grids, identity activations, selected dequantization, source-state
ownership, actual bounded Circle constants and quantized boundaries. The
composition test chains one shared `.pt` lookup through real Circle PLE,
decoder and head stages, including continuation decode, using FP and UINT8
semantic reference execution. It does not establish integer NPU equivalence.
