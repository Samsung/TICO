# Llama KV-cache branch validation — 2026-10-26

PR: https://github.com/Samsung/TICO/pull/1083

## Revisions and reproduction

- Fetched main: `294cd692f5f0d8fcf668459d689d6402801a96f3`.
- Original PR head: `fd5cbf43e36e680ab192b5d797af6bb1560fbee7`.
- Validated implementation: `fe47b34d6f9a3c49210ba79d3f5a8186ed5cff2a`.
- Final PR revision: the signed-off validation commit containing this report, on
  `fix/llama-kv-cache-branch-domains` (resolve with `git rev-parse HEAD`). A committed
  report cannot embed its own commit hash; the implementation SHA above identifies
  the exact runtime code exported in the clean candidate worktree.
- The final documentation commit follows the tested implementation commit; it changes no runtime code.

The reusable [comparison script](llama_kv_cache_branch_validation.py) creates a
single local HF float checkpoint, exports both worktrees with the real
`export_llama_per_layer` API, verifies every Circle document, normalizes tensor
contracts and graph connectivity, and executes the serialized decoder stages on
CPU. No weights or datasets are downloaded. Generated models and numerical arrays
stay outside Git.

## Observer and cache contracts

`QuantLlamaAttention` no longer creates `obs_past_key` or `obs_past_value`, or
calls `_observe_past_cache`. There are no aliases or diagnostic replacements.
Only functional observers enter calibration/freeze. `new_k/new_v` own persistent
storage S; `present_key/present_value` own compute C. The current producer P is
an ordinary upstream activation domain.

```text
prefill:
  producer P --+--> Q(P->C) ----------------> attention
              +--> Q(P->S) ----------------> cache delta S

decode:
  cache S --------> Q(S->C) --+
                             +--> Concat C -> attention
  producer P --+--> Q(P->C) --+
              +--> Q(P->S) ----------------> cache delta S
```

Tuple/HF normalization leaves values unchanged in CALIB. QUANT enforces S on
inputs, then converts directly S->C on the compute branch. DynamicCache and
StaticCache append/update storage deltas for both return modes; compute reads do
not overwrite persistent values. StaticCache reads its occupied prefix and
receives cache positions. The full-model decode adapter selects the new delta
without resetting HF cache layers.

Old state dicts and full-module pickle checkpoints containing past-observer
state fail explicitly, including non-strict state-dict loads. The error instructs
users to reprepare the float model and recalibrate with cache generation enabled.
Obsolete qparams are never interpreted as storage or compute qparams. Current
state-dict and full-module roundtrips are tested.

The recipe explicitly enables cache generation with `decode_steps=0`, collecting
storage and compute statistics during prefill alone. Positive decode steps remain
available for optional decode activation calibration.

## Focused numerical and lifecycle coverage

Controlled affine tests use P=0.125, S=0.09375, C=0.21875 (C/S=7/3), with valid
separate UINT8 zero points (P=107, S=113, C=91), and symmetric INT16 zero points
of zero. A hand-computed counterexample distinguishes direct P->C from P->S->C.
Inputs model the serialized affine ABI explicitly; persistent cache values are
already on S. The same frozen weights, qparams, masks and pre-negated-sine RoPE
convention feed eager QUANT and Circle FAKE_QUANTIZE.

All 72 output comparisons across UINT8/INT16, equal/distinct grids,
optimization off/on, prefill/two decode steps, and hidden/K/V outputs were exact:
maximum absolute error 0, mean absolute error 0, differing elements 0. Serialized
metadata tests check S cache inputs/outputs, C Concat operands/results, explicit
branch conversions, value-preserving qparams, signed dtypes, and removal of
identity Quantize operations.

Multi-step tuple/DynamicCache/StaticCache coverage exercises both attention
layouts, UINT8/INT16, delta/present return modes, prefix preservation, updates,
and nonmutating compute reads. Actual Transformers 5.18.0 cache implementations
are exercised, without cache-type skips. Full-model adapters also chain two
steps and return only the new delta.

Prefill-only calibration covers MinMax/Percentile × UINT8/INT16 through normal
prepare/calibrate/convert, with source `config.use_cache=False`. Every functional
observer has finite statistics; storage and compute qparams freeze; eight real
decoder-layer prefill/decode Circle conversions succeed. No past statistics or
decode calls are injected.

## Real model comparison

The locally initialized `LlamaForCausalLM` uses hidden_size=16,
intermediate_size=32, 2 decoder layers, 2 attention heads, 1 KV head, head_dim=8,
max_position_embeddings=16, vocab_size=32, attention_bias=False,
attention_dropout=0, and untied embedding/head weights. The actual NPU-export
PTQ wrapper uses default affine UINT8 MinMax activation calibration. Calibration
uses token IDs `[1,5,3,7,9,4,6,8]`, then connected prefill `[1,5,3]` and two
teacher-forced decode tokens `7,9`. The full-length call enables cache output so
S and C see the same natural range. The seed is 1083; export input seed is 817.
Export uses max_seq_len=8 and prefill_decode=True.

Isolated clean worktrees: `/workspace/TICO-main` and `/workspace/TICO-candidate`.
Both use the same Python executable/environment, configuration, float state,
calibration inputs and export settings. The only export patch supplies the
supported `CompileConfigV1(circle_optimize=...)` to the real converter. Neither
`export_llama_per_layer` nor `convert_and_save_circle` is replaced.

Versions: Python `3.12.14`, Torch `2.13.0+cpu`, Transformers `5.18.0`.

Shared float-state SHA256: `855ed7bca7fd855aa4c291dbabbc237ba260eeb9f2cc50cfa4ed18a61ac4c224`.

In each scenario, **all 135 common observer qparam records are identical**
between main and candidate. Removal of the four layer-local past observers does
not change any common observer's scale, zero point or dtype.

### Operator comparison

Counts include all operators. Every emitted Circle was parsed with
`CircleDocument.from_bytes` and verified. The normalized comparison records I/O,
shapes, dtype, affine qparams, constant content hashes, opcode/options, and
producer/consumer edges; generated names and raw tensor indexes are excluded.
A second normalization bypasses only Quantize nodes to align all other edges.


**natural; o1**

| artifact | main operator count | PR operator count | main Quantize count | PR Quantize count | expected changes | unexpected changes | verdict |
|---|---:|---:|---:|---:|---|---|---|
| decoder_layer_decode_0.q.circle | 56 | 56 | 2 | 2 | none; identical graph | none | pass |
| decoder_layer_decode_1.q.circle | 56 | 56 | 2 | 2 | none; identical graph | none | pass |
| decoder_layer_prefill_0.q.circle | 52 | 52 | 2 | 2 | none; identical graph | none | pass |
| decoder_layer_prefill_1.q.circle | 52 | 52 | 2 | 2 | none; identical graph | none | pass |
| lm_head.q.circle | 2 | 2 | 0 | 0 | none; identical graph | none | pass |
| token_embedding.q.circle | 3 | 3 | 1 | 1 | none; identical graph | none | pass |

**natural; unoptimized**

| artifact | main operator count | PR operator count | main Quantize count | PR Quantize count | expected changes | unexpected changes | verdict |
|---|---:|---:|---:|---:|---|---|---|
| decoder_layer_decode_0.q.circle | 58 | 58 | 2 | 2 | none; identical graph | none | pass |
| decoder_layer_decode_1.q.circle | 58 | 58 | 2 | 2 | none; identical graph | none | pass |
| decoder_layer_prefill_0.q.circle | 54 | 54 | 2 | 2 | none; identical graph | none | pass |
| decoder_layer_prefill_1.q.circle | 54 | 54 | 2 | 2 | none; identical graph | none | pass |
| lm_head.q.circle | 2 | 2 | 0 | 0 | none; identical graph | none | pass |
| token_embedding.q.circle | 3 | 3 | 1 | 1 | none; identical graph | none | pass |

**distinct; o1**

| artifact | main operator count | PR operator count | main Quantize count | PR Quantize count | expected changes | unexpected changes | verdict |
|---|---:|---:|---:|---:|---|---|---|
| decoder_layer_decode_0.q.circle | 58 | 62 | 4 | 8 | A: four branch boundaries | none | pass |
| decoder_layer_decode_1.q.circle | 58 | 62 | 4 | 8 | A: four branch boundaries | none | pass |
| decoder_layer_prefill_0.q.circle | 56 | 56 | 6 | 6 | none; identical graph | none | pass |
| decoder_layer_prefill_1.q.circle | 56 | 56 | 6 | 6 | none; identical graph | none | pass |
| lm_head.q.circle | 2 | 2 | 0 | 0 | none; identical graph | none | pass |
| token_embedding.q.circle | 3 | 3 | 1 | 1 | none; identical graph | none | pass |

**distinct; unoptimized**

| artifact | main operator count | PR operator count | main Quantize count | PR Quantize count | expected changes | unexpected changes | verdict |
|---|---:|---:|---:|---:|---|---|---|
| decoder_layer_decode_0.q.circle | 60 | 64 | 4 | 8 | A: four branch boundaries | none | pass |
| decoder_layer_decode_1.q.circle | 60 | 64 | 4 | 8 | A: four branch boundaries | none | pass |
| decoder_layer_prefill_0.q.circle | 58 | 58 | 6 | 6 | none; identical graph | none | pass |
| decoder_layer_prefill_1.q.circle | 58 | 58 | 6 | 6 | none; identical graph | none | pass |
| lm_head.q.circle | 2 | 2 | 0 | 0 | none; identical graph | none | pass |
| token_embedding.q.circle | 3 | 3 | 1 | 1 | none; identical graph | none | pass |

Natural graphs are completely identical, including tensor contracts. Both
distinct-grid prefill graphs, embedding, and LM head also remain identical. All
24 paired artifact comparisons have identical non-Quantize topology, operator
options and shapes. No unexpected operator insertion, deletion or rewiring was
found. There are no candidate identity Quantizes or value-preserving qparam
mismatches, including views, splits, tiles, pack/unpack and Concats.

For reference, the natural unoptimized prefill opcode multiset is ADD×7,
BATCH_MATMUL×4, CONCATENATION×6, DEQUANTIZE×1, FULLY_CONNECTED×7, LOGISTIC×1,
MUL×8, QUANTIZE×2, RESHAPE×7, RMS_NORM×2, SOFTMAX×2, SPLIT_V×3,
STRIDED_SLICE×2, TRANSPOSE×2. Decode adds two storage views and two compute
Concats. Every count/type difference from this base in the paired distinct
comparison is the four explicit Quantizes listed below.

The Q/K/V projection weights and inputs, RoPE arithmetic, logits matmuls,
mask additions, softmaxes, attention-value matmuls, output projections,
normalizations, MLPs, residual additions, embedding and LM head retain identical
non-Quantize dataflow and options. The intended distinct-grid K/V producer
output metadata correction is described explicitly below.

O1 canonicalizes three SPLIT_V operators to SPLIT and removes two redundant
Reshapes per decoder artifact, equally on both branches. It removes **no**
Quantize. Missing main branch boundaries are already missing before O1; this
is a TICO folding/propagation issue, not an optimizer deletion.

### Distinct-grid paths and qparam changes

Qparams were loaded and locked through the ordinary observer API after
calibration, before normal conversion. All domains are valid per-tensor UINT8:

| branch | P (scale, zp) | S (scale, zp) | C (scale, zp) |
|---|---|---|---|
| K, both layers | 0.0014, 121 | 0.0017, 119 | 0.0011, 127 |
| V, both layers | 0.0015, 125 | 0.0019, 123 | 0.0013, 129 |

Actual serialized values retain the corresponding float32 scale representation.
Each candidate decode adds precisely K P->C, V P->C, K S->C, and V S->C.
The P->S cache output Quantizes were already present and now consume genuine P.

Main had overwritten the V projection output and post-RoPE K ADD output with C;
the candidate restores their legitimate P grids. Past-input view tensors change
from C to S. The two single-input storage stack Concats previously had C inputs
and P outputs; their inputs now correctly retain P. This resolves exactly two
main value-preserving metadata anomalies per distinct decode artifact. Both
compute Concat operands/results have C. These are category A, required domain
corrections. There are no category C unexplained changes or category D newly
introduced regressions. O1's identical canonicalizations are category B.

### Serialized cache ABI and reuse

All four candidate K/V checks (two layers independently) pass in both scenarios
and both optimization settings: prefill cache output, decode past input and
decode delta output match in dtype, scale and zero point. Main passes naturally,
but fails all four in the distinct case: main past inputs carry C instead of S.

Natural serialized storage contracts:

| layer | K scale / zp | V scale / zp | dtype |
|---|---|---|---|

| 0 | 0.000502778973896 / 129 | 0.00145123817492 / 135 | uint8 |
| 1 | 0.000499287678394 / 142 | 0.00157690723427 / 145 | uint8 |

Runtime prefill uses eight static slots, preserves the first three causal prompt
slots, then appends two token deltas into a seven-slot past allocation with
explicit masks. Cache composition only copies/appends serialized S values; it
performs no host-side requantization. Each layer successfully chains two decode
steps. External hidden/mask/RoPE inputs are encoded to the serialized affine ABI
before eager and Circle execution; past cache values come directly from prior
Circle outputs.

### Real-model CPU runtime results

Main versus candidate natural results are **exact for every tested output**:
18 outputs (two layers × prefill/two decodes × hidden/K/V), repeated with O1 off
and on. Maximum absolute error=0, mean absolute error=0, differing elements=0.
Associated output scales appear in the eager comparison tables below. O1 does
not change any numerical results on either branch in either scenario.

Candidate versus eager QUANT is recorded below for **every** tested output.
An exact row has zero differing elements. These figures apply identically to O1
off and on; no tolerance is used to declare equality.


**Candidate Circle versus eager: natural**

| layer/stage | output | scale | exact | max absolute error | mean absolute error | differing / elements |
|---|---|---:|---|---:|---:|---:|
| layer0_decode1 | hidden | 0.00044009994599 | yes | 0 | 0 | 0 / 16 |
| layer0_decode1 | k | 0.000502778973896 | yes | 0 | 0 | 0 / 8 |
| layer0_decode1 | v | 0.00145123817492 | yes | 0 | 0 | 0 / 8 |
| layer0_decode2 | hidden | 0.00044009994599 | yes | 0 | 0 | 0 / 16 |
| layer0_decode2 | k | 0.000502778973896 | yes | 0 | 0 | 0 / 8 |
| layer0_decode2 | v | 0.00145123817492 | yes | 0 | 0 | 0 / 8 |
| layer0_prefill0 | hidden | 0.00044009994599 | no | 0.00044010207057 | 1.03148486232e-05 | 3 / 128 |
| layer0_prefill0 | k | 0.000502778973896 | no | 0.0045250095427 | 0.000102126999991 | 3 / 64 |
| layer0_prefill0 | v | 0.00145123817492 | yes | 0 | 0 | 0 / 64 |
| layer1_decode1 | hidden | 0.000460266543087 | yes | 0 | 0 | 0 / 16 |
| layer1_decode1 | k | 0.000499287678394 | yes | 0 | 0 | 0 / 8 |
| layer1_decode1 | v | 0.00157690723427 | yes | 0 | 0 | 0 / 8 |
| layer1_decode2 | hidden | 0.000460266543087 | yes | 0 | 0 | 0 / 16 |
| layer1_decode2 | k | 0.000499287678394 | no | 0.000499289482832 | 6.2411185354e-05 | 1 / 8 |
| layer1_decode2 | v | 0.00157690723427 | yes | 0 | 0 | 0 / 8 |
| layer1_prefill0 | hidden | 0.000460266543087 | no | 0.000460267066956 | 7.19167292118e-06 | 2 / 128 |
| layer1_prefill0 | k | 0.000499287678394 | no | 0.000499289482832 | 1.56027963385e-05 | 2 / 64 |
| layer1_prefill0 | v | 0.00157690723427 | yes | 0 | 0 | 0 / 64 |

**Candidate Circle versus eager: distinct**

| layer/stage | output | scale | exact | max absolute error | mean absolute error | differing / elements |
|---|---|---:|---|---:|---:|---:|
| layer0_decode1 | hidden | 0.00044009994599 | yes | 0 | 0 | 0 / 16 |
| layer0_decode1 | k | 0.00170000002254 | yes | 0 | 0 | 0 / 8 |
| layer0_decode1 | v | 0.00190000003204 | yes | 0 | 0 | 0 / 8 |
| layer0_decode2 | hidden | 0.00044009994599 | yes | 0 | 0 | 0 / 16 |
| layer0_decode2 | k | 0.00170000002254 | yes | 0 | 0 | 0 / 8 |
| layer0_decode2 | v | 0.00190000003204 | yes | 0 | 0 | 0 / 8 |
| layer0_prefill0 | hidden | 0.00044009994599 | no | 0.00044010207057 | 1.37531314977e-05 | 4 / 128 |
| layer0_prefill0 | k | 0.00170000002254 | no | 0.00510000064969 | 0.000106249994133 | 2 / 64 |
| layer0_prefill0 | v | 0.00190000003204 | yes | 0 | 0 | 0 / 64 |
| layer1_decode1 | hidden | 0.000460266543087 | yes | 0 | 0 | 0 / 16 |
| layer1_decode1 | k | 0.00170000002254 | yes | 0 | 0 | 0 / 8 |
| layer1_decode1 | v | 0.00190000003204 | yes | 0 | 0 | 0 / 8 |
| layer1_decode2 | hidden | 0.000460266543087 | yes | 0 | 0 | 0 / 16 |
| layer1_decode2 | k | 0.00170000002254 | yes | 0 | 0 | 0 / 8 |
| layer1_decode2 | v | 0.00190000003204 | yes | 0 | 0 | 0 / 8 |
| layer1_prefill0 | hidden | 0.000460266543087 | no | 0.000460267066956 | 3.59583646059e-06 | 1 / 128 |
| layer1_prefill0 | k | 0.00170000002254 | yes | 0 | 0 | 0 / 64 |
| layer1_prefill0 | v | 0.00190000003204 | yes | 0 | 0 | 0 / 64 |

Candidate distinct decode is exact for all 12 outputs per optimization setting.
Main distinct decode cache deltas differ from the candidate by up to one S
step: K max≈0.0017, V max≈0.0019, across both successive steps. The full metrics
are preserved locally in `tico-kv-validation-comparison.json`.

### Inherited RoPE numerical limitation

The natural eager comparison is **not** an unconditional parity pass. The largest
error is layer-0 prefill K: max 0.00452500954 = 9 storage LSBs, 3/64 values.
The comparison established identical main/candidate graphs and numerical
results before O1, then traced the first differing intermediate. RMS and K
projection output are exact. Eager's first-half K observer has scale
0.00043621636 and zp115, saturating a projection value near -0.06600351 to
-0.05016488. Circle's split/Concat path retains the projection grid
(scale≈0.0005238374, zp133) and bypasses that independent half-vector observer.
The first RoPE Concat differs at four values (max≈0.01571512); sine multiplication
and addition produce the three final K differences. This is **lost intermediate
quantization/saturation**, not a Torch/Circle tie-rounding difference.

Layer 1 has the same independently calibrated half-vector issue: its x1 scale
is≈0.0003388508/zp147 while projection/Concat use≈0.00050466246/zp142.
Changing only the diagnostic eager x1 grid to the projection grid makes each
layer's prefill K exactly match its unchanged Circle artifact. This
counterfactual does not alter either comparison model or any saved export.

The small hidden-output differences have a separate rounding-only cause. In
layer 0 the mask-add logits match exactly, but six uniform softmax probabilities
1/6 have scaled float32 value 42.5 on the UINT8 softmax grid. Torch rounds ties to
even (42/255 = 0.1647058874); Circle rounds ties away from zero
(43/255 = 0.1686274558). Six of 64 softmax elements differ by 1/255, ultimately
changing three of 128 hidden values by one output LSB. Aligning only the RoPE
half-vector grid and this diagnostic softmax rounding convention makes all three
layer-0 outputs exactly equal (128 hidden, 64 K, 64 V elements). Neither
counterfactual is used in the reported paired exports or their metrics. The same
six half-ties also explain the two layer-1 prefill hidden differences; aligning
its diagnostic RoPE halves and softmax rounding likewise makes all outputs exact.
[The diagnostic script](llama_kv_cache_rope_diagnostic.py) and local JSON traces
make the attribution reproducible. It is an inherited limitation outside the
new cache branch domains; the main comparison proves no new numerical regression.


## Commands and outcomes

Commands ran from the PR checkout with its virtual environment activated and
CPU thread counts limited for the small models:

| command | outcome |
|---|---|
| `./ccex configure test --cpu_only` | pass; Torch/TorchVision pair and pip check |
| `./ccex configure format` | pass |
| `./ccex test -k quant_attention` | 24 pass, no skips (final rerun) |
| `./ccex test -k kv_cache_circle` | 4 pass, no skips; 72 exact output comparisons |
| `./ccex test -k quantize_attention_decode` | 4 pass, no skips |
| `./ccex test -k propagate_qparam_backward` | 15 pass, no skips |
| `./ccex test -k quant_decoder_layer` | 23 pass, no skips |
| `./ccex test -k llama_adapter` | 9 pass, no skips |
| `./ccex test -k hf_cache` | 4 pass, no skips (includes unrelated Qwen contract) |
| `./ccex test -k quant_model` | 52 run, 41 pass, 11 existing internal smoke skips |
| `./ccex test -k llama_prefill_calibration` | 1 pass, four observer/dtype subcases, 8 actual Circle exports |
| teacher-forcing/negative decode recipe unittest cases | 2 pass |
| `./ccex format` | pass, including final rerun after typing fixes and diagnostic additions |
| `./ccex format --no-apply-patches` | final pass: no lint issues; earlier local typing findings corrected |
| `./ccex test` | 4,443 run: 4,286 pass, 157 skips, zero failures (1,532.823 seconds) |
| real export script, main and candidate | pass; 48 verified Circle artifacts total |
| normalized graph/runtime comparison | complete; results above |

The full suite completed successfully. Its 157 skips remain skips, not passes;
they follow existing optional/internal/version/backend gating. The 11 skips
in the broader model filter are opt-in internal smoke tests
requiring `RUN_INTERNAL_TESTS=1`; no Llama cache type was skipped. An early direct
recipe unittest invocation lacked the harness's test PYTHONPATH; the canonical
ccex invocation passed. An early comparison invocation preceded the main
manifest and was rerun successfully after both exports finished. A duplicate
format check was stopped while apply-format was still running, then rerun in
sequence. These interrupted/failed attempts are not counted as passing checks.

Reproduction (use the same Python executable for both exports):

```bash
python docs/archive/llama_kv_cache_branch_validation.py init --fixture /tmp/tico-kv-validation-fixture
python docs/archive/llama_kv_cache_branch_validation.py export --repo /workspace/TICO-main --fixture /tmp/tico-kv-validation-fixture --output /tmp/tico-kv-validation-main
python docs/archive/llama_kv_cache_branch_validation.py export --repo /workspace/TICO-candidate --fixture /tmp/tico-kv-validation-fixture --output /tmp/tico-kv-validation-candidate
python docs/archive/llama_kv_cache_branch_validation.py compare --main /tmp/tico-kv-validation-main --candidate /tmp/tico-kv-validation-candidate --output /tmp/tico-kv-validation-comparison.json
```

Generated files are retained locally at:

- `/tmp/tico-kv-validation-fixture/float_state.pt`
- `/tmp/tico-kv-validation-main/{natural,distinct}/{o1,unoptimized}/`
- `/tmp/tico-kv-validation-candidate/{natural,distinct}/{o1,unoptimized}/`
- `/tmp/tico-kv-validation-comparison.json`
- `/tmp/llama-rope-diagnostic.json` and `/tmp/llama-rope-diagnostic-layer1.json`
- `/tmp/tico-*.log` validation logs

No generated Circle models, checkpoints, arrays or traces are committed.

## Remaining risks and readiness

The branch-domain implementation has no observed regression versus main:
controlled UINT8/INT16 parity, HF cache reuse, prefill-only calibration and all
serialized ABI/topology checks pass. There are **zero unexpected operator/node
changes**. However, full real-model eager/Circle parity is **not established**:
the inherited RoPE half-observer loss above remains, with a 9-LSB cache error in
one prefill case. Therefore this report does **not** give unconditional numerical
approval or claim the PR fully safe to merge under a general eager-parity gate.
The branch fix is ready for review relative to main; the preexisting RoPE issue
needs its own correction/acceptance decision. No NPU execution is claimed.

Reference FAKE_QUANTIZE validates
quantization semantics on CPU; it does not establish bit-exact NPU arithmetic or
target-backend performance. The PR remains draft and is not merged automatically.
