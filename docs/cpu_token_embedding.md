# Explicit quantization for CPU token embedding

This implementation connects PyTorch export to serialized Circle and the built-in
CPU **reference runtime in NATIVE mode**. It does not implement an ONERT/TV kernel.
See [design](design.md), [runtime contracts](../tico/circle/README.md), and
[large-buffer ownership](large_circle_export.md) for the existing architecture.

## Baselines and observed contracts

Work started at `f8e9b036cbd91cc030ae5e8e712d6a9702cf2e87`, clean branch `work`;
implementation branch: `cpu-explicit-embedding`. There was no local `main`.
The configured origin is the `mhs4670go/TICO` fork, with stale `origin/main`
`376800445abeee2004a6fd761daf9077c17b4df2`. Live Samsung/TICO main was queried
separately: `42779caf9229ac176b47bfa6e341de84b07e8622`. The checkout is an ancestor of live main, which is four commits ahead. A fetched
`upstream-review/main` comparison found the relevant embedding wrappers, passes,
serializers, executor and heavy folding unchanged. Main adds unrelated fixes for
reverse STRIDED_SLICE bounds, scalar pow promotion, negative affine channel axes
and fake-quant enable flags (axis-0 enabled embedding is unaffected). No merge/reset
was performed.

Code inspected, rather than assuming all configuration defaults:

* `wrapq/wrappers/gemma4/quant_text_scaled_word_embedding.py` explicitly sets
  weight PER_CHANNEL_ASYMM, axis 0. `obs_embedding`, `obs_embed_scale`, and
  `obs_act_out` are per-tensor. Its QUANT forward is weight FQ → embedding →
  embedding FQ → MUL(fake-quantized scale) → output FQ. IDs remain integer indices.
  Generic `wrappers/nn/quant_embedding.py` similarly forces row-wise weight observation.
* The `gemma4_quantize.yaml` preset selects UINT8 embedding/LM head and INT16
  activation; the bare Gemma4 builder defaults to INT16 weights. UINT8 is therefore
  a configuration contract, not an unconditional model-family property.
  Affine snapshots use 1D row scales and INT32 zero points; imported explicit
  graphs can also supply INT64 zero points. Circle stores FP32 scales/INT64 zps.
* `gemma4/embedding_scale_fusion.py` operates before quantization, only on fresh
  FP PLE `embed_tokens_per_layer`, and rejects shared storage. The ordinary tied
  token table is not modified by that transform or by this change.
* Before this change, `FoldQuantOps` folded Q/DQ into the embedding output's
  qparams, then `RemoveWeightDequantOp` removed the table DQ. Thus the serializer
  emitted UINT8 GATHER with independent activation-domain output, hiding conversion.
* `ConstPropPass` already skips weight DQ and caches constant quantization by
  storage identity plus qparams. `circle_serializer.py` preserves compatible aliases
  within an artifact using storage identity, layout and qparams. Separate staged
  token-embedding and LM-head files already have separate physical buffers.
  Logical tying does not imply cross-file sharing or a common observer.
* Staged token embedding has IDs `[1,S]`, output `[1,S,H]`; decoder input has its
  own frozen `obs_act_in` contract. This change preserves existing exported output
  qparams. It does not assert that independently calibrated stage boundaries agree.

## Export and execution

A detached baseline worktree also reproduced actual serialized INT16-activation
behavior: `GATHER(UINT8,INT64)->INT16; MUL(INT16,INT16)->INT16`. NATIVE execution
failed at GATHER with a UINT8-result/INT16-contract dtype error. A UINT8 Q_A to
INT16 Q_B scaled fixture failed baseline export itself in dtype-mismatch legalization.
Neither invalid baseline was used as the benchmark oracle. The changed fixture
serialized `DEQUANTIZE,GATHER,QUANTIZE,DEQUANTIZE,DEQUANTIZE(scale),MUL,QUANTIZE`
and prepared one selected-row plan.

The narrow structural export policy matches constant UINT8 rank-2 per-channel
axis-0 DQ feeding `aten.embedding` (range 0..255). Other Conv/FC/Concat and legacy
non-target embedding policies are unchanged.

```text
Before (implicit domains):
  Wq[UINT8,row qparams] -> GATHER[activation qparams] -> quantized MUL -> INT16

After (ordinary standard Circle operators):
  Wq -> DEQUANTIZE -> GATHER(float) -> Q_A -> DQ_A
       -> MUL(float, DQ(quantized scale constant)) -> Q_B[INT16]

CPU prepared plan:
  Gather integer rows + corresponding row scale/zp
       -> selected FP32 rows -> Q_A -> DQ_A -> float MUL -> Q_B
```

`PreserveEmbeddingQuantization` runs before Q/DQ folding. It marks only the
structurally identified CPU island, keeps its internal rounding/clipping boundaries,
and leaves the terminal Q as the existing integer ABI. The marker is export policy,
not runtime metadata. A shared table's FC connection still uses original integer
codes and qparams. `check_missing_qparam` recognizes these real float tensors and
semantic index/qparam operands; it does not invent float qparams or disable NPU
checks globally. The per-channel DQ serializer now attaches qparams to the integer
**input**, decodes zp storage correctly, and leaves the output truly float.
Default Circle O1 BASIC does not fold DQ; an additional HEAVY-folding guard protects
the row-wise DQ feeding dynamic GATHER from becoming a full-table float constant.

The runtime builds its plan once from loaded Circle tensors, operator options,
constants and qparams, independent of wrapper names/model family. It skips the
virtual full-table DQ and executes `(float32(Wq[ids]) - zp[ids]) * scale[ids]`.
Existing Q, DQ, MUL and bounds validation are reused. Q/MUL are not algebraically
folded into the weight or fused across quantization boundaries.

Supported optimized runtime pattern:

* constant UINT8 rank-2 table, static shape, row-wise finite positive FP32 scales
  and UINT8-range zero points, axis 0 and exactly V entries;
* GATHER axis 0 (or equivalent -2), `batchDims=0`, unquantized INT32/INT64 IDs of
  any rank (including empty/dynamic sequence dimensions), FLOAT32 output;
* DQ has only compatible GATHER consumers, and its full result is not a graph
  output. Multiple GATHERs can share that virtual DQ. Other consumers of the raw
  integer weight remain valid and are untouched.

Unsupported target qparams, batch dimensions, shared full-DQ outputs or non-GATHER
DQ consumers produce an explicit unsupported diagnostic; no hidden full-table
fallback occurs. Export diagnoses a fake-quantized dynamic MUL side operand explicitly (static
scale is required for this island). Generic per-tensor/non-UINT8/non-target graphs retain their normal
execution. `enable_rowwise_gather=False` is an explicit diagnostic/oracle option for
small models, not an automatic fallback. Trace mode omits the virtual full DQ.

Loading keeps UINT8 storage plus O(V) qparam arrays. Execution allocates selected
integer/FP32 rows and O(number of IDs) selected qparams, not full FP32/INT16/scaled
copies of W. The native path never uses the fake-quant constant cache. In fake mode
an exclusive selected-row table also avoids that cache; other consumers such as FC
can still require the existing full-weight fake-quant cache. That mode is not the
native embedding memory claim.

## Arithmetic and ABI

Reference Q uses FP32 division, half-away-from-zero rounding and storage-range
saturation. WrapQ PyTorch fake quant uses ties-to-even. Exact ties can therefore
legitimately differ between those two arithmetic contracts; no tolerance hides this.
Optimized versus full-DQ **native** execution is exact, including ties and saturation.
The deterministic Gemma4 WrapQ sample avoids ties and matches dequantized native
output exactly. The default INT16 range is `[-32768,32767]`. For explicitly supplied
`quant_min=-32767`, export adds a standard float MAXIMUM before Q to preserve that
restricted lower bound; it does not misuse Circle min/max calibration metadata.

The original output shape/layout, scale/zp, dtype and clipping range are retained.
CPU float intermediates are never the NPU boundary output. Original shared codes,
row qparams and LM-head layout/domain are unchanged. There is no added table copy
for embedding; existing package/file ownership is unchanged.

## Reproduction and evidence

Use the installed CPU environment (measured with Python 3.12.14, Torch 2.13.0+cpu,
NumPy 2.5.4). No model/data download, accelerator or external runtime is needed.
The tiny optional Gemma4 wrapper test needs a transformers installation providing
Gemma4 (measured with 5.18.0); generic export/runtime coverage has no such requirement.

```bash
source .venv/bin/activate
python -m unittest test.quantization.passes.test_explicit_embedding_export -v
python -m unittest test.quantization.passes.test_preserve_embedding_quantization -v
python -m unittest discover -s test/unit_test/circle/runtime -q
python -m unittest test.unit_test.circle.passes.optimization.fold.test_heavy -q
./ccex test
./ccex format --no-apply-patches
python -m test.performance.benchmark_rowwise_embedding --repeat 30 --warmup 5
python -m test.performance.benchmark_rowwise_embedding --vocab 8192 --prefill 256 --repeat 10 --warmup 3
```

Tests serialize actual Circle bytes, repack/reload, verify structure, and execute
NATIVE integer output. They cover decode/prefill, unequal row grids, repeated and
unsorted IDs, first/last rows, bounds errors, INT32/INT64 IDs, empty prefill,
rounding/saturation, scale operand FQ, intermediate Q_A, shared integer weight,
multiple DQ consumers and explicit rejection, and per-tensor fallback. A shared
embedding/Linear export asserts one table constant, exact original codes/qparams,
unchanged FC layout/domain and independent FC fake-quant reference semantics.
FC native quantized arithmetic remains outside reference runtime support.

Memory tests hook DQ kernels and selected integer-row gathers during **loading and
execution**, forbid full-table DQ calls, inspect the fake cache, and compare selected
shapes. HEAVY folding tests forbid the full DQ evaluator. This is stronger than
output equality alone, but does not constitute OS/native allocator instrumentation.

Cloud benchmark: Intel Xeon Platinum 8573C VM, 5 visible CPUs, Torch 1 thread,
W `[4096,128]`, warmup 5, 30 repeats, median excluding loading; memory sampled
separately without trace. No CPU affinity pinning or exclusive VM reservation was used. Both paths implement the same explicit graph and compare
final INT16 exactly before timing.

| Input | Full-DQ oracle | Selected rows | tracked run peak, full / selected |
|---|---:|---:|---:|
| decode `[1,1]` | 1.0664 ms | 0.1855 ms | 4,262,800 / 9,340 B |
| prefill `[1,128]` | 1.2165 ms | 0.3878 ms | 4,262,800 / 396,806 B |

With V=8192, selected decode peak remains 9,340 B, while full peak grows to
8,489,872 B. T=256 raises selected peak to 790,022 B. Selected FP32 row payload is
512 B/decode or 65,536 B/T128; full-table payload would be 2,097,152 B. Instrumented
restoration shapes were `[1,1,128]` / `[1,T,128]` plus small scalar DQ, never `[V,D]`.
Tracked loading peak grows about 3.05 MB → 6.04 MB with doubled V, consistent with
integer storage and O(V) metadata, with no DQ evaluation during loading.
`tracemalloc` counts tracked Python/NumPy allocations, not process RSS, allocator
caches, or untracked native memory. Timings are reference-runtime VM measurements,
not estimates for TV CPU or NPU.

## External runtime and device work

The default interpreter dispatch remains reference; external adapters are explicit.
No ONERT or circle-interpreter code was changed. Production runtimes need to accept
standard per-axis UINT8 DQ and FLOAT32→INT16 Q, preserve Q_A/DQ_A and scale FQ,
support any reduced-range clamp, and implement their own prepare-time selected-row
plan without full-table constant folding. Their arithmetic must be compared on exact
ties/saturation. The existing ONERT adapter only resizes FLOAT32 dynamic inputs;
dynamic INT64 token IDs need separate adapter work. circle-interpreter has its
existing dynamic-shape restrictions. Full Gemma4 checkpoint export, actual staged
decoder calibration compatibility, large-file packaging, TV CPU/NPU execution and
performance were not validated by these small CPU tests.

Completed focused checks (all commands run with the repository CPU environment):

| Check | Result |
|---|---|
| `python -m unittest test.quantization.passes.test_explicit_embedding_export test.unit_test.circle.runtime.test_rowwise_gather -q` | PASS, 15 tests |
| `python -m unittest test.quantization.passes.test_preserve_embedding_quantization -q` | PASS, 4 tests |
| `python -m unittest discover -s test/unit_test/circle/runtime -v` | PASS, 175 tests, no skips |
| `python -m unittest discover -s test/quantization/passes -p 'test_*quant*.py' -v` | PASS, 94 run, 2 existing Torch slice-related skips (before adding the dynamic-scale negative case) |
| `python -m unittest test.unit_test.circle.passes.optimization.fold.test_heavy -q` | PASS, 18 tests, including unsupported batchDims protection |
| `python -m unittest test.quantization.wrapq.wrappers.nn.test_quant_embedding test.quantization.wrapq.wrappers.nn.test_quantize_tied_embedding test.quantization.wrapq.wrappers.gemma4.test_quant_text_scaled_word_embedding test.quantization.wrapq.wrappers.gemma4.test_gemma4_ple_scale_fusion test.unit_test.utils.test_runtime_independence -q` | PASS, 41 tests |
| `python -m unittest test.quantization.recipes.test_llama_circle_runtime.TestFakeQuantizedLlamaCircleRuntime -q` | PASS, 3 tests |
| `python -m unittest test.quantization.wrapq.test_conditional_export_entries -q` | PASS, 6 tests |
| `./ccex format --no-apply-patches` | PASS |
| `lintrunner -r HEAD --output oneline` plus explicit paths for the last HEAVY/Llama/conditional-test edits | PASS (UFMT, Pylint, mypy) |
| Both benchmark commands above | PASS, exact native INT16 equality |

Some preliminary suite runs were interrupted to incorporate review fixes and replace
obsolete single-GATHER structural assertions; those runs are not counted as passes.
One complete preliminary run (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./ccex test`)
ran 4,849 tests in 1,242 s: FAIL with one obsolete Gemma4 assertion forbidding Q/DQ,
and 158 existing skips. The assertion now checks explicit Q/DQ, FLOAT32 GATHER,
and a selected-row plan instead; its module rerun passed all 9 tests:
`python -m unittest test.quantization.wrapq.wrappers.gemma4.test_conditional_capabilities -q`.
Its configured UFMT/Pylint/mypy checks also passed.
Test annotation/style errors found by the initial changed-file lint were corrected.
A direct benchmark-only mypy invocation reported four existing imported MX utility
errors; the configured repository lint commands above passed.


After the final structural-test updates, the full rerun
`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ./ccex test` was **INTERRUPTED** by the execution
environment transition after 2,762 tests had started, with no failure logged. Its
process exited without a unittest completion summary; it is not a passing full run.
A completely green full-suite run after the last test-only update was not obtained.
The completed broad run and the focused reruns above must be read separately.
After reconnecting, the runtime suite passed again (175 tests), and the combined
embedding, tied-weight, Gemma4 conditional, Llama saved-artifact, runtime-independence
and HEAVY-folding regressions passed again (77 tests, no skips). The entire
quantization-pass directory was also rerun:
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s test/quantization/passes -q`
passed 181 tests with 2 existing skips. Together these recovery checks ran 433 tests:
431 PASS, 2 SKIP, 0 FAIL. The newly added native embedding cases were not skipped.
