# WrapQ frozen-state follow-up: storage validity and execution compatibility

This file records the follow-up requested for PR #1097 after the earlier archived
investigation was deliberately removed. Historical investigation and validation
remain in Git history; they are not relabeled as runs of this correction.

## Baseline and reproduction

- Working branch: `fix/wrapq-frozen-state`.
- Inspected baseline: `5575029ebdea5fe16a19dfc5b672316d281b1557`.
- Environment: Python 3.12.14, Torch 2.13.0+cpu, Torch source commit
  `cf30153c4c131c8164ee7798e5022d810682e2cb`. The repository's qualified stable
  window is 2.11–2.13, with 2.13 the default.
- CUDA build version: none; CUDA availability: false; device count: zero.

The storage regression was reproduced before modifying `affine_base.py`:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest \
  test.quantization.wrapq.observers.test_execution_compatibility.TestAffineExecutionCompatibility.test_cpu_bfloat16_grid_is_valid_frozen_model_state
```

Exit 1: one test with two subtest errors, for MinMax and EMA. Both failures came
from `load_qparams()` → `_validate_grid()` rejecting CPU BF16 scale storage. The
fixture established extent `(3, 2)` through `observe_shape()` and imported exact
BF16 scales `[0.125, 0.25, 0.5]` with int32 zero points `[3, 7, 11]`. There were no
dependency or fixture failures. The retained log is
`/tmp/tico-bf16-storage-before.log` in the validation workspace.

## Root cause and ownership

The affine grid validator mixed state validity with a native CPU kernel
restriction. Import, candidate derivation/installation, lifecycle validation and
checkpoint restoration consequently treated the current qparam storage device as
an execution target. A structurally valid checkpoint could not be stored or
inspected on CPU even if its intended execution device was elsewhere.

The affine owner now separates those responsibilities:

| Boundary | Responsibility |
|---|---|
| `_validate_grid()` and lifecycle/checkpoint validation | Validate scale dtype, finite positive values, grid/ZP schema and range, symmetry, semantic channel extent, policy, provenance and locks without the CPU storage restriction. |
| `validate_execution(device, module_path=...)` | Check the enabled per-channel BF16-scale CPU operation using metadata only; never freeze, derive, cast, move or mutate state. |
| `fake_quant(x)` | Preserve policy/readiness and bypass behavior, then apply the same compatibility rule using the actual `x.device` before native dispatch. |
| Shared recipe export entry points | Check explicitly declared stage execution sites before conversion, retaining capability and captured-mode guards. |

Valid CPU BF16 snapshots retain exact values and dtype, int32 ZP, shape, policy,
provenance, lock and fake-quant policy. An imported grid with unknown channel
extent remains pending. Unsupported FP16/FP64 per-channel scales and corrupt
strict/non-strict checkpoints still reject. No replacement grid or fake
calibration is introduced. Alias-path ownership, conditional entry, atomic
freezing, cache domains and independent pooler profile snapshots are unchanged.

The execution restriction concerns BF16 **per-channel scales**. It does not reject
BF16 activations/weights using FP32 scales, per-tensor policies, Identity or MX.
A disabled site returns its original input. Frozen/capability readiness does not
promise execution on every device.

## Explicit export participation

`convert_and_save_circle()` and `export_full_circle()` invoke common preflight
before `tico.convert()`. The private `_export_execution_sites(example_inputs,
kwargs)` declaration identifies observer paths and the actual operation devices.
Transparent wrappers delegate only their selected computation. The validator
does not recursively discover executable paths from module registration or
capability readiness.

Weight fake quantization executes where the weight lives, which can differ from
the token/input device. Bound input sites use their actual example tensor device.
Qparam storage, a first model parameter and global CUDA availability are not
execution-device selectors. Unregistered live capability-owner references and
unused source/profile branches are excluded from the computation declaration.

Current early declarations cover Linear and Embedding leaves, Gemma4 scaled-word
embeddings, LLaMA/Gemma4/Qwen3-VL token lookup adapters, Gemma4 PLE lookup and its
producer-owned output site, and the token path in Qwen3-VL multimodal embedding.
These declarations preserve distinct model semantics. Rotation children delegate
known operand sites without fabricating an input; an input-dependent rotation
site without a bound example remains runtime-only.

Decoder, whole-vision, assistant core, arbitrary full-model/custom routes without
an exact declaration, and direct `tico.convert()` or
`torch.export.export()` calls retain runtime checking. Preflight makes no blanket
support promise for undeclared sites or devices. The lower-level guard also
protects subsequent model moves and direct observer calls.

## Native CPU evidence and CUDA limitation

```bash
.venv/bin/python /tmp/tico_bf16_native_probe.py > /tmp/tico_bf16_native_probe.json
```

Exit 0; the script independently asserts eight native CPU cases without TICO or
mocks:

| Input dtype | Scale dtype | Granularity | Native result on installed Torch |
|---|---|---|---|
| FP32 and BF16 | FP32 | Per-channel | Both pass with exact fixture outputs. |
| FP32 and BF16 | BF16 | Per-channel | Both raise the CPU `Loops.h:311` `needs_dynamic_casting` internal assertion. |
| FP32 and BF16 | FP32 and BF16 | Per-tensor | All four combinations pass with exact fixture outputs. |

Primary PyTorch sources/tests were inspected at the installed Torch source commit:

- [`FakeQuantPerChannelAffine.cpp`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/quantized/FakeQuantPerChannelAffine.cpp#L51)
  accepts Float/BF16 scales at its top-level schema check.
- [`QuantizedOpKernels.cpp`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp#L2794)
  supplies CPU lambdas with a float scale; installed `cpu/Loops.h:304–311` rejects
  the required dynamic cast, matching the native failure.
- [`FakeQuantizeCore.cu`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/aten/src/ATen/native/quantized/cuda/FakeQuantizeCore.cu#L169)
  uses `gpu_kernel`; installed `cuda/CUDALoops.cuh:965` has a dynamic-casting
  path. This suggests an implementation route, not verified CUDA support.
- [`test_workflow_ops.py`](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/test/quantization/core/test_workflow_ops.py#L737)
  tests BF16 inputs with FP32 observer scales; its CPU scale-dtype test at
  lines 1064–1084 separately expects BF16-scale failure.

There is no real CUDA device in this environment. The regression suite includes
an explicit CUDA-availability skip for native/observer execution with CPU-stored
BF16 qparams, device moves and later CPU rejection. CUDA execution and integer NPU
correctness remain unverified. CPU reference parity is not target-NPU validation.

## Regression coverage and production changes

- `observers/test_execution_compatibility.py`: ten deterministic observer tests.
  Exact CPU BF16 snapshots pass MinMax/EMA derivation, installation, repeated model
  freeze/validation, strict state-dict restoration and whole-module CPU mapping.
  Pending extents, non-strict corruption, kernel rejection before dispatch,
  bypass, supported input/scale controls and unchanged snapshots are covered.
  The CUDA test uses only an explicit real-availability skip.
- `observers/test_frozen_state.py`: the former CPU-storage rejection assertion is
  replaced with storage acceptance and execution rejection; dtype and corruption
  coverage remains intact.
- `recipes/test_export_execution_preflight.py`: thirteen tests check rejection
  before forward/conversion, positional/keyword input routes, actual operand
  devices, all three token adapter families, dormant registered branches,
  unregistered readiness owners, unavailable capabilities, stale modes,
  NO_QUANT, bypass and the full-model writer. The BF16-input positive stage runs
  eagerly with BF16 parameters and FP32 scales before mocking conversion.
- `recipes/test_export_execution_circle.py`: two real Circle tests inspect exact
  supported and bypassed artifacts, reference output, unchanged source state,
  refreeze and strict equivalent-topology restoration followed by re-export.

The old writer's missing early check was isolated after correcting observer
storage, without modifying the working tree:

```bash
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
  /tmp/tico-bf16-old-writer-preflight-regression.py
```

Exit 1: the primary preflight regression failed with `RuntimeError not raised`;
the original `5575029` writer entered the converter mock. Loading only the old
writer in an isolated namespace separates that defect from its preceding import
rejection. The corrected writer rejects before the converter or forward spy.

Production Python files changed:

```text
tico/quantization/wrapq/observers/affine_base.py
tico/quantization/wrapq/utils/static_export.py
tico/quantization/recipes/export/circle.py
tico/quantization/wrapq/wrappers/ptq_wrapper.py
tico/quantization/wrapq/wrappers/nn/quant_linear.py
tico/quantization/wrapq/wrappers/nn/quant_embedding.py
tico/quantization/wrapq/wrappers/gemma4/quant_text_scaled_word_embedding.py
tico/quantization/wrapq/wrappers/llama/export_adapters.py
tico/quantization/wrapq/wrappers/gemma4/export_adapters.py
tico/quantization/wrapq/wrappers/qwen_vl/export_adapters.py
```

The new execution-site hook is a private opt-in contract; out-of-tree stages
without it retain the runtime guard. No checkpoint schema or frozen-grid dtype
policy changes, no new dependency and no numerical tolerances or skips are
broadened. Existing checkpoint topology requirements, conditional-path declarations
and pooler profile ownership remain unchanged.

## Validation evidence for this correction

All commands below run from the repository root. The first intentionally stopped
full-suite attempt and the later user-interrupted attempt are not counted as
passing runs. The initial standard static run found eight errors in the new test
fixture typing; precise casts/types and real `nn.Module` routing fixtures corrected
them without changing production behavior or weakening assertions.

Focused observer validation:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest \
  test.quantization.wrapq.observers.test_execution_compatibility \
  test.quantization.wrapq.observers.test_frozen_state \
  test.quantization.wrapq.observers.test_affine_base
```

Exit 0: 58 tests, one CUDA-availability skip, 0.624 seconds.

Focused lifecycle/consumer validation:

```bash
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest \
  test.quantization.wrapq.test_lifecycle \
  test.quantization.wrapq.test_conditional_entry_guard \
  test.quantization.wrapq.test_conditional_export_entries \
  test.quantization.wrapq.test_control \
  test.quantization.recipes.test_frozen_snapshot_consumers \
  test.quantization.test_public_interface
```

Exit 0: 60 tests, 1.943 seconds. Existing intentional unquantized-site warnings
were not suppressed.

Final typed preflight test fixture:

```bash
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest \
  test.quantization.recipes.test_export_execution_preflight -v
.venv/bin/python -m mypy --config-file mypy.ini --follow-imports=silent \
  test/quantization/recipes/test_export_execution_preflight.py
```

Both exit 0: thirteen tests pass in 0.202 seconds; the owning test file has no mypy
issues. Auxiliary raw narrow mypy commands followed unchanged MX imports and
reported four existing MX errors; the repository-standard invocation, which
excludes those paths, is the authoritative whole-repository check.

Real Circle conversion and independent inspection:

```bash
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python \
  /tmp/tico_bf16_circle_artifact_validation.py
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest \
  test.quantization.recipes.test_export_execution_circle
```

Both exit 0; two tests pass in 4.153 seconds. Artifacts and machine-readable
inspection are retained under `/tmp/tico-bf16-circle-artifacts/`.

| Property | Enabled FP32 per-channel scale | Disabled BF16 per-channel scale |
|---|---|---|
| Artifact bytes | 768, byte-identical to original `5575029` writer | 664 |
| Final operators | One `FULLY_CONNECTED`, no Q/DQ nodes | One `FULLY_CONNECTED`, no Q/DQ nodes |
| Input/output ABI | UINT8, `(1, 2)` | UINT8, `(1, 2)` |
| Input scale / ZP | `0.007843137718737125` / `127` | Same |
| Output scale / ZP | `0.007352941203862429` / `85` | Same |
| Weight | UINT8, scale `(0.125, 0.25)`, ZP `(1, 3)`, channel axis 0 | FLOAT32 without qparams |
| Structural verification issues | 0 | 0 |
| Eager / fake-quant reference maximum absolute difference | 0 | 0 |

The disabled policy intentionally emits one unquantized-weight QuantCheck warning.
Each source remains valid and unchanged after conversion/refreeze; strict loading
into equivalent topology and a second real export preserve the same contracts.
No identical-artifact promise is made between enabled and disabled policies.

Wrapper-smoke commands:

```bash
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u \
  -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case llama_decoder_layer_prefill --no-plot \
  --calibration-iters 1 --output-dir /tmp/tico-bf16-smoke-llama_decoder_layer_prefill
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u \
  -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case qwen3_vl_text_model --no-plot \
  --calibration-iters 1 --output-dir /tmp/tico-bf16-smoke-qwen3_vl_text_model
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -u \
  -m tico.quantization.examples.inspector \
  --config tico/quantization/examples/configs/wrapper_smoke.yaml \
  --mode wrapper-smoke --case gemma4_text_scaled_word_embedding --no-plot \
  --calibration-iters 1 --output-dir /tmp/tico-bf16-smoke-gemma4_text_scaled_word_embedding
```

All exit 0 with matching shapes and finite quantized outputs. These are eager
quantized-versus-float smoke metrics, with export disabled in the existing config:

| Case | Mean absolute difference | Maximum absolute difference | PEIR |
|---|---:|---:|---:|
| LLaMA decoder prefill | 0.014960 | 0.189073 | 0.030593 |
| Qwen text model | 0.067532 | 0.446470 | 0.076387 |
| Gemma scaled embedding | 0.000922 | 0.030796 | 0.038800 |

Related existing wrapper, pooler, PLE and recipe export tests:

```bash
PYTHONPATH=$PWD/test:$PWD OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m unittest \
  test.quantization.wrapq.wrappers.nn.test_quant_linear \
  test.quantization.wrapq.wrappers.nn.test_quant_embedding \
  test.quantization.wrapq.wrappers.gemma4.test_gemma4_ple_scale_fusion_integration \
  test.quantization.wrapq.wrappers.gemma4.test_pooler_export_ownership \
  test.quantization.wrapq.wrappers.gemma4.test_pooler_profile_policy \
  test.quantization.recipes.test_export_llama \
  test.quantization.recipes.test_export_gemma4 \
  test.quantization.recipes.test_export_qwen3_vl
```

Exit 0: 61 tests, 12.896 seconds. Existing host-table size-policy messages and
intentional QuantCheck warnings remain visible.

Repository-standard formatting/static and dependency checks:

```bash
source .venv/bin/activate
./ccex format --no-apply-patches
python -m pip check
git diff --check
```

All exit 0. UFMT/Pylint/mypy report `No lint issues`; dependency validation reports
`No broken requirements found`. All three new Python files carry the full standard
Apache 2.0 header, preserving the repository's copyright format. SHA256 snapshots
cover all fourteen changed Python files during final validation; documentation
updates do not alter those executable/test contents.

Final standard broad non-model suite:

```bash
source .venv/bin/activate
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ./ccex test
```

Exit 0: **4,760 tests in 1,398.032 seconds; 4,602 passed, 158 skipped**. The 157
existing top-level skips remain unchanged; the only added skip is the explicit
CUDA-availability test. Existing exclusions cover internal/opt-in tests,
unsupported kernels/operators, unused Conv3d wrappers, Torch-version graph
differences, absent local checkpoints, optional plotting, ONE and real >2 GiB
serialization. An inner deliberately skipped harness fixture in the log is not
an additional top-level skipped test.

Logs and recorded exit statuses:

```text
/tmp/tico-bf16-full-final.log
/tmp/tico-bf16-full-final.status
/tmp/tico-bf16-full-final-summary.json
/tmp/tico-bf16-format-final.log
/tmp/tico-bf16-format-final.status
/tmp/tico-bf16-final-validation-files.json
```

All fourteen changed Python file hashes still match the contents captured before
the final full/static runs. No production or test changes were made during those
runs. Only this follow-up's documentation was completed afterward. Final review
and repository searches found no remaining CPU-storage rejection, tracing bypass,
new catch-and-continue path, qparam casting/repair, introduced TODO or weakened
tolerance. Real CUDA, other Torch-version environments and integer NPU execution
were not available/run for this correction; no support claim is inferred from
metadata-only checks or CPU reference output.
