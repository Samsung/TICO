# MediaPipe Palm Detector

This example reconstructs the MediaPipe palm detector as a PyTorch module,
quantizes it with TICO WrapQ, and exports static Circle models with an NHWC
input boundary.

The example is organized around four entry points:

```text
convert.py   Convert source TFLite weights and graph metadata to PyTorch artifacts.
export.py    Export floating-point or calibrated quantized Circle models.
analyze.py   Run reusable numerical quantization analyses.
verify.py    Verify torch.export and Circle artifacts.
```

Model-independent numerical analysis lives in `tico.quantization.analysis`.
The example only supplies palm-detector model loading, data normalization, and
output-boundary selection.

## Model interface

The exported model accepts one image tensor:

```text
shape:  [1, 192, 192, 3]
layout: NHWC
range:  [0, 1]
dtype:  float32 before quantized export
```

It returns:

```text
regressors:  [1, 2016, 18]
classifiers: [1, 2016, 1]
```

The PyTorch implementation uses NCHW internally. `NHWCInputAdapter` keeps the
external ABI explicit while Circle layout optimization removes redundant
internal layout transitions.

## Setup

Build and install TICO first:

```bash
./ccex build
./ccex install --cpu_only
```

Install the example dependencies:

```bash
python -m pip install -r examples/hand_detector/requirements.txt
```

Run commands from the repository root with `python -m`. This keeps package
imports stable and avoids depending on the current working directory.

## Convert the source TFLite model

```bash
python -m examples.hand_detector.convert \
  /path/to/hand_detector.tflite
```

The default outputs are:

```text
examples/hand_detector/hand_detector_spec.json
examples/hand_detector/hand_detector_float.pt
```

The converter currently supports the operator subset used by the supplied palm
detector. It is not a general TFLite-to-PyTorch frontend.

## Export Circle models

### Floating point

```bash
python -m examples.hand_detector.export float \
  --output examples/hand_detector/hand_detector_float.circle
```

The command verifies the NHWC input, layout optimization, and the two
`RESIZE_BILINEAR` operators unless `--skip-verification` is supplied.

### Quantized

Use representative tensors produced by the same preprocessing path as runtime
inputs:

```bash
python -m examples.hand_detector.export quantized \
  --calibration-dir /path/to/calibration_npy \
  --bits 8 16
```

The quantization policies are:

| Tensor role | UINT8 | INT16 |
|---|---|---|
| Image and activations | per-tensor asymmetric | per-tensor symmetric |
| Conv/depthwise weights | per-channel asymmetric | per-channel symmetric |
| PReLU slope | per-channel asymmetric | per-channel symmetric |
| Convolution bias | INT32 | INT64 |

Synthetic inputs are available only for smoke tests:

```bash
python -m examples.hand_detector.export quantized \
  --synthetic-calibration-samples 32 \
  --bits 8
```

## Quantization analysis

### A/B/C/D/E ablation

The standard profiles isolate the major quantization error sources:

```text
A  output-only
B  weight-only with floating-point activations and outputs
C  internal activation-only with floating-point weights and outputs
D  full quantization
E  internal full quantization with floating-point outputs
```

Run all five profiles from one calibrated candidate:

```bash
python -m examples.hand_detector.analyze ablation \
  --calibration-dir /path/to/calibration_npy \
  --evaluation-dir /path/to/evaluation_npy \
  --bits 8
```

The same API is reusable from Python:

```python
from tico.quantization.analysis import (
    QuantizationAblation,
    QuantizationBoundaries,
    QuantizationProfile,
    SiteSelector,
)
```

A model adapter only needs to define which observer sites represent final model
outputs. Parameter and internal-activation profiles are derived from observer
roles.

### Output clipping

Compare MinMax, fixed percentile, and calibration-L1 clipping while leaving all
internal model computation in floating point:

```bash
python -m examples.hand_detector.analyze output-clipping \
  --calibration-dir /path/to/calibration_npy \
  --evaluation-dir /path/to/evaluation_npy \
  --bits 8
```

This reports calibration and evaluation MAE, selected ranges, affine qparams,
saturation, and integer-code utilization. The L1 candidate is selected only
from calibration outputs.

### Activation observer sweep

Keep per-channel MinMax weight quantization fixed and compare activation range
estimators across three ablation profiles:

```text
C  activation-only with floating-point weights and outputs
D  full quantization
E  internal full quantization with floating-point outputs
```

```bash
python -m examples.hand_detector.analyze observer-sweep \
  --calibration-dir /path/to/calibration_npy \
  --evaluation-dir /path/to/evaluation_npy \
  --bits 8 \
  --percentiles 99.9 99.99 99.999
```

`PercentileObserver` uses bounded sampling, so it does not retain every value
from every activation tensor. The console ranks candidates by E regressor MAE,
which isolates internal W8A8 behavior from the selected final-output domains.
The JSON report stores C/D/E metrics under each candidate's `profiles` mapping.
For compatibility with existing report readers, the candidate-level `outputs`
field remains an alias of D outputs.

### Activation block sensitivity

Rank semantic activation-domain groups from a percentile-calibrated
E:internal-full baseline. Weight quantization remains enabled, final-output
domains remain floating point, and one internal activation group is disabled at
a time:

```bash
python -m examples.hand_detector.analyze activation-sensitivity \
  --calibration-dir /path/to/npy \
  --calibration-offset 0 \
  --calibration-limit 200 \
  --evaluation-dir /path/to/npy \
  --evaluation-offset 200 \
  --evaluation-limit 79 \
  --require-disjoint \
  --bits 8 \
  --percentile 99.99 \
  --score-output regressors \
  --top-k 20
```

Groups follow logical activation domains rather than individual wrappers.
Producer `act_out` and downstream consumer `act_in` observers for the same
tensor are assigned to the producer block where possible. The report includes
the E baseline, all group metrics, matched observer paths, operation indices,
and regressor/classifier MAE recovery. Positive recovery means that leaving the
group floating point improved the selected baseline. Independent gains are not
additive because changing one block changes the sensitivity of later blocks.

Evaluate an explicit cumulative path in the supplied order:

```bash
python -m examples.hand_detector.analyze activation-sensitivity \
  --calibration-dir /path/to/npy \
  --evaluation-dir /path/to/npy \
  --bits 8 \
  --percentile 99.99 \
  --strategy cumulative \
  --groups \
    stem \
    feature_block_00 \
    feature_block_03 \
    feature_block_04 \
    feature_block_10 \
    feature_block_28
```

Accumulate the initial independent ranking without recomputing it after each
selection:

```bash
python -m examples.hand_detector.analyze activation-sensitivity \
  --calibration-dir /path/to/npy \
  --evaluation-dir /path/to/npy \
  --bits 8 \
  --percentile 99.99 \
  --strategy ranked \
  --max-steps 5
```

Run greedy cumulative selection, re-ranking the remaining groups after every
step:

```bash
python -m examples.hand_detector.analyze activation-sensitivity \
  --calibration-dir /path/to/npy \
  --evaluation-dir /path/to/npy \
  --bits 8 \
  --percentile 99.99 \
  --strategy greedy \
  --max-steps 5 \
  --minimum-improvement 0
```

Ranked search pays for one independent sweep and then follows that fixed
ranking. Greedy search evaluates every remaining candidate at each step. Use
`--groups` to restrict either candidate pool when a full search is too
expensive. The JSON report stores cumulative, ranked, or greedy results under
`steps`; every step includes the newly added group, all selected groups and
sites, incremental recovery, and total recovery from the E baseline.

### Group-specific activation observer overrides

Keep a global activation policy and independently replace only the logical
activation domains assigned to selected sensitivity groups:

```bash
python -m examples.hand_detector.analyze group-observer-sweep \
  --calibration-dir /path/to/npy \
  --calibration-offset 0 \
  --calibration-limit 200 \
  --evaluation-dir /path/to/npy \
  --evaluation-offset 200 \
  --evaluation-limit 79 \
  --require-disjoint \
  --bits 8 \
  --global-percentile 99.99 \
  --percentiles 99.9 99.95 99.99 99.995 99.999 \
  --groups \
    stem \
    feature_block_00 \
    feature_block_04 \
    feature_block_13 \
    feature_block_28 \
  --score-output regressors \
  --report-json examples/hand_detector/reports/group_observer_sweep.json
```

For each group, all other activation domains remain on the global percentile.
The sweep compares the unchanged global policy, MinMax, and the requested
percentiles under E:internal-full. The candidate equal to the global percentile
is kept as the no-override control and is not recalibrated. Use
`--skip-minmax` to evaluate only percentile overrides.

Observer overrides are applied through exact floating-point `PTQConfig` paths,
not prepared wrapper paths containing `.wrapped`. Each policy candidate is
prepared and calibrated independently, and the command verifies that every
requested path instantiated the expected observer class before evaluation.

Group results are independent. Combining each group's individually best policy
requires a separate validation run because observer choices may interact across
blocks. The JSON report includes every override path and may retain the global
policy as the best candidate when all overrides are worse.

## Calibration and evaluation data

Supported NumPy shapes are:

```text
[192, 192, 3]
[1, 192, 192, 3]
[3, 192, 192]
[1, 3, 192, 192]
```

Integer arrays are converted to float32 and divided by 255. Floating-point
arrays are assumed to already use the model input range.

Calibration and evaluation may point to the same directory for numerical-floor
analysis, but policy selection and final reporting should use disjoint data.
Use offsets to split a naturally sorted directory:

```bash
python -m examples.hand_detector.analyze observer-sweep \
  --calibration-dir /path/to/npy \
  --calibration-offset 0 \
  --calibration-limit 200 \
  --evaluation-dir /path/to/npy \
  --evaluation-offset 200 \
  --evaluation-limit 79 \
  --require-disjoint
```

For frames extracted from video, split by source video or capture session rather
than adjacent frame number.

## Verification

Verify the PyTorch export graph:

```bash
python -m examples.hand_detector.verify torch
```

Verify a floating-point Circle model:

```bash
python -m examples.hand_detector.verify circle \
  examples/hand_detector/hand_detector_float.circle
```

Verify a quantized Circle model:

```bash
python -m examples.hand_detector.verify quantized \
  examples/hand_detector/exported/hand_detector_uint8.circle \
  --bits 8
```

## Internal support modules

Implementation helpers are under `_support/` and are not separate user-facing
commands:

```text
_support/circle.py
_support/conversion.py
_support/data.py
_support/group_observer_sweep.py
_support/quantization.py
_support/sensitivity.py
_support/tflite_flatbuffer.py
_support/verify_circle_layout.py
_support/verify_circle_resize.py
_support/verify_quantized_circle.py
```

## Tests

Run reusable analysis tests:

```bash
python -m unittest discover -s test/quantization/analysis -v
python -m unittest discover -s test/quantization/wrapq -p "test_control.py" -v
python -m unittest discover -s test/quantization/wrapq/observers \
  -p "test_percentile.py" -v
```

Run the model example tests:

```bash
python -m examples.hand_detector.test_hand_detector
```

See `docs/layout_optimization.md` for the Circle layout-region optimization design
and `THIRD_PARTY_NOTICES.md` for source-model attribution.


## Activation block reconstruction

The first reconstruction stage keeps W8 parameters fixed and optimizes
per-tensor activation scale and zero-point against cached FP32 block outputs.
Blocks are processed in model execution order and are evaluated under
E (`internal-full`) after every committed block.

```bash
python -m examples.hand_detector.analyze block-reconstruction \
  --calibration-dir /path/to/npy \
  --calibration-offset 0 \
  --calibration-limit 200 \
  --evaluation-dir /path/to/npy \
  --evaluation-offset 200 \
  --evaluation-limit 79 \
  --require-disjoint \
  --bits 8 \
  --percentile 99.99 \
  --max-samples 524288 \
  --groups stem feature_block_00 feature_block_04 \
  --steps 500 \
  --report-json examples/hand_detector/reports/block_reconstruction.json
```

The cache stores both FP32 and quantized-prefix inputs. PR 1 uses the
quantized-prefix input and normalized block-output MSE; QDrop and adaptive
weight rounding are intentionally left for follow-up changes.
