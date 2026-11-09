# NHWC input and Circle-side layout-region optimization

The PyTorch detector remains an ordinary NCHW implementation. Circle export uses
`NHWCInputAdapter` to expose an NHWC image input with shape
`[1, 192, 192, 3]`.

The adapter places the input `QuantStub` before the NHWC-to-NCHW permutation.
Consequently, UINT8 and INT16 exports attach the input affine qparam directly to
the NHWC Circle input tensor.

## Circle optimization pipeline

`tico.convert()` enables Circle O1 by default through
`CompileConfigV1.circle_optimize=True`; O1 already includes
`EliminateTransposeBoundedLayoutRegionPass`. After conversion, the hand-detector
example applies this additional Circle-to-Circle pipeline:

```text
FuseActivationFunctionPass
EliminateTransposeBoundedLayoutRegionPass
SimplifyViewOpsPass
DeadCodeEliminationPass
CompactIndicesPass
```

The example's post-processing pass manager uses `CirclePassStrategy.RESTART`.
A rewrite can expose a new inverse Transpose pair or dead object, and restart
scheduling allows the earlier passes to run again until the complete sequence
reaches a fixed point. The converter's O1 pipeline uses separate
[complete-round scheduling](../../../tico/circle/README.md#rule-and-pipeline-scheduling).

`EliminateTransposeBoundedLayoutRegionPass` finds connected Circle regions of
supported layout-convertible operators. The detector's regions use:

```text
ADD
PAD with a constant rank-by-two padding tensor
```

The full supported operator set is documented in the
[Circle optimization reference](../../../tico/circle/README.md#optimize).

Every external data input to a candidate region must enter through the same
Transpose permutation. Every external data output must leave through the
inverse permutation. The pass then executes the complete region directly in the
source layout:

```text
source-layout tensors
  -> Transpose(P)
  -> ADD/PAD region
  -> Transpose(P^-1)
  -> source-layout tensors
```

becomes:

```text
source-layout tensors
  -> ADD/PAD region
  -> source-layout tensors
```

For PAD, the rank-by-two constant is cloned and its rows are reordered to match
the source-layout axes. Cloning avoids changing another operator that may share
the original constant buffer.

The detector contains five such regions:

- three downsample residual regions containing PAD and ADD;
- two decoder regions containing two connected ADD operators and an NHWC side
  path between them.

Together, the regions bypass all 19 Circle Transpose operators. Dead-code
elimination removes the unused Transpose nodes, and index compaction removes the
unused permutation and padding constants.

## Safety conditions

A region is rewritten only when all of the following conditions hold:

- every internal operator is supported by the region pass;
- binary ADD inputs and outputs have identical shapes, so broadcasting is not
  involved;
- PAD uses a constant INT32 tensor with shape `[rank, 2]`;
- every input boundary uses one common permutation;
- every output boundary uses the inverse permutation;
- no region-layout tensor is exposed directly as a graph or signature output;
- every unsupported external consumer is separated by the expected inverse
  Transpose;
- every boundary Transpose preserves tensor type and affine qparams;
- quantized activations have exactly one scale and one zero point;
- every region data tensor is unquantized or uses per-tensor activation qparams.

Per-channel activation qparams are deliberately rejected. The target activation
policy is per-tensor, so the rewrite never has to remap a quantized axis.
Conv2D and DepthwiseConv2D weight qparams remain per-channel and are unrelated
to this activation-layout transformation.

## Export

Floating-point export:

```bash
python -m examples.hand_detector.export float \
  --output examples/hand_detector/exported/hand_detector_float.circle
```

Quantized export:

```bash
python -m examples.hand_detector.export quantized \
  --calibration-dir /path/to/calibration_npy \
  --bits 8 16 \
  --output-dir examples/hand_detector/exported
```

Calibration and evaluation arrays are normalized to NHWC by
[`_support/data.py`](../_support/data.py).
The accepted source shapes remain:

```text
[192, 192, 3]
[1, 192, 192, 3]
[3, 192, 192]
[1, 3, 192, 192]
```

## Verify the exported graph

```bash
python -m examples.hand_detector.verify circle \
  examples/hand_detector/exported/hand_detector_float.circle
```

The verifier checks:

- one Circle input with shape `[1, 192, 192, 3]`;
- no consecutive inverse Transpose pair;
- no remaining Transpose-ADD-Transpose round trip;
- exactly zero Circle Transpose operators;
- two `RESIZE_BILINEAR` operators with `align_corners=False` and
  `half_pixel_centers=True` by default.

## Use the Circle pass from the CLI

To run the same pass explicitly on an existing Circle artifact:

```bash
tico-circle optimize input.circle \
  --passes \
eliminate-transpose-bounded-layout-region,simplify-view-ops,dce,compact \
  --strategy restart \
  -o output.circle
```

Changing the external input ABI remains an explicit model-authoring decision via
`NHWCInputAdapter`. The Circle pass performs only semantics-preserving internal
graph rewrites.
