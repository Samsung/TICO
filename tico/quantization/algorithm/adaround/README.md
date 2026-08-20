# AdaRound

This package learns hard floor/ceil decisions for calibrated Conv2d weights while
keeping weight scale, zero-point, bit width, granularity, channel axis, and all
activation qparams fixed.

The optimizer uses soft rounding during gradient updates and hard rounding for
checkpoint selection, acceptance, export, and reporting. The selected hard weights are
written back as dequantized affine-grid values, so the existing WrapQ weight observer
and Circle export path remain unchanged.

The validation lifecycle is:

```text
quantized-prefix block cache
-> soft AdaRound optimization
-> hard checkpoint selection
-> separate acceptance-set commit or rollback
-> external evaluation
```

Only `torch.nn.Conv2d` weights with affine per-output-channel qparams and
`channel_axis=0` are supported. PReLU slopes, bias tensors, activation qparams, and
weight scales/zero-points are not modified.

## Full-integer Circle export

The hand-detector AdaRound CLI can export the selected model in the same process so
accepted hard-rounded weights and replayed activation qparams are not lost. Circle
export always enables every fake-quant site, including graph outputs, and therefore
uses the `D:full` deployment profile. The numerical AdaRound table continues to report
`E:internal-full`, which excludes graph-output fake quantization for diagnostics.

```bash
python -m examples.hand_detector.analyze adaround \
  ... \
  --groups feature_block_28 \
  --steps 2000 \
  --export-circle examples/hand_detector/exported/hand_detector_uint8.circle
```

Unless verification is skipped, export validates UINT8/INT16 graph I/O, per-channel
Conv2d weights, quantized biases, the absence of `DEQUANTIZE`, and the optimized NHWC
layout. A sidecar manifest records both E and D evaluation metrics, accepted AdaRound
windows, the artifact SHA-256, and verification summaries.
