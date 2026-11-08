# AdaRound

This package learns hard floor/ceil decisions for calibrated Conv2d weights while
keeping weight scale, zero-point, bit width, granularity, channel axis, and all
activation qparams fixed.

Reconstruction consumes committed observer snapshots. Learned-scale and discrete
code proxies are temporary optimizer state and must be finalized into the original
observers before model freezing or export. Rollback restores the original common
snapshot, including calibration provenance, lock, and fake-quantization policy.

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

The [hand-model AdaRound export workflow](../../../../examples/hand_detector/README.md#adaround-circle-export)
owns the application CLI, diagnostic/deployment profiles, artifact verification,
and sidecar manifest. It exports the accepted hard-rounded weights through the
existing WrapQ/Circle path described above.
