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
