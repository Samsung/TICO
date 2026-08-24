# Legal UINT8/INT16 mixed precision

This workflow searches a deployment-legal precision topology for the hand detector.
It does not replay checkpoints produced by the earlier W8/A16 profile because that
profile assigns a different dtype to Conv2D/DepthwiseConv2D weights and activations.

## Operator contract

Each semantic region is assigned one of two precision domains:

- `uint8`: data input/output and Conv2D, DepthwiseConv2D, or PReLU parameter use
  UINT8.
- `int16`: data input/output and Conv2D, DepthwiseConv2D, or PReLU parameter use
  symmetric INT16.

Conv and PReLU parameters remain per-channel while activations remain per-tensor.
An explicit `QuantStub` models a dtype-changing edge between two semantic regions.
Final concatenations use one shared observer for every input and the output.

## Search stages

The command performs the following steps with fresh calibration for every topology:

1. Evaluate the legal all-INT16 internal floor.
2. Evaluate the legal all-UINT8 internal floor.
3. Demote each variable semantic region from INT16 to UINT8 independently.
4. Run constrained reverse-greedy or reverse-beam search from the all-INT16 entry.
5. Write the selected precision map as a compact JSON artifact.

The regressor and classifier output domains are fixed independently. Their defaults
are INT16 and UINT8, respectively.

## Example

```bash
python -m examples.hand_detector.analyze legal-mixed-precision \
  --calibration-dir ~/test/convert-tflite2tvn/input/npy \
  --calibration-offset 0 \
  --calibration-limit 200 \
  --evaluation-dir ~/test/convert-tflite2tvn/input/npy \
  --evaluation-offset 200 \
  --evaluation-limit 79 \
  --require-disjoint \
  --uint8-percentile 99.99 \
  --int16-observer minmax \
  --search reverse-beam \
  --beam-width 4 \
  --target-regressor-mae 0.1 \
  --target-classifier-mae 0.1 \
  --report-json examples/hand_detector/reports/legal_mixed_precision.json \
  --assignment-json examples/hand_detector/reports/legal_precision_map.json
```

The full search can require many complete calibration/evaluation passes. Use
`--candidate-count` and `--max-search-steps` for an initial screening run, then set
both to zero for an unrestricted run.

## Cost model

Search ranks target-feasible assignments using a normalized cost:

```text
parameter_weight * INT16 parameter fraction
+ activation_weight * INT16 activation fraction
+ boundary_weight * dtype-transition fraction
```

The JSON report also records raw parameter bytes, activation bytes, transition count,
operator contract validation, output metrics, and the complete precision map.
