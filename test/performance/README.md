# Performance Benchmarks

This directory contains explicit, opt-in performance scripts. They are not part of the
model-independent unit-test suite.

## Conversion speed and model size

The existing model benchmark covers the performance requirements documented in
`docs/system_test.md`:

```bash
python3 -m test.performance.benchmark_perf
```

It uses the configured Llama baseline models and therefore requires their model-test
dependencies.

## Full Circle O1 scheduler benchmark

Use a locally generated or downloaded full `.circle` artifact to compare the former
restart scheduler with O1's round-based fixed-point scheduler:

```bash
python3 -m test.performance.benchmark_circle_optimizer \
  model.circle \
  --repeat 3
```

The benchmark clones the input for every run, verifies each output, and requires the
two scheduler variants to produce byte-identical Circle binaries. It reports elapsed
time, pass-execution counts, the reduction in pass invocations, and the output SHA-256.
No model artifact is stored in this repository.

Heavy constant folding and optional O1 transforms can be selected explicitly:

```bash
python3 -m test.performance.benchmark_circle_optimizer \
  model.circle \
  --constant-folding-profile heavy \
  --fuse-transpose-conv-slice \
  --json
```
