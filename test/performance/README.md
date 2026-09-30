# Performance Benchmarks

This directory contains explicit, opt-in performance scripts. They are not part of the
model-independent unit-test suite.

## Conversion speed and model size

The existing model benchmark covers the performance requirements documented in
`docs/system_test.md`:

```bash
./ccex test -p
# Direct equivalent:
python3 -m test.performance.benchmark_perf
```

`./ccex test -p` selects this threshold benchmark. It uses the configured Llama baseline models and therefore requires their model-test
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

The scheduler comparison requires a caller-provided full Circle artifact and is run
directly; it is not selected by `./ccex test -p` and has no repository pass/fail
threshold. See [`tico/circle/README.md`](../../tico/circle/README.md) for O1 pass
selection and scheduling semantics.

## Circle extraction memory benchmark

`benchmark_circle_extract_memory.py` generates a synthetic appended-layout fixture in the
driver process and then measures three extraction paths, each in its own subprocess so
that one run's peak RSS cannot leak into the next:

- `legacy`: eager `bytes` load, whole-document deepcopy, complete output bytes joined in
  memory (the behaviour before metadata-first extraction and streaming saves);
- `public`: eager load, the public detached extraction API, streaming save;
- `cli`: `tico-circle extract` with a read-only mapping, borrowed payloads, and a
  streaming save.

Each path runs a `keep-one` scenario (one constant survives) and a `keep-most` scenario
(all but one survive):

```bash
# 8 x 128 MiB constants with a lowered private FlatBuffer budget, so the 1 GiB
# fixture already uses the appended layout.
python3 -m test.performance.benchmark_circle_extract_memory \
  --buffers 8 --buffer-mib 128 --flatbuffer-limit 268435456 --tracemalloc

# Real >2 GiB behaviour without touching the budget; needs the disk and RAM for
# the legacy path's in-memory copies.
python3 -m test.performance.benchmark_circle_extract_memory \
  --buffers 5 --buffer-mib 512 --modes public,cli
```

Peak RSS comes from `getrusage` and includes file-backed pages the mapping touched, so
the `cli` figure grows with the amount of payload actually read and written; it is not a
Python-heap number. `--tracemalloc` adds the Python allocator's peak as a secondary
figure that excludes mapped pages. Outputs below the streaming inline budget still show
the FlatBuffer builder's inline packing cost in every mode; see
[`docs/large_circle_export.md`](../../docs/large_circle_export.md#streaming-file-saves).
