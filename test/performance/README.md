# Performance Benchmarks

This directory contains explicit, opt-in performance scripts. They are not part of the
model-independent unit-test suite.

## Full conversion memory benchmark

`benchmark_conversion_memory.py` compares the existing `tico.convert(...).save(...)`
bytes path with `tico.convert_to_file(...)`. It constructs deterministic FP32 embedding
tables, touches every source-weight page, and sums a few lookups so that large constants
survive the complete conversion while activations stay small. No checkpoint, network,
GPU, or external runtime is required.

```bash
python3 -m test.performance.benchmark_conversion_memory \
  --sizes-mib 16,64,256,512 --repeat 3 --optimization O1

# Isolate the effect without Circle O1, retaining the same FX conversion passes.
python3 -m test.performance.benchmark_conversion_memory \
  --sizes-mib 16,64,256,512 --repeat 3 --optimization O0 --json > o0.json
```

Every size, mode, and repetition runs sequentially in a fresh subprocess. Peak RSS is
the process high-water mark, including imports, source weights, export, conversion, and
saving; it is not incremental Python allocation. The timer covers the conversion API
and output save, including the file path's atomic save, and excludes model construction,
hashing, and validation. `--threads` defaults to one for reproducibility. The report
includes the source constant size, output size, elapsed time, peak RSS before conversion,
and full conversion peak RSS. Use several repetitions because allocator and operating
system noise can matter, especially for small models with a large import-time baseline.

Each output is hashed in bounded chunks. A separate subprocess loads it through
`CircleDocument.load_mapped()`, verifies its structure, compares three deterministic
reference-runtime probes to PyTorch with explicit FP32 tolerances, and fingerprints all
metadata and buffer payloads. Metadata fingerprints normalize only physical buffer
offsets and sizes, allowing exact constant and metadata equivalence when the file path
uses appended storage and the complete file hashes differ. Validation allocations do
not enter the measured conversion RSS. This does not test a target backend or predict
the memory cost of different graph shapes and optimization passes.

An unchanged `main` checkout can serve as an additional bytes baseline using the same
Python interpreter and installed dependencies:

```bash
git worktree add --detach /tmp/tico-main main
python3 -m test.performance.benchmark_conversion_memory \
  --baseline-root /tmp/tico-main --sizes-mib 16,64,256,512 \
  --repeat 3 --optimization O1 --json > conversion-memory.json
```

`--modes file` measures only the explicit file path for larger models. Sizes are requested
aggregate source-weight MiB, rounded down to complete rows across `--tables` tables of
`--width` features; the actual byte count is reported. `--flatbuffer-limit` lowers the
private serialization limit for a small appended-layout experiment; omit it for actual
large-file measurements. Default temporary artifacts are removed. `--workdir PATH`
retains small per-run reports; add `--keep-outputs` to retain the `.circle` and numerical
probe files too. Allow sufficient disk space for the selected sizes and retained outputs.

Use `--input-kind pt2` to measure conversion from an already exported program, including
the `.pt2` load in the timed API call:

```bash
python3 -m test.performance.benchmark_conversion_memory \
  --input-kind pt2 --sizes-mib 256,512,2304 --modes file \
  --baseline-root /tmp/tico-main --optimization O1 --json > pt2-memory.json
```

For each size, an unmeasured subprocess exports one shared `.pt2` fixture and saves the
small expected numerical probes and a source-size manifest. Fresh measured workers use
`tico.convert_from_pt2(...).save(...)` or `tico.convert_from_pt2_to_file(...)` without
constructing a source module. The fixture is shared by all modes and repetitions, then
deleted unless `--keep-outputs` is selected. `--workdir` retains its small probe file and
manifest; default temporary directories are removed completely. A matching prepared
fixture in `--workdir` is reused. Reports include `input_kind`, the artifact size, and
the setup scope. This distinguishes the module API's export high-water mark from the saved-program
workflow; `module` remains the default and includes export in its peak RSS and timing.

`--identity-mul` adds an internal multiply by one followed by negation. This optional
sensitivity case can exercise Circle O1 rewrites while all large tables remain live;
the default fixture is unchanged. Compare `validation.operator_count` between O0 and O1
to confirm that Circle optimization changed the serialized graph, since a preceding FX
pass could otherwise remove a candidate rewrite. Use a separate work directory for this
fixture. For example, add `--identity-mul --sizes-mib 2304` to the saved-program command
above after confirming the operator-count change on a small model.

Compare the reported peak RSS for the same model and optimization setting to quantify
RAM reduction. Sweeping increasing sizes under a fixed memory limit can measure the
largest model that completes; this benchmark does not extrapolate that limit from a
single successful run. See [Large Circle export](../../docs/large_circle_export.md) for
the file-output API, layout selection, ownership, and remaining memory limits.

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
