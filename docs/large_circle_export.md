# Large Circle export

TICO's ordinary conversion path can represent models whose constant payloads
make the complete Circle binary exceed 2 GiB. The converter does not expose an
extended-buffer API, a file-path argument to `build_circle`, or a model-family
specific serialization branch.

This is an in-memory serialization implementation. It is not an out-of-core or
streaming converter. The complete output still has to fit in the Python process's
address space and available memory. See [Memory and backend limits](#memory-and-backend-limits).

## Usage

Existing calls are unchanged:

```python
import tico

circle_model = tico.convert(model.eval(), example_inputs)
circle_model.save("model.circle")
```

The same selection applies to the other entry points:

```python
circle_model = tico.convert_from_exported_program(exported_program)
circle_model.save("model.circle")

circle_model = tico.convert_from_pt2("model.pt2")
circle_model.save("model.circle")
```

`build_circle()`, `convert_exported_module_to_circle()`, and
`CircleModel.circle_binary` continue to contain complete `bytes`. They never
contain just a FlatBuffer header, a pathname, or a lazy object. Serialization does
not create or open a file; the existing save APIs remain responsible for I/O.

A successful export does not establish that a particular compiler or interpreter
can import and execute the model. Backend validation is a separate step.

## Layout selection

The shared implementation is `tico/serialize/circle_binary.py`.
`build_circle()` delegates its final Object API packing to this module.
`tico.circle.io.model_to_bytes()` uses the same implementation for document
repacking and post-serialization optimization.

The serializer first measures the actual encoded `Buffer.data` byte vectors.
Packed low-bit tensors are measured after packing, not from their logical element
count or their original floating-point tensor size. Shared buffers are counted
once per buffer-table entry, rather than once per tensor reference.

For an ordinary small model, the existing inline FlatBuffer layout is retained.
The implementation selects an appended-payload layout in either of these cases:

- The aggregate buffer byte count approaches the FlatBuffer budget. The internal
  preflight reserves 16 MiB for metadata, so this is a conservative boundary,
  not a promise to keep every file below exactly 2 GiB inline.
- The actual bounded pack reaches its budget because of metadata, alignment,
  or many individually small buffers.

Only a size-limit exception selects the second representation. Invalid graph
values, schema errors, and `MemoryError` do not become a fallback path.

Consequently, two individually smaller constants whose combined data exceeds the
budget are supported, as is one constant larger than 2 GiB. The serializer does
not require any single buffer to exceed a per-buffer threshold.

### Appended payloads

The result is still one self-contained `.circle` file:

```text
+-------------------------------+
| CIR0 FlatBuffer               |
| - graph and tensor metadata   |
| - buffer table                |
| - small inline data           |
+-------------------------------+
| alignment padding             |
+-------------------------------+
| constant payload A            | <- Buffer.offset / Buffer.size
+-------------------------------+
| alignment padding             |
+-------------------------------+
| constant payload B            | <- Buffer.offset / Buffer.size
+-------------------------------+
```

The payloads are outside the FlatBuffer, not outside the file. Each appended
payload begins at a 16-byte-aligned, file-relative offset. `Buffer.offset` and
`Buffer.size` are unsigned 64-bit fields. Buffer 0 remains the empty sentinel;
empty and one-byte constants remain inline.

The serializer shallow-copies only the schema tables it needs to change. It does
not mutate the caller's buffer data or offsets. After a single header pack, it
patches the reserved offset fields in that finished header. The result of this
planning step is a `CircleBinaryLayout`: the finished header plus the payload
views and their file-relative offsets. Bytes-returning APIs join header, padding,
and contiguous payload views into the complete output bytes; file saves stream the
same layout instead (see [Streaming file saves](#streaming-file-saves)). There is
no intermediate per-payload `tobytes()` allocation on either path. Non-contiguous
data and Python byte-value sequences may still require a copy.

The FlatBuffer itself remains below its 32-bit budget. Constant relocation cannot
solve a graph whose metadata or non-relocatable custom options alone exceed that
budget; the serializer reports this case rather than producing a broken file.

## Streaming file saves

`CircleDocument.save()`, `tico.circle.io.save_model()`, and therefore the
`tico-circle extract` and `optimize` commands no longer materialize the complete
binary before writing it. The layout is planned first, so every `Buffer.offset`
in the header is final before the first byte is written and the output does not
need to seek. The writer then emits the header from its existing storage, at most
15 bytes of alignment padding per payload, and each payload view in bounded
chunks (8 MiB slices of the existing view, never a copy). Partial writes are
retried until complete, a stream that reports no progress raises an error, and
every write failure is reported; none is turned into a successful save.

Atomic saves still write a temporary file beside the destination, `fsync` it,
and `os.replace()` it. Any failure removes the temporary file and leaves the
existing destination untouched. Standard output receives the same stream;
diagnostics stay on standard error.

Which layout a file save produces:

| Aggregate `Buffer.data` bytes | `to_bytes()` / converter bytes | File save |
|---|---|---|
| ≤ 1 GiB | inline FlatBuffer | inline FlatBuffer (byte-identical to `to_bytes()`) |
| 1 GiB < total ≤ FlatBuffer budget | inline FlatBuffer | appended payloads |
| > FlatBuffer budget, or the bounded pack overflows | appended payloads | appended payloads |

Inline packing is not streaming: the FlatBuffers builder converts each payload
with `tobytes()`, copies it into a buffer that grows by doubling, and copies the
finished FlatBuffer once more, so packing a payload total of *P* bytes transiently
needs several times *P* (the extraction benchmark measured about 3 GiB of peak
RSS beyond the input for a 512 MiB inline output). Ordinary small
models keep the inline layout and their existing byte equivalence, and no output
that used to be inline becomes appended below 1 GiB. Above that private
streaming budget a file save selects the appended layout directly, which keeps
the transient cost to the small header. Bytes-returning APIs keep the full
FlatBuffer budget so their return contract and layout are unchanged; a model
between the two thresholds therefore saves as appended payloads but serializes to
inline bytes. The budget lowers the inline threshold only; it never raises the
FlatBuffer limit, and schema errors, invalid values, and `MemoryError` still do not
select a different layout.

## Mapped input and extraction ownership

`CircleDocument.load()` reads the whole input into a Python `bytes` object.
`CircleDocument.load_mapped(path)` instead maps a regular file read-only and
parses it through the same buffer-protocol loader: external buffer ranges are
validated with unbounded Python integers, only the FlatBuffer portion is copied
into a writable `bytearray`, and appended payloads become read-only NumPy views
of the mapping. Metadata remains fully editable. A file whose constants are all
inline still copies the complete FlatBuffer, including the inline constant bytes,
because that copy is what keeps metadata vectors writable; mapping helps files
that use appended payloads.

Ownership and lifetime:

- The mapped `CircleDocument` owns a `CirclePayloadMapping`. `release_payloads()`
  (or leaving a `with CircleDocument.load_mapped(...) as document:` block) drops
  that ownership. The file descriptor used for mapping is closed immediately after
  mapping; the mapping itself is unmapped when the owner releases it and no
  payload view is alive, otherwise when the last view is dropped. Releasing never
  invalidates live data.
- `clone()`, `copy.deepcopy()`, and `to_bytes()` never depend on the mapping.
- Public extraction (`extract_by_operator_indices`, `extract_by_tensor_indices`,
  `extract_by_tensor_patterns`) deep-copies graph and buffer-table metadata while
  *borrowing* payload storage, runs the existing selection, signature policy,
  dead-code elimination, and index compaction on the copy, and then copies only
  the payloads that survived (`PayloadOwnership.DETACHED`, the default). Discarded
  constants are never copied, buffers that alias one storage object are copied
  once, and the result does not pin the source `bytes` or mapping. The source
  document is not modified, and neither side sees the other's later edits.
- `PayloadOwnership.BORROWED` skips the final copy. The result shares the surviving
  payload storage with the source and is valid only while the source document and
  its backing storage are alive and unmodified. `tico-circle extract` uses this
  path: it owns the freshly mapped document exclusively, saves the result at once,
  and releases the mapping afterwards. It does not reimplement selection, cleanup,
  or compaction.
- Standard input, pipes, and other non-regular inputs keep the eager loader.

Writing the extraction result over its own input is supported. The atomic save
writes a sibling temporary file and replaces the destination only after the
complete output is written and synced; on POSIX the replaced inode stays readable
through the existing mapping, so payloads are read from the old file while the
new one is written. Symlink or hard-link aliases of the input behave the same way.
Non-atomic saves onto a mapped source are refused, because truncating the mapped
file would invalidate the pages still being read. Windows does not allow replacing
a mapped file, so the CLI falls back to the eager loader when the output names the
input; that branch is selected by platform and was not exercised on Windows here.

Memory semantics: mapping removes the input-sized Python heap allocation and the
payload copies, not the page cache. Pages the extraction reads or writes are
file-backed and count toward RSS while resident; the operating system may drop
them under pressure. Peak RSS therefore still scales with the amount of payload
actually written. Use `test/performance/benchmark_circle_extract_memory.py` to
measure a specific case rather than assuming a constant footprint.

## CircleDocument and default optimization

Default Circle O1 optimization remains enabled. There is no size-specific branch
that silently skips optimization.

When loading a complete binary, the document I/O layer validates external buffer
ranges, unpacks the writable header, and resolves external constants to NumPy
views of the original input bytes. It clears the old file-relative offset and size
fields in the Object API model. Existing constant inspection and graph passes can
therefore see the actual data, and a later pack calculates fresh offsets.

```python
from tico.circle.document import CircleDocument

document = CircleDocument.load("model.circle")
document.model.description = "Reviewed model"
document.save("reviewed.circle")
```

Changing header sizes or compacting buffer indices does not preserve stale file
positions. If an edited model becomes small enough, a later save can return to the
inline layout. Byte-identical layout is not promised for document transformations;
constant values and valid references are the relevant contract.

External payload views are read-only and retain ownership of their source bytes.
Metadata vectors remain writable. A transformation that must edit external payload
bytes in place must first make an explicit copy:

```python
buffer = document.model.buffers[buffer_index]
buffer.data = buffer.data.copy()
# Apply a semantically valid, dtype-aware transformation to buffer.data here.
```

Assigning newly computed `buffer.data` is also supported. A document clone or a
constant-folding pass may allocate additional large arrays; this change does not
make those operations out-of-core. Only the extraction workflow avoids copying
discarded constants; `optimize` still clones or folds constants as its passes
require.

Unresolved Object API offsets obtained by directly unpacking only a header are
rejected during repacking. Load the complete binary through `CircleDocument` or
`tico.circle.io.model_from_bytes()` instead. Truncated ranges, incomplete offset /
size pairs, external buffer 0, and simultaneous inline and external data are
rejected. External custom-operator-option references are explicitly unsupported
and rejected, rather than silently losing their payloads.

## Recipe format policies

Choosing `.circle` versus a host `.pt` artifact remains a recipe-level policy;
choosing inline versus appended storage inside a `.circle` file is now entirely
serializer-owned.

In particular, this change deliberately does not modify the existing Gemma4 PLE
`auto` policy, which chooses `.pt` for an oversized table. To request Circle for
that stage, use the already existing option:

```yaml
export:
  ple_embedding_format: circle
```

Once Circle is selected, it uses the ordinary converter and the same automatic
serializer as every other model. There is no Gemma-specific large-buffer API.
The real Gemma4 PLE checkpoint was not exercised in the accompanying validation.

## Memory and backend limits

This change removes the FlatBuffer-embedded-constant size bottleneck, not every
large-model limit:

1. **RAM:** bytes-returning APIs materialize the complete output. Source tensor
   storage, conversion intermediates, the original binary during O1, and an
   optimized replacement binary can coexist. Memory requirements can
   substantially exceed the final file size. A multi-gigabyte export should run
   in a sufficiently provisioned 64-bit process. File saves stream appended
   payloads, and `tico-circle extract` maps its input, but inline packing and
   any pass that copies constants still allocate in proportion to the payloads
   involved.
2. **Metadata:** graph metadata and inline custom options must still fit below
   the FlatBuffer limit. External custom-option serialization is not implemented.
3. **Consumers:** a compiler, interpreter, or NPU may impose smaller file,
   per-buffer, tensor-element-count, tensor-shape, or allocation limits. This
   implementation does not alter consumer integer widths or kernel indexing.
4. **Performance:** an export completing successfully does not establish import
   speed, runtime memory use, numerical parity, or device executability.

The opt-in real-size test writes a binary larger than 2 GiB and verifies its
payload digest. It is not a substitute for end-to-end target-backend testing.

## Tests

Use the development environment described in [Development Guide](development.md).
The focused suite is:

```bash
python -m unittest -v \
  test.unit_test.serialize.test_circle_binary \
  test.unit_test.circle.test_io \
  test.unit_test.circle.test_large_circle_roundtrip
```

The normal tests use a reduced private FlatBuffer budget to exercise the same
selection and packing paths with small arrays. Coverage includes aggregate size,
alignment, 64-bit offset arithmetic across 2 and 4 GiB, non-mutation, error
propagation, header-only copying on read, generated-schema round trips, packed
UINT4 metadata, and all three public conversion entry points with O1 enabled.
Extraction ownership, streaming saves, and mapped loading are covered by
`test.unit_test.circle.operations.test_extract`,
`test.unit_test.circle.test_document`, `test.unit_test.circle.test_cli`, and the
`LargeCircleExtractionRoundTripTest` class: detached and borrowed results, the
absence of discarded-payload copies, the absence of a joined binary on the
streaming path, partial and failed writes, atomic destination preservation, binary
standard output, the same-file and alias cases, and mapping release on success and
failure paths.

Real multi-gigabyte tests are opt-in. Use a high-memory host with sufficient free
disk space; at least 16 GiB RAM and 12 GiB free temporary disk space are a
practical starting point, not a bound on arbitrary model conversion memory. The
serialization test still needs the complete binary in RAM; the two extraction
round trips (several constants whose total exceeds 2 GiB, and one constant that
alone exceeds 2 GiB) write sparse file-backed fixtures and run the mapped CLI
path, then compare payload digests after reloading. They validate the file
extraction round trip only, not backend inference:

```bash
TICO_RUN_LARGE_CIRCLE=1 python -m unittest -v \
  test.unit_test.circle.test_large_circle_roundtrip.LargeCircleRealSizeTest
```

The reproducible memory measurement is described in
[`test/performance/README.md`](../test/performance/README.md#circle-extraction-memory-benchmark).

A separate optional ONE check executes a small model whose data is forced into
appended storage. It requires a working ONE interpreter installation:

```bash
TICO_RUN_ONE_LARGE_CIRCLE=1 python -m unittest -v \
  test.unit_test.circle.test_large_circle_roundtrip.LargeCircleConversionTest
```

Before merging, also run the existing serializer and Circle suites, the full
non-model suite, and repository formatting:

```bash
./ccex test -k circle
./ccex test
./ccex format
./ccex format --no-apply-patches
```

The delivery's `VALIDATION.md` distinguishes executed checks from tests that were
only added. Do not treat the existence of an opt-in test as evidence it was run.

## Format references

- [Circle schema, Buffer and Model definitions](https://github.com/Samsung/ONE/blob/master/res/CircleSchema/0.10/circle_schema.fbs)
- [FlatBuffers Python builder, version 25.2.10](https://github.com/google/flatbuffers/blob/v25.2.10/python/flatbuffers/builder.py)
- [TICO system design](design.md)
