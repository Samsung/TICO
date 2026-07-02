# Development Guide

This guide covers setting up a development environment, running tests, and formatting
code. All developer workflows go through the `./ccex` helper script in the repository
root.

## Table of Contents

- [Environment setup](#environment-setup)
- [Testing](#testing)
  - [Run all tests](#run-all-tests)
  - [Run a subset](#run-a-subset)
  - [Debugging aids](#debugging-aids)
  - [Model tests](#model-tests)
  - [Runtime selection](#runtime-selection)
- [Code formatting](#code-formatting)

## Environment setup

Run the commands below to configure the testing and/or formatting environment:

```bash
./ccex configure          # testing & formatting environment
./ccex configure format   # formatting environment only
./ccex configure test     # testing environment only
```

> [!NOTE]
> `./ccex configure test` installs `TICO` in editable mode. Use
> `./ccex configure test --dist` to install from the built wheel instead.

**Available options**

| Option | Description |
|---|---|
| `--torch_ver <ver>` | Torch (and torch-family packages, e.g. torchvision) version to install: a family (`2.5` ~ `2.10`), an exact version (e.g. `2.7.0+cu118`), or `nightly`. Default: `2.7` |
| `--cuda_ver <maj.min>` | Override the detected CUDA version (e.g. `12.1`) |
| `--cpu_only` | Force a CPU-only installation |

```bash
./ccex configure test                    # stable 2.7.x (default)
./ccex configure test --torch_ver 2.6    # stable 2.6.x
./ccex configure test --torch_ver nightly
```

## Testing

### Run all tests

```bash
./ccex test
# OR
./ccex test run-all-tests
```

> [!NOTE]
> Unit tests don't include model tests — see [Model tests](#model-tests).

### Run a subset

To run a subset of `test.modules.*`, use `./ccex test -k <keyword>`:

```bash
# Tests in a specific sub-directory (op/, net/, ...)
./ccex test -k op
./ccex test -k net

# Tests in one file (single/op/add, single/op/sub, ...)
./ccex test -k add
./ccex test -k sub

# The SimpleAdd test in test/modules/single/op/add.py
./ccex test -k SimpleAdd
```

### Debugging aids

To see the full debug log, add `-v` or set `TICO_LOG=4`:

```bash
TICO_LOG=4 ./ccex test -k add
# OR
./ccex test -v -k add
```

To dump the intermediate torch graph IR as `.png`, set `TICO_GRAPH_DUMP=1`:

```bash
TICO_GRAPH_DUMP=1 ./ccex test -k add
# Images are dumped into $(pwd)/.tico_temp
```

### Model tests

To run model tests locally, install each model's dependencies first, then run the tests
one by one:

```bash
pip install -r test/modules/model/<model_name>/requirements.txt

# Run a single model
./ccex test -m <model_name>

# Run models whose names contain "Llama" (Llama, LlamaDecoderLayer, LlamaWithGQA, ...)
# Note: quote the wildcard(*) pattern
./ccex test -m "Llama*"
```

For example:

```bash
./ccex test -m InceptionV3
```

### Runtime selection

By default, `./ccex test` runs all modules with the `circle-interpreter` engine. You can
run tests with the `onert` runtime instead.

**Supported runtimes**

- `circle-interpreter` (default) — uses the Circle interpreter for inference.
- `onert` — uses the ONERT package for inference; useful when the Circle interpreter
  cannot run a given module.

**0. Install ONERT**

```bash
pip install onert
```

**1. Command-line flag**

```bash
# Run with the default circle-interpreter
./ccex test

# Run all tests with onert
./ccex test --runtime onert
# or
./ccex test -r onert
```

**2. Environment variable**

```bash
# Temporarily override for one command
CCEX_RUNTIME=onert ./ccex test

# Persist in your shell session
export CCEX_RUNTIME=onert
./ccex test
```

## Code formatting

Install the formatting requirements:

```bash
./ccex configure format
```

Run the formatter:

```bash
./ccex format
```

## See also

- [System design](./design.md) — architecture, pass pipeline, and behavior design
- [Requirements](./requirements.md) — functional and non-functional requirements
- [Quantization recipes developer guide](../tico/quantization/recipes/README.md) — adding adapters, stages, and configs
