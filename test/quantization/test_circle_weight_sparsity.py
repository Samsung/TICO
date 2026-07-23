# Copyright (c) 2026 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import tico.quantization.circle_weight_sparsity as circle_sparsity
from tico.quantization.circle_weight_sparsity import (
    _count_tensor_semantic_zeros,
    _deduplicate_across_files,
    aggregate_circle_weight_stats,
    analyze_circle_binary,
    CircleWeightSparsityReport,
    CircleWeightSparsityRow,
    CircleWeightTensorStats,
    discover_circle_files,
    QuantizationInfo,
    render_csv,
    render_json,
    render_markdown,
)


class _TensorTypeValues:
    """Provide the tensor type values used by fake Circle models."""

    FLOAT32 = 0
    INT32 = 2
    UINT8 = 3
    INT64 = 4
    BOOL = 6
    INT16 = 7
    INT8 = 9
    FLOAT16 = 10
    UINT4 = 17
    INT4 = 18
    MXFP4 = 30


class _BuiltinOperatorValues:
    """Provide the builtin operator values used by fake Circle models."""

    ADD = 0
    CONV_2D = 3
    FULLY_CONNECTED = 9
    GATHER = 36
    DEQUANTIZE = 60
    RMS_NORM = 120


class _FakeQuantization:
    """Mimic generated Circle quantization accessors."""

    def __init__(
        self,
        scales: list[float] | None = None,
        zero_points: list[int] | None = None,
        axis: int = 0,
    ) -> None:
        self._scales = np.asarray(scales or [], dtype=np.float32)
        self._zero_points = np.asarray(zero_points or [], dtype=np.int64)
        self._axis = axis

    def ScaleAsNumpy(self) -> np.ndarray:
        """Return scale values."""

        return self._scales

    def ScaleLength(self) -> int:
        """Return the number of scale values."""

        return int(self._scales.size)

    def Scale(self, index: int) -> float:
        """Return one scale value."""

        return float(self._scales[index])

    def ZeroPointAsNumpy(self) -> np.ndarray:
        """Return zero-point values."""

        return self._zero_points

    def ZeroPointLength(self) -> int:
        """Return the number of zero-point values."""

        return int(self._zero_points.size)

    def ZeroPoint(self, index: int) -> int:
        """Return one zero-point value."""

        return int(self._zero_points[index])

    def QuantizedDimension(self) -> int:
        """Return the per-channel quantized dimension."""

        return self._axis


class _FakeTensor:
    """Mimic a generated Circle tensor object."""

    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        tensor_type: int,
        buffer_index: int,
        quantization: _FakeQuantization | None = None,
        *,
        is_variable: bool = False,
    ) -> None:
        self._name = name
        self._shape = np.asarray(shape, dtype=np.int32)
        self._type = tensor_type
        self._buffer_index = buffer_index
        self._quantization = quantization
        self._is_variable = is_variable

    def Name(self) -> bytes:
        """Return the encoded tensor name."""

        return self._name.encode("utf-8")

    def ShapeAsNumpy(self) -> np.ndarray:
        """Return the tensor shape."""

        return self._shape

    def ShapeLength(self) -> int:
        """Return the tensor rank."""

        return int(self._shape.size)

    def Shape(self, index: int) -> int:
        """Return one tensor dimension."""

        return int(self._shape[index])

    def Type(self) -> int:
        """Return the tensor type."""

        return self._type

    def Buffer(self) -> int:
        """Return the backing buffer index."""

        return self._buffer_index

    def Quantization(self) -> _FakeQuantization | None:
        """Return optional quantization metadata."""

        return self._quantization

    def IsVariable(self) -> bool:
        """Return whether the tensor is mutable."""

        return self._is_variable


class _FakeBuffer:
    """Mimic a generated Circle buffer object."""

    def __init__(self, data: bytes = b"") -> None:
        self._data = np.frombuffer(data, dtype=np.uint8)

    def DataAsNumpy(self) -> np.ndarray:
        """Return buffer bytes."""

        return self._data

    def DataLength(self) -> int:
        """Return the buffer byte length."""

        return int(self._data.size)

    def Data(self, index: int) -> int:
        """Return one buffer byte."""

        return int(self._data[index])


class _FakeOperator:
    """Mimic a generated Circle operator object."""

    def __init__(
        self,
        opcode_index: int,
        inputs: tuple[int, ...],
        outputs: tuple[int, ...],
    ) -> None:
        self._opcode_index = opcode_index
        self._inputs = np.asarray(inputs, dtype=np.int32)
        self._outputs = np.asarray(outputs, dtype=np.int32)

    def OpcodeIndex(self) -> int:
        """Return the opcode table index."""

        return self._opcode_index

    def InputsAsNumpy(self) -> np.ndarray:
        """Return input tensor IDs."""

        return self._inputs

    def InputsLength(self) -> int:
        """Return the number of inputs."""

        return int(self._inputs.size)

    def Inputs(self, index: int) -> int:
        """Return one input tensor ID."""

        return int(self._inputs[index])

    def OutputsAsNumpy(self) -> np.ndarray:
        """Return output tensor IDs."""

        return self._outputs

    def OutputsLength(self) -> int:
        """Return the number of outputs."""

        return int(self._outputs.size)

    def Outputs(self, index: int) -> int:
        """Return one output tensor ID."""

        return int(self._outputs[index])


class _FakeOperatorCode:
    """Mimic a generated Circle operator code object."""

    def __init__(self, builtin_code: int) -> None:
        self._builtin_code = builtin_code

    def BuiltinCode(self) -> int:
        """Return the current builtin code."""

        return self._builtin_code

    def DeprecatedBuiltinCode(self) -> int:
        """Return the legacy builtin code."""

        return min(self._builtin_code, 127)


class _FakeSubgraph:
    """Mimic a generated Circle subgraph object."""

    def __init__(
        self,
        tensors: list[_FakeTensor],
        operators: list[_FakeOperator],
        inputs: tuple[int, ...],
    ) -> None:
        self._tensors = tensors
        self._operators = operators
        self._inputs = np.asarray(inputs, dtype=np.int32)

    def TensorsLength(self) -> int:
        """Return the number of tensors."""

        return len(self._tensors)

    def Tensors(self, index: int) -> _FakeTensor:
        """Return one tensor."""

        return self._tensors[index]

    def OperatorsLength(self) -> int:
        """Return the number of operators."""

        return len(self._operators)

    def Operators(self, index: int) -> _FakeOperator:
        """Return one operator."""

        return self._operators[index]

    def InputsAsNumpy(self) -> np.ndarray:
        """Return runtime input tensor IDs."""

        return self._inputs

    def InputsLength(self) -> int:
        """Return the number of runtime inputs."""

        return int(self._inputs.size)

    def Inputs(self, index: int) -> int:
        """Return one runtime input tensor ID."""

        return int(self._inputs[index])


class _FakeModelRoot:
    """Mimic a generated Circle model object."""

    def __init__(
        self,
        subgraphs: list[_FakeSubgraph],
        buffers: list[_FakeBuffer],
        operator_codes: list[_FakeOperatorCode],
    ) -> None:
        self._subgraphs = subgraphs
        self._buffers = buffers
        self._operator_codes = operator_codes

    def SubgraphsLength(self) -> int:
        """Return the number of subgraphs."""

        return len(self._subgraphs)

    def Subgraphs(self, index: int) -> _FakeSubgraph:
        """Return one subgraph."""

        return self._subgraphs[index]

    def BuffersLength(self) -> int:
        """Return the number of buffers."""

        return len(self._buffers)

    def Buffers(self, index: int) -> _FakeBuffer:
        """Return one buffer."""

        return self._buffers[index]

    def OperatorCodesLength(self) -> int:
        """Return the number of operator codes."""

        return len(self._operator_codes)

    def OperatorCodes(self, index: int) -> _FakeOperatorCode:
        """Return one operator code."""

        return self._operator_codes[index]


class _FakeCircle:
    """Provide the generated Circle module hierarchy used by the analyzer."""

    class TensorType:
        """Hold the fake tensor type enum class."""

        TensorType = _TensorTypeValues

    class BuiltinOperator:
        """Hold the fake builtin operator enum class."""

        BuiltinOperator = _BuiltinOperatorValues

    class Model:
        """Hold a fake root accessor class."""

        class Model:
            """Return the model registered by a test."""

            root: _FakeModelRoot | None = None

            @classmethod
            def GetRootAsModel(cls, data: Any, offset: int) -> _FakeModelRoot:
                """Return the test model regardless of the binary payload."""

                assert offset == 0
                assert cls.root is not None
                return cls.root


def _pack_uint4(values: list[int]) -> bytes:
    """Pack low-nibble-first UINT4 values for fake Circle buffers."""

    array = np.asarray(values, dtype=np.uint8)
    packed = np.zeros((array.size + 1) // 2, dtype=np.uint8)
    packed[:] = array[0::2]
    packed[: array.size // 2] |= array[1::2] << np.uint8(4)
    return packed.tobytes()


def _make_model_with_linear_embedding_and_norm() -> _FakeModelRoot:
    """Build a fake mixed-dtype model with three selected weights."""

    buffers = [
        _FakeBuffer(),
        _FakeBuffer(_pack_uint4([2, 2, 3, 4, 5, 1, 5, 0])),
        _FakeBuffer(np.asarray([0, 0], dtype="<i4").tobytes()),
        _FakeBuffer(np.asarray([0, 1, 0, 2], dtype=np.uint8).tobytes()),
        _FakeBuffer(np.asarray([0, 0], dtype=np.uint8).tobytes()),
        _FakeBuffer(np.asarray([0, 1, 0, 2], dtype="<i2").tobytes()),
    ]
    tensors = [
        _FakeTensor("input", (1, 4), _TensorTypeValues.FLOAT32, 0),
        _FakeTensor(
            "tico::p_model_layers_0_q_proj_weight",
            (2, 4),
            _TensorTypeValues.UINT4,
            1,
            _FakeQuantization([1.0, 1.0], [2, 5], axis=0),
        ),
        _FakeTensor(
            "tico::p_model_layers_0_q_proj_bias",
            (2,),
            _TensorTypeValues.INT32,
            2,
            _FakeQuantization([1.0], [0]),
        ),
        _FakeTensor("linear_output", (1, 2), _TensorTypeValues.FLOAT32, 0),
        _FakeTensor("indices", (2,), _TensorTypeValues.INT32, 0),
        _FakeTensor(
            "tico::p_model_embed_tokens_weight",
            (2, 2),
            _TensorTypeValues.UINT8,
            3,
            _FakeQuantization([1.0], [0]),
        ),
        _FakeTensor("embedding_output", (2, 2), _TensorTypeValues.FLOAT32, 0),
        _FakeTensor(
            "const_quantized_table",
            (2,),
            _TensorTypeValues.UINT8,
            4,
            _FakeQuantization([1.0], [0]),
        ),
        _FakeTensor(
            "tico::p_model_norm_weight",
            (4,),
            _TensorTypeValues.INT16,
            5,
            _FakeQuantization([1.0], [0]),
        ),
        _FakeTensor("norm_output", (1, 4), _TensorTypeValues.FLOAT32, 0),
    ]
    operator_codes = [
        _FakeOperatorCode(_BuiltinOperatorValues.FULLY_CONNECTED),
        _FakeOperatorCode(_BuiltinOperatorValues.GATHER),
        _FakeOperatorCode(_BuiltinOperatorValues.RMS_NORM),
    ]
    operators = [
        _FakeOperator(0, (0, 1, 2), (3,)),
        _FakeOperator(1, (5, 4), (6,)),
        _FakeOperator(2, (3, 8), (9,)),
    ]
    return _FakeModelRoot(
        [_FakeSubgraph(tensors, operators, inputs=(0, 4))],
        buffers,
        operator_codes,
    )


def _report(sparsity_pct: float = 12.5) -> CircleWeightSparsityReport:
    """Build a compact report for serialization tests."""

    return CircleWeightSparsityReport(
        row=CircleWeightSparsityRow(
            scope="All model weights",
            qdtype="mixed (uint4, uint8)",
            sparsity_pct=sparsity_pct,
        ),
        source_count=1,
        tensor_count=2,
        zero_count=1,
        numel=8,
        duplicate_tensor_count=0,
        skipped_tensor_count=0,
        skipped_messages=(),
    )


def test_auto_selection_counts_model_weights_and_excludes_biases() -> None:
    _FakeCircle.Model.Model.root = _make_model_with_linear_embedding_and_norm()

    stats, duplicates, skipped = analyze_circle_binary(
        b"fake-circle",
        circle_module=_FakeCircle,
        selection="auto",
        chunk_numel=4,
    )

    assert duplicates == 0
    assert skipped == []
    assert {item.tensor_name for item in stats} == {
        "tico::p_model_layers_0_q_proj_weight",
        "tico::p_model_embed_tokens_weight",
        "tico::p_model_norm_weight",
    }
    assert sum(item.zero_count for item in stats) == 8
    assert sum(item.numel for item in stats) == 16
    assert {item.qdtype for item in stats} == {"uint4", "uint8", "int16"}


def test_aggregate_report_is_element_weighted_and_has_stable_dtype_order() -> None:
    _FakeCircle.Model.Model.root = _make_model_with_linear_embedding_and_norm()
    stats, duplicates, skipped = analyze_circle_binary(
        b"fake-circle",
        circle_module=_FakeCircle,
        selection="auto",
    )

    report = aggregate_circle_weight_stats(
        stats,
        source_count=1,
        duplicate_tensor_count=duplicates,
        skipped_messages=skipped,
    )

    assert report.row.scope == "All model weights"
    assert report.row.qdtype == "mixed (uint4, uint8, int16)"
    assert report.row.sparsity_pct == pytest.approx(50.0)
    assert report.zero_count == 8
    assert report.numel == 16


def test_quantized_constant_mode_includes_unconsumed_quantized_constants() -> None:
    _FakeCircle.Model.Model.root = _make_model_with_linear_embedding_and_norm()

    stats, _, _ = analyze_circle_binary(
        b"fake-circle",
        circle_module=_FakeCircle,
        selection="quantized-constants",
    )

    names = {item.tensor_name for item in stats}
    assert "const_quantized_table" in names
    assert "tico::p_model_layers_0_q_proj_bias" in names


def test_dequantize_passthrough_resolves_the_stored_weight() -> None:
    buffers = [
        _FakeBuffer(),
        _FakeBuffer(np.asarray([0, 1, 0, 2], dtype=np.uint8).tobytes()),
    ]
    tensors = [
        _FakeTensor("input", (1, 4), _TensorTypeValues.FLOAT32, 0),
        _FakeTensor(
            "tico::p_weight",
            (2, 2),
            _TensorTypeValues.UINT8,
            1,
            _FakeQuantization([1.0], [0]),
        ),
        _FakeTensor("dequantized_weight", (2, 2), _TensorTypeValues.FLOAT32, 0),
        _FakeTensor("output", (1, 2), _TensorTypeValues.FLOAT32, 0),
    ]
    operator_codes = [
        _FakeOperatorCode(_BuiltinOperatorValues.DEQUANTIZE),
        _FakeOperatorCode(_BuiltinOperatorValues.FULLY_CONNECTED),
    ]
    operators = [
        _FakeOperator(0, (1,), (2,)),
        _FakeOperator(1, (0, 2, -1), (3,)),
    ]
    _FakeCircle.Model.Model.root = _FakeModelRoot(
        [_FakeSubgraph(tensors, operators, inputs=(0,))],
        buffers,
        operator_codes,
    )

    stats, _, _ = analyze_circle_binary(
        b"fake-circle",
        circle_module=_FakeCircle,
        selection="operator-inputs",
    )

    assert len(stats) == 1
    assert stats[0].tensor_name == "tico::p_weight"
    assert stats[0].zero_count == 2
    assert stats[0].numel == 4


def test_uint4_unpacking_trims_unused_high_nibble() -> None:
    raw = np.frombuffer(_pack_uint4([2, 3, 2, 4, 2]), dtype=np.uint8)

    zero_count, numel = _count_tensor_semantic_zeros(
        raw,
        "uint4",
        (5,),
        QuantizationInfo((2,), 0, True),
        chunk_numel=4,
    )

    assert zero_count == 3
    assert numel == 5


def test_float_weights_count_positive_and_negative_zero() -> None:
    values = np.asarray([0.0, -0.0, 1.0, -2.0], dtype="<f4")

    zero_count, numel = _count_tensor_semantic_zeros(
        values.view(np.uint8),
        "float32",
        (4,),
        QuantizationInfo((), None, False),
        chunk_numel=2,
    )

    assert zero_count == 2
    assert numel == 4


def test_signed_int4_uses_twos_complement_nibbles() -> None:
    raw = np.frombuffer(_pack_uint4([0, 8, 15, 1]), dtype=np.uint8)

    zero_count, numel = _count_tensor_semantic_zeros(
        raw,
        "int4",
        (4,),
        QuantizationInfo((0,), 0, True),
        chunk_numel=2,
    )

    assert zero_count == 1
    assert numel == 4


def test_per_channel_zero_points_support_nonzero_quantized_axis() -> None:
    values = np.asarray(
        [
            [[1, 0], [2, 2], [3, 0]],
            [[0, 1], [2, 4], [5, 3]],
        ],
        dtype=np.uint8,
    )

    zero_count, numel = _count_tensor_semantic_zeros(
        values.reshape(-1),
        "uint8",
        (2, 3, 2),
        QuantizationInfo((1, 2, 3), 1, True),
        chunk_numel=5,
    )

    assert zero_count == 7
    assert numel == 12


def test_model_buffer_reuse_is_counted_once() -> None:
    model = _make_model_with_linear_embedding_and_norm()
    shared = model.Subgraphs(0).Tensors(5)
    model.Subgraphs(0)._tensors.append(
        _FakeTensor(
            "tico::p_tied_lm_head_weight",
            (2, 2),
            _TensorTypeValues.UINT8,
            shared.Buffer(),
            shared.Quantization(),
        )
    )
    _FakeCircle.Model.Model.root = model

    stats, duplicates, _ = analyze_circle_binary(
        b"fake-circle",
        circle_module=_FakeCircle,
        selection="auto",
    )

    assert len(stats) == 3
    assert duplicates == 1


def test_cross_file_deduplication_requires_name_and_payload_match() -> None:
    base = CircleWeightTensorStats(
        source="a.circle",
        subgraph_index=0,
        tensor_index=1,
        tensor_name="tico::p_weight",
        buffer_index=1,
        qdtype="uint4",
        shape=(2, 2),
        zero_count=2,
        numel=4,
        roles=("FULLY_CONNECTED:input1",),
        fingerprint="abc",
    )
    duplicate = replace(base, source="b.circle", tensor_index=2)
    different_name = replace(
        base,
        source="c.circle",
        tensor_name="tico::p_other_weight",
    )

    unique, duplicate_count = _deduplicate_across_files(
        [base, duplicate, different_name]
    )

    assert unique == [base, different_name]
    assert duplicate_count == 1


def test_renderers_expose_exactly_three_columns() -> None:
    report = _report()

    markdown = render_markdown(report, precision=3)
    csv_text = render_csv(report, precision=3)
    json_payload = __import__("json").loads(render_json(report, precision=3))

    assert markdown.splitlines()[0] == "| Scope | Qdtype | Sparsity (%) |"
    assert csv_text.splitlines()[0] == "Scope,Qdtype,Sparsity (%)"
    assert list(json_payload[0]) == ["Scope", "Qdtype", "Sparsity (%)"]
    assert json_payload[0]["Sparsity (%)"] == 12.5


def test_cli_writes_one_model_level_csv_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeCircle.Model.Model.root = _make_model_with_linear_embedding_and_norm()
    model_path = tmp_path / "model.circle"
    output_path = tmp_path / "result.csv"
    model_path.write_bytes(b"fake-circle")
    monkeypatch.setattr(circle_sparsity, "_load_circle_schema", lambda: _FakeCircle)

    status = circle_sparsity.main(
        [
            str(model_path),
            "--format",
            "csv",
            "--output",
            str(output_path),
            "--precision",
            "3",
        ]
    )

    assert status == 0
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "Scope,Qdtype,Sparsity (%)",
        'All model weights,"mixed (uint4, uint8, int16)",50.000',
    ]


def test_discover_circle_files_supports_recursive_directories(tmp_path: Path) -> None:
    direct = tmp_path / "a.circle"
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    nested = nested_dir / "b.circle"
    ignored = nested_dir / "notes.txt"
    direct.write_bytes(b"a")
    nested.write_bytes(b"b")
    ignored.write_text("ignored", encoding="utf-8")

    assert discover_circle_files([tmp_path], recursive=False) == [direct.resolve()]
    assert discover_circle_files([tmp_path], recursive=True) == sorted(
        [direct.resolve(), nested.resolve()]
    )


def test_unsupported_mx_dtype_fails_clearly() -> None:
    raw = np.asarray([0], dtype=np.uint8)

    with pytest.raises(Exception, match="MX formats are not yet decoded"):
        _count_tensor_semantic_zeros(
            raw,
            "mxfp4",
            (2,),
            QuantizationInfo((), None, False),
            chunk_numel=2,
        )
