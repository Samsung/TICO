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

"""Circle contracts for independent Qwen3-VL cache storage and compute domains."""

import unittest

import numpy as np

import tico

import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attention import (
    QuantQwen3VLTextAttention,
)

from test.quantization.quant_spec_helpers import make_affine_ptq_config

_SKIP_MESSAGE = (
    "required transformers not installed — skipping Qwen3-VL cache Circle tests"
)
_STORAGE_QPARAM = (0.09375, 113)
_COMPUTE_QPARAM = (0.21875, 91)
_PRODUCER_QPARAM = (0.125, 107)


class _PresentCacheExport(torch.nn.Module):
    """Expose full persistent tuple outputs in addition to attention output."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, hidden, rope, mask, past):
        output, _, present = self.attention(
            hidden,
            rope,
            attention_mask=mask,
            past_key_values=past,
            use_cache=True,
            cache_output_mode="present",
        )
        return output, *present


class _CacheBranches(torch.nn.Module):
    """Expose the wrapper's actual S/C branches with no attention math."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, raw_k, raw_v, past_k, past_v):
        attention = self.attention
        stored_k = attention._fq(raw_k, attention.obs_new_k)
        stored_v = attention._fq(raw_v, attention.obs_new_v)
        past = attention._apply_storage_grid((past_k, past_v))
        compute = attention._build_present_key_values(
            past=past, raw_k=raw_k, raw_v=raw_v
        )
        return *compute, stored_k, stored_v


@unittest.skipUnless(has_transformers_for("qwen3-vl"), _SKIP_MESSAGE)
class TestQwenKVCacheCircle(unittest.TestCase):
    """Verify branch-local cache domains in serialized tiny Qwen3-VL attention."""

    def setUp(self) -> None:
        torch.manual_seed(20261024)
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )

        self.config = Qwen3VLTextConfig.from_dict(
            {
                "hidden_size": 8,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "max_position_embeddings": 8,
            }
        )
        self.config._attn_implementation = "eager"

    def _make_attention(
        self,
        dtype: DType,
        *,
        layout: str = "unrolled",
        compute_dtype: DType | None = None,
    ) -> None:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        compute_dtype = compute_dtype or dtype
        self.compute_numpy_dtype = np.dtype(
            "int16" if compute_dtype.signed else "uint8"
        )
        self.numpy_dtype = np.dtype("int16" if dtype.signed else "uint8")
        self.storage_qparam = (_STORAGE_QPARAM[0], 0 if dtype.signed else 113)
        self.compute_qparam = (_COMPUTE_QPARAM[0], 0 if compute_dtype.signed else 91)
        self.producer_qparam = (_PRODUCER_QPARAM[0], 0 if dtype.signed else 107)
        qcfg = make_affine_ptq_config(
            dtype=dtype,
            overrides={
                "present_key": affine(compute_dtype),
                "present_value": affine(compute_dtype),
            },
            model_args={"attention": {"layout": layout}},
        )
        self.attention = QuantQwen3VLTextAttention(
            Qwen3VLTextAttention(self.config, layer_idx=0).eval(),
            qcfg=qcfg,
        ).eval()
        self._calibrate_and_freeze()
        self._load_controlled_qparams()

    @staticmethod
    def _qparam(observer, scale: float, zero_point: int) -> None:
        observer.load_qparams(
            torch.tensor(scale),
            torch.tensor(zero_point, dtype=torch.int32),
            lock=True,
        )

    @staticmethod
    def _rope(length: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, dtype=torch.float32).reshape(1, length, 1)
        frequencies = torch.tensor([0.07, 0.13, 0.07, 0.13]).reshape(1, 1, 4)
        angles = positions * frequencies
        cosine = angles.cos()
        sine = angles.sin()
        return cosine, sine

    @staticmethod
    def _hidden(length: int) -> torch.Tensor:
        return torch.linspace(-0.91, 1.07, length * 8).reshape(1, length, 8)

    @staticmethod
    def _mask(query_length: int, key_length: int) -> torch.Tensor:
        return torch.zeros(1, 1, query_length, key_length)

    def _calibrate_and_freeze(self) -> None:
        self.attention.enable_calibration()
        with torch.no_grad():
            # Prefill with cache output disabled must establish both S and C.
            self.attention(
                self._hidden(2),
                self._rope(2),
                attention_mask=self._mask(2, 2),
                use_cache=False,
            )
        self.attention.freeze_qparams()

    def _load_controlled_qparams(self) -> None:
        for observer in (self.attention.obs_new_k, self.attention.obs_new_v):
            self._qparam(observer, *self.storage_qparam)
        for observer in (
            self.attention.obs_present_key,
            self.attention.obs_present_value,
        ):
            self._qparam(observer, *self.compute_qparam)

        # Keep unrelated RoPE split/concatenation boundaries on the norm's
        # grid. Independently calibrated slice grids exercise the generic
        # split quantization lowering, rather than the KV storage contract.
        for kind in ("q", "k"):
            norm = getattr(self.attention, kind + "_norm").wrapped
            self._qparam(norm.obs_act_out, *self.producer_qparam)
            for suffix in ("x1", "x2", "neg", "cat"):
                self._qparam(
                    getattr(self.attention, f"obs_{kind}_{suffix}"),
                    *self.producer_qparam,
                )
        self._qparam(self.attention.obs_k_rot, *self.producer_qparam)
        value_projection = self.attention.v_proj.wrapped
        self.assertIsInstance(value_projection, QuantLinear)
        self._qparam(value_projection.obs_act_out, *self.producer_qparam)

    @staticmethod
    def _tensor_qparam(tensor) -> tuple[float, int] | None:
        quantization = tensor.quantization
        if quantization is None or not quantization.scale:
            return None
        return float(quantization.scale[0]), int(quantization.zero_point[0])

    @staticmethod
    def _producer_map(program):
        producers = {}
        for operator_index, operator in enumerate(program.operators):
            for tensor_index in operator.outputs:
                producers[tensor_index] = (operator_index, operator)
        return producers

    @staticmethod
    def _find_named_tensor(program, indices, suffix: str):
        for index in indices:
            tensor = program.tensor(index)
            if tensor.name and tensor.name.endswith(suffix):
                return tensor
        raise AssertionError(f"Tensor ending with {suffix!r} was not found")

    def _convert(self, module, args, *, optimize: bool):
        converted = tico.convert(
            module.eval(),
            args,
            strict=False,
            config=tico.CompileConfigV1(circle_optimize=optimize),
        )
        document = CircleDocument.from_bytes(converted.circle_binary)
        document.verify()
        return CircleReferenceRuntime(document)

    def _prefill_args(self):
        return self._hidden(2), self._rope(2), self._mask(2, 2)

    def _assert_value_preserving_qparams(self, program) -> None:
        for operator in program.operators:
            if operator.name not in {"RESHAPE", "TRANSPOSE", "CONCATENATION"}:
                continue
            # Shape and permutation operands are integer metadata, not values
            # transformed in the activation quantization domain.
            data_inputs = (
                operator.inputs
                if operator.name == "CONCATENATION"
                else operator.inputs[:1]
            )
            tensors = [program.tensor(index) for index in data_inputs]
            output = program.tensor(operator.outputs[0])
            qparams = [self._tensor_qparam(tensor) for tensor in (*tensors, output)]
            quantized = [qparam for qparam in qparams if qparam is not None]
            if not quantized:
                continue
            self.assertEqual(len(quantized), len(qparams), operator.name)
            self.assertEqual(len(set(quantized)), 1, operator.name)
            self.assertEqual(
                len({tensor.dtype for tensor in (*tensors, output)}), 1, operator.name
            )

    def _assert_decode_topology(self, runtime) -> None:
        program = runtime.program
        producers = self._producer_map(program)

        past_key = self._find_named_tensor(program, program.inputs, "past_key_values_0")
        past_value = self._find_named_tensor(
            program, program.inputs, "past_key_values_1"
        )
        self.assertEqual(self._tensor_qparam(past_key), self.storage_qparam)
        self.assertEqual(self._tensor_qparam(past_value), self.storage_qparam)
        for tensor in (past_key, past_value):
            self.assertEqual(tensor.dtype, self.numpy_dtype)

        delta_key_index, delta_value_index = program.outputs[-2:]
        delta_key = program.tensor(delta_key_index)
        delta_value = program.tensor(delta_value_index)
        self.assertEqual(self._tensor_qparam(delta_key), self.storage_qparam)
        self.assertEqual(self._tensor_qparam(delta_value), self.storage_qparam)
        self.assertEqual(delta_key.dtype, self.numpy_dtype)
        self.assertEqual(delta_value.dtype, self.numpy_dtype)

        main_concats = []
        for operator in program.operators:
            if operator.name != "CONCATENATION" or len(operator.inputs) != 2:
                continue
            inputs = [program.tensor(index) for index in operator.inputs]
            output = program.tensor(operator.outputs[0])
            input_shapes = {tuple(tensor.shape) for tensor in inputs}
            if input_shapes == {(1, 1, 2, 4), (1, 1, 1, 4)} and tuple(output.shape) == (
                1,
                1,
                3,
                4,
            ):
                main_concats.append(operator)

        self.assertEqual(len(main_concats), 2)
        for concat in main_concats:
            for tensor_index in (*concat.inputs, concat.outputs[0]):
                self.assertEqual(
                    program.tensor(tensor_index).dtype, self.compute_numpy_dtype
                )
                self.assertEqual(
                    self._tensor_qparam(program.tensor(tensor_index)),
                    self.compute_qparam,
                )

            for tensor_index in concat.inputs:
                _, producer = producers[tensor_index]
                self.assertEqual(producer.name, "QUANTIZE")
                source = program.tensor(producer.inputs[0])
                target = program.tensor(producer.outputs[0])
                self.assertEqual(self._tensor_qparam(target), self.compute_qparam)
                sequence_length = tuple(target.shape)[-2]
                expected_source = (
                    self.storage_qparam
                    if sequence_length == 2
                    else self.producer_qparam
                )
                self.assertEqual(self._tensor_qparam(source), expected_source)

        for output_index in (delta_key_index, delta_value_index):
            output = program.tensor(output_index)
            _, producer = producers[output_index]
            self.assertEqual(producer.name, "QUANTIZE")
            source = program.tensor(producer.inputs[0])
            self.assertEqual(self._tensor_qparam(source), self.producer_qparam)
            self.assertEqual(self._tensor_qparam(output), self.storage_qparam)

        self._assert_value_preserving_qparams(program)

    def _assert_parity(self, module, args, runtime, *, label: str):
        # FAKE_QUANTIZE uses dequantized FLOAT32 at its integer ABI. Explicitly
        # encode each input first, and give eager exactly those same values.
        self.assertIs(self.attention._mode, Mode.QUANT)
        leaves, spec = torch.utils._pytree.tree_flatten(args)
        self.assertEqual(len(leaves), len(runtime.program.inputs))
        encoded = []
        for value, index in zip(leaves, runtime.program.inputs):
            tensor = runtime.program.tensor(index)
            array = value.detach().numpy()
            if tensor.is_quantized_integer:
                array = fake_quantize_array(array, tensor)
            encoded.append(np.array(array, copy=True))
        eager_args = torch.utils._pytree.tree_unflatten(
            [torch.from_numpy(value.copy()) for value in encoded], spec
        )
        with torch.no_grad():
            eager = module(*eager_args)
        result = runtime.run(encoded, mode=ExecutionMode.FAKE_QUANTIZE)
        for name, expected, actual, index in zip(
            ("hidden", "K_delta", "V_delta"),
            eager,
            result.outputs,
            runtime.program.outputs,
        ):
            expected_array = expected.detach().numpy()
            error = np.abs(expected_array - actual)
            metrics = {
                "max_abs": float(error.max()),
                "mean_abs": float(error.mean()),
                "differing": int(np.count_nonzero(expected_array != actual)),
                "qparam": self._tensor_qparam(runtime.program.tensor(index)),
            }
            print(f"{label} {name}: {metrics}")
            np.testing.assert_array_equal(
                actual, expected_array, err_msg=f"{label} {name}: {metrics}"
            )
        return result.outputs

    def _exercise_domains(
        self,
        dtype: DType,
        *,
        equal: bool = False,
        layout: str = "unrolled",
        compute_dtype: DType | None = None,
        zero_point_only: bool = False,
    ) -> None:
        self._make_attention(dtype, layout=layout, compute_dtype=compute_dtype)
        if equal:
            self.compute_qparam = self.storage_qparam
            self._load_controlled_qparams()
        if zero_point_only:
            self.compute_qparam = (self.storage_qparam[0], self.compute_qparam[1])
            self._load_controlled_qparams()
        frozen = {name: value.clone() for name, value in self.attention.named_buffers()}
        prefill_module = self.attention.as_export_module("prefill", return_kv=True)
        decode_module = self.attention.as_export_module("decode", return_kv=True)
        for optimize in (False, True):
            label = f"{dtype} layout={layout} equal={equal} O1={optimize}"
            with self.subTest(dtype=str(dtype), equal=equal, circle_optimize=optimize):
                prefill_args = self._prefill_args()
                prefill = self._convert(prefill_module, prefill_args, optimize=optimize)
                for output_index in prefill.program.outputs[-2:]:
                    tensor = prefill.program.tensor(output_index)
                    self.assertEqual(self._tensor_qparam(tensor), self.storage_qparam)
                    self.assertEqual(tensor.dtype, self.numpy_dtype)
                outputs = self._assert_parity(
                    prefill_module, prefill_args, prefill, label=label + " prefill"
                )
                # Reuse serialized storage outputs without host requantization.
                past = tuple(torch.from_numpy(value.copy()) for value in outputs[-2:])
                for step in range(2):
                    length = past[0].shape[-2]
                    cosine, sine = self._rope(length + 1)
                    args = (
                        self._hidden(1) + 0.31 + step * 0.11,
                        (cosine[:, -1:], sine[:, -1:]),
                        self._mask(1, length + 1),
                        past,
                    )
                    decode = self._convert(decode_module, args, optimize=optimize)
                    for index in (
                        *decode.program.inputs[-2:],
                        *decode.program.outputs[-2:],
                    ):
                        tensor = decode.program.tensor(index)
                        self.assertEqual(tensor.dtype, self.numpy_dtype)
                        self.assertEqual(
                            self._tensor_qparam(tensor), self.storage_qparam
                        )
                    if step == 0 and not equal:
                        self._assert_decode_topology(decode)
                    self._assert_value_preserving_qparams(decode.program)
                    for operator in decode.program.operators:
                        if operator.name == "QUANTIZE":
                            source = decode.program.tensor(operator.inputs[0])
                            target = decode.program.tensor(operator.outputs[0])
                            self.assertNotEqual(
                                (source.dtype, self._tensor_qparam(source)),
                                (target.dtype, self._tensor_qparam(target)),
                                "Redundant identity Quantize",
                            )
                    outputs = self._assert_parity(
                        decode_module, args, decode, label=label + f" decode{step + 1}"
                    )
                    past = tuple(
                        torch.cat((previous, torch.from_numpy(delta.copy())), dim=2)
                        for previous, delta in zip(past, outputs[-2:])
                    )

        for name, value in self.attention.named_buffers():
            torch.testing.assert_close(value, frozen[name], atol=0, rtol=0, msg=name)

    def test_noninteger_domains_expose_double_rounding(self) -> None:
        self._make_attention(DType.uint(8))
        # This value is already in the raw producer grid. Direct C gives 7/32 while
        # routing through S gives 0: the two graphs are observably different.
        current = torch.full((1, 1, 1, 4), 0.125)
        stored = self.attention.obs_new_k.fake_quant(current)
        expected_current = torch.full_like(current, 0.21875)
        self.assertTrue(
            torch.equal(
                self.attention.obs_present_key.fake_quant(current), expected_current
            )
        )
        self.assertFalse(
            torch.equal(
                self.attention.obs_present_key.fake_quant(stored), expected_current
            )
        )
        present_k, present_v = self.attention._build_present_key_values(
            past=(stored, stored),
            raw_k=current,
            raw_v=current,
        )
        expected = torch.cat((torch.zeros_like(stored), expected_current), dim=2)
        self.assertTrue(torch.equal(present_k, expected))
        self.assertTrue(torch.equal(present_v, expected))

    def test_current_input_owns_storage_domain_and_requantizes_for_compute(
        self,
    ) -> None:
        self._make_attention(DType.uint(8))
        current = torch.full((1, 1, 1, 4), 0.125)
        stored = self.attention.obs_new_k.fake_quant(current)
        # This boundary-only fixture has no upstream producer domain. Its first
        # marker defines an S input ABI; the C branch then requires requantization.
        # Use independent tensors so export preserves all four input identities.
        module = _CacheBranches(self.attention).eval()
        args = (current, current.clone(), stored, stored.clone())
        for optimize in (False, True):
            with self.subTest(circle_optimize=optimize):
                runtime = self._convert(module, args, optimize=optimize)
                program = runtime.program
                self.assertEqual(program.outputs[-2:], program.inputs[:2])
                for index in program.inputs:
                    tensor = program.tensor(index)
                    self.assertEqual(tensor.dtype, self.numpy_dtype)
                    self.assertEqual(self._tensor_qparam(tensor), self.storage_qparam)
                    users = [op for op in program.operators if index in op.inputs]
                    self.assertEqual(len(users), 1)
                    self.assertEqual(users[0].name, "QUANTIZE")
                    self.assertEqual(
                        self._tensor_qparam(program.tensor(users[0].outputs[0])),
                        self.compute_qparam,
                    )
                self._assert_value_preserving_qparams(program)
                # This graph contains only cache data tensors, with no semantic
                # index, shape, or control tensors requiring exclusion.
                for tensor in program.tensors:
                    self.assertIsNotNone(tensor.quantization)
                    self.assertNotEqual(tensor.dtype, np.dtype(np.float32))

                # Eager raw-branch behavior is tested separately above. Compare
                # serialized execution using the S-encoded ABI inputs on both sides.
                encoded = [
                    fake_quantize_array(value.numpy(), program.tensor(index))
                    for value, index in zip(args, program.inputs)
                ]
                with torch.no_grad():
                    eager = module(
                        *(torch.from_numpy(value.copy()) for value in encoded)
                    )
                result = runtime.run(encoded, mode=ExecutionMode.FAKE_QUANTIZE)
                self.assertEqual(len(result.outputs), len(eager))
                for actual, wanted in zip(result.outputs, eager):
                    np.testing.assert_array_equal(actual, wanted.numpy())
                self.assertFalse(np.any(result.outputs[0]))
                self.assertFalse(np.any(result.outputs[1]))
                for output in result.outputs[-2:]:
                    np.testing.assert_array_equal(output, stored.numpy())

    def test_prefill_and_decode_cache_abi_and_branch_quantizes(self) -> None:
        """UINT8: separate raw→C/raw→S/S→C with noninteger C/S scale ratio."""
        for layout in ("batched", "unrolled"):
            self._exercise_domains(DType.uint(8), layout=layout)

    def test_int16_prefill_and_two_decodes(self) -> None:
        """Signed symmetric storage and compute grids execute on CPU."""
        for layout in ("batched", "unrolled"):
            self._exercise_domains(DType.int(16), layout=layout)

    def test_present_outputs_keep_a_separate_storage_concat(self) -> None:
        """Full cache outputs concatenate S while attention concatenates C."""
        for dtype in (DType.uint(8), DType.int(16)):
            for layout in ("batched", "unrolled"):
                with self.subTest(dtype=str(dtype), layout=layout):
                    self._make_attention(dtype, layout=layout)
                    with torch.no_grad():
                        prefill = self.attention.as_export_module("prefill")(
                            *self._prefill_args()
                        )
                    cosine, sine = self._rope(3)
                    args = (
                        self._hidden(1) + 0.31,
                        (cosine[:, -1:], sine[:, -1:]),
                        self._mask(1, 3),
                        prefill[-2:],
                    )
                    module = _PresentCacheExport(self.attention).eval()
                    runtime = self._convert(module, args, optimize=True)
                    program = runtime.program
                    concats = [
                        op
                        for op in program.operators
                        if op.name == "CONCATENATION"
                        and program.tensor(op.outputs[0]).shape == (1, 1, 3, 4)
                    ]
                    self.assertEqual(len(concats), 4)
                    for qparam in (self.storage_qparam, self.compute_qparam):
                        self.assertEqual(
                            sum(
                                self._tensor_qparam(program.tensor(op.outputs[0]))
                                == qparam
                                for op in concats
                            ),
                            2,
                        )
                    for index in program.outputs[-2:]:
                        self.assertEqual(
                            self._tensor_qparam(program.tensor(index)),
                            self.storage_qparam,
                        )
                    for index in program.outputs[-2:]:
                        self.assertEqual(program.tensor(index).dtype, self.numpy_dtype)
                    self._assert_value_preserving_qparams(program)
                    self._assert_parity(
                        module, args, runtime, label=f"{dtype} {layout} present"
                    )

    def test_int16_storage_to_uint8_compute(self) -> None:
        """Mixed affine dtypes retain signed storage and explicit UINT8 compute."""
        self._exercise_domains(DType.int(16), compute_dtype=DType.uint(8))

    def test_uint8_storage_to_int16_compute(self) -> None:
        """UINT8 cache deltas also remain independent of signed compute."""
        self._exercise_domains(DType.uint(8), compute_dtype=DType.int(16))

    def test_zero_point_only_mismatch_retains_conversion(self) -> None:
        """Equal dtype and scale cannot erase a distinct affine zero-point."""
        self._exercise_domains(DType.uint(8), zero_point_only=True)

    def test_equal_domains_remove_identity_quantizes(self) -> None:
        for dtype in (DType.uint(8), DType.int(16)):
            for layout in ("batched", "unrolled"):
                self._exercise_domains(dtype, equal=True, layout=layout)


if __name__ == "__main__":
    unittest.main()
