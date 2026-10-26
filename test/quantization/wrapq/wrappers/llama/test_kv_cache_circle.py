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

"""Circle contracts for independent Llama cache storage and compute domains."""

import unittest

import numpy as np

import tico

import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.circle.runtime.quantization import fake_quantize_array
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.mode import Mode
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.llama.quant_attention import QuantLlamaAttention
from tico.quantization.wrapq.wrappers.nn.quant_linear import QuantLinear

from test.quantization.quant_spec_helpers import make_affine_ptq_config

_SKIP_MESSAGE = (
    "required transformers not installed — skipping Llama cache Circle tests"
)
_STORAGE_QPARAM = (0.09375, 113)
_COMPUTE_QPARAM = (0.21875, 91)
_PRODUCER_QPARAM = (0.125, 107)


@unittest.skipUnless(has_transformers_for("llama"), _SKIP_MESSAGE)
class TestLlamaKVCacheCircle(unittest.TestCase):
    """Verify branch-local cache domains in serialized tiny Llama attention."""

    def setUp(self) -> None:
        torch.manual_seed(20261024)
        from transformers.models.llama.configuration_llama import LlamaConfig

        self.config = LlamaConfig.from_dict(
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

    def _make_attention(self, dtype: DType) -> None:
        from transformers.models.llama.modeling_llama import LlamaAttention

        self.numpy_dtype = np.dtype("int16" if dtype.signed else "uint8")
        self.storage_qparam = (_STORAGE_QPARAM[0], 0 if dtype.signed else 113)
        self.compute_qparam = (_COMPUTE_QPARAM[0], 0 if dtype.signed else 91)
        self.producer_qparam = (_PRODUCER_QPARAM[0], 0 if dtype.signed else 107)
        qcfg = make_affine_ptq_config(
            dtype=dtype,
            model_args={"profile": "npu_export"},
        )
        self.attention = QuantLlamaAttention(
            LlamaAttention(self.config, layer_idx=0).eval(),
            qcfg=qcfg,
            layer_idx=0,
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
        sine = sine.clone()
        sine[..., :2] = -sine[..., :2]
        return cosine, sine

    @staticmethod
    def _hidden(length: int) -> torch.Tensor:
        return torch.linspace(-0.91, 1.07, length * 8).reshape(1, length, 8)

    @staticmethod
    def _mask(query_length: int, key_length: int) -> torch.Tensor:
        return torch.zeros(1, query_length, key_length)

    def _calibrate_and_freeze(self) -> None:
        self.attention.enable_calibration()
        with torch.no_grad():
            prefill = self.attention(
                self._hidden(2),
                self._rope(2),
                attention_mask=self._mask(2, 2),
                use_cache=True,
                cache_output_mode="delta",
            )
            self.attention(
                self._hidden(1) + 0.17,
                self._rope(1),
                attention_mask=self._mask(1, 3),
                past_key_value=prefill[2],
                use_cache=True,
                cache_output_mode="delta",
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

        past_key = self._find_named_tensor(program, program.inputs, "past_key_value_0")
        past_value = self._find_named_tensor(
            program, program.inputs, "past_key_value_1"
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

        main_concats = []
        for operator in program.operators:
            if operator.name != "CONCATENATION" or len(operator.inputs) != 2:
                continue
            inputs = [program.tensor(index) for index in operator.inputs]
            output = program.tensor(operator.outputs[0])
            input_shapes = {tuple(tensor.shape) for tensor in inputs}
            if input_shapes == {(1, 2, 4), (1, 1, 4)} and tuple(output.shape) == (
                1,
                3,
                4,
            ):
                main_concats.append(operator)

        self.assertEqual(len(main_concats), 2)
        for concat in main_concats:
            for tensor_index in (*concat.inputs, concat.outputs[0]):
                self.assertEqual(program.tensor(tensor_index).dtype, self.numpy_dtype)
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

    def _exercise_domains(self, dtype: DType, *, equal: bool = False) -> None:
        self._make_attention(dtype)
        if equal:
            self.storage_qparam = self.producer_qparam
            self.compute_qparam = self.producer_qparam
            self._load_controlled_qparams()
        prefill_module = self.attention.as_export_module(
            "prefill", return_kv=True, require_npu_profile=True
        )
        decode_module = self.attention.as_export_module(
            "decode", return_kv=True, require_npu_profile=True
        )
        for optimize in (False, True):
            label = f"{dtype} equal={equal} O1={optimize}"
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

    def test_noninteger_domains_expose_double_rounding(self) -> None:
        self._make_attention(DType.uint(8))
        # This value is already in producer P. Direct C gives 7/32 while
        # routing through S gives 0: the two graphs are observably different.
        current = torch.full((1, 1, 4), 0.125)
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
        present_k, present_v = self.attention._build_present_kv_head(
            past_k_i=stored,
            past_v_i=stored,
            new_k_i=current,
            new_v_i=current,
        )
        expected = torch.cat((torch.zeros_like(stored), expected_current), dim=1)
        self.assertTrue(torch.equal(present_k, expected))
        self.assertTrue(torch.equal(present_v, expected))

    def test_prefill_and_decode_cache_abi_and_branch_quantizes(self) -> None:
        """UINT8: separate P→C/P→S/S→C with noninteger C/S scale ratio."""
        self._exercise_domains(DType.uint(8))

    def test_int16_prefill_and_two_decodes(self) -> None:
        """Signed symmetric storage and compute grids execute on CPU."""
        self._exercise_domains(DType.int(16))

    def test_equal_domains_remove_identity_quantizes(self) -> None:
        for dtype in (DType.uint(8), DType.int(16)):
            self._exercise_domains(dtype, equal=True)


if __name__ == "__main__":
    unittest.main()
