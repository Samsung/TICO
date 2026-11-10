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
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.utils.version import has_transformers_for

from test.support.quantization.kv_cache import (
    assert_cache_parity,
    assert_value_preserving_qparams,
    hidden,
    tensor_qparam,
)
from test.support.quantization.qwen_kv_cache import (
    assert_qwen_decode_topology,
    load_controlled_qparams,
    make_qwen_attention,
    mask,
    prefill_args as build_prefill_args,
    rope,
)

_SKIP_MESSAGE = (
    "required transformers not installed — skipping Qwen3-VL cache Circle tests"
)


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

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(20261024)

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

    def _exercise_domains(
        self,
        dtype: DType,
        *,
        equal: bool = False,
        layout: str = "unrolled",
        compute_dtype: DType | None = None,
        zero_point_only: bool = False,
    ) -> None:
        self.attention, self.domains = make_qwen_attention(
            dtype, layout=layout, compute_dtype=compute_dtype
        )
        if equal:
            self.domains.compute_qparam = self.domains.storage_qparam
            load_controlled_qparams(self.attention, self.domains)
        if zero_point_only:
            self.domains.compute_qparam = (
                self.domains.storage_qparam[0],
                self.domains.compute_qparam[1],
            )
            load_controlled_qparams(self.attention, self.domains)
        frozen = {name: value.clone() for name, value in self.attention.named_buffers()}
        prefill_module = self.attention.as_export_module("prefill", return_kv=True)
        decode_module = self.attention.as_export_module("decode", return_kv=True)
        for optimize in (False, True):
            label = f"{dtype} layout={layout} equal={equal} O1={optimize}"
            with self.subTest(
                dtype=str(dtype),
                compute_dtype=str(compute_dtype or dtype),
                layout=layout,
                equal=equal,
                zero_point_only=zero_point_only,
                circle_optimize=optimize,
            ):
                prefill_args = build_prefill_args()
                prefill = self._convert(prefill_module, prefill_args, optimize=optimize)
                for output_index in prefill.program.outputs[-2:]:
                    tensor = prefill.program.tensor(output_index)
                    self.assertEqual(tensor_qparam(tensor), self.domains.storage_qparam)
                    self.assertEqual(tensor.dtype, self.domains.numpy_dtype)
                outputs = assert_cache_parity(
                    self,
                    self.attention,
                    prefill_module,
                    prefill_args,
                    prefill,
                    label=label + " prefill",
                )
                # Reuse serialized storage outputs without host requantization.
                past = tuple(torch.from_numpy(value.copy()) for value in outputs[-2:])
                for step in range(2):
                    length = past[0].shape[-2]
                    cosine, sine = rope(length + 1)
                    args = (
                        hidden(1) + 0.31 + step * 0.11,
                        (cosine[:, -1:], sine[:, -1:]),
                        mask(1, length + 1),
                        past,
                    )
                    decode = self._convert(decode_module, args, optimize=optimize)
                    for index in (
                        *decode.program.inputs[-2:],
                        *decode.program.outputs[-2:],
                    ):
                        tensor = decode.program.tensor(index)
                        self.assertEqual(tensor.dtype, self.domains.numpy_dtype)
                        self.assertEqual(
                            tensor_qparam(tensor), self.domains.storage_qparam
                        )
                    if step == 0 and not equal:
                        assert_qwen_decode_topology(self, decode, self.domains)
                    assert_value_preserving_qparams(self, decode.program)
                    for operator in decode.program.operators:
                        if operator.name == "QUANTIZE":
                            source = decode.program.tensor(operator.inputs[0])
                            target = decode.program.tensor(operator.outputs[0])
                            self.assertNotEqual(
                                (source.dtype, tensor_qparam(source)),
                                (target.dtype, tensor_qparam(target)),
                                "Redundant identity Quantize",
                            )
                    outputs = assert_cache_parity(
                        self,
                        self.attention,
                        decode_module,
                        args,
                        decode,
                        label=label + f" decode{step + 1}",
                    )
                    past = tuple(
                        torch.cat((previous, torch.from_numpy(delta.copy())), dim=2)
                        for previous, delta in zip(past, outputs[-2:])
                    )

        for name, value in self.attention.named_buffers():
            torch.testing.assert_close(value, frozen[name], atol=0, rtol=0, msg=name)

    def test_noninteger_domains_expose_double_rounding(self) -> None:
        self.attention, self.domains = make_qwen_attention(DType.uint(8))
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
        self.attention, self.domains = make_qwen_attention(DType.uint(8))
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
                    self.assertEqual(tensor.dtype, self.domains.numpy_dtype)
                    self.assertEqual(tensor_qparam(tensor), self.domains.storage_qparam)
                    users = [op for op in program.operators if index in op.inputs]
                    self.assertEqual(len(users), 1)
                    self.assertEqual(users[0].name, "QUANTIZE")
                    self.assertEqual(
                        tensor_qparam(program.tensor(users[0].outputs[0])),
                        self.domains.compute_qparam,
                    )
                assert_value_preserving_qparams(self, program)
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
                    self.attention, self.domains = make_qwen_attention(
                        dtype, layout=layout
                    )
                    with torch.no_grad():
                        prefill = self.attention.as_export_module("prefill")(
                            *build_prefill_args()
                        )
                    cosine, sine = rope(3)
                    args = (
                        hidden(1) + 0.31,
                        (cosine[:, -1:], sine[:, -1:]),
                        mask(1, 3),
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
                    for qparam in (
                        self.domains.storage_qparam,
                        self.domains.compute_qparam,
                    ):
                        self.assertEqual(
                            sum(
                                tensor_qparam(program.tensor(op.outputs[0])) == qparam
                                for op in concats
                            ),
                            2,
                        )
                    for index in program.outputs[-2:]:
                        self.assertEqual(
                            tensor_qparam(program.tensor(index)),
                            self.domains.storage_qparam,
                        )
                    for index in program.outputs[-2:]:
                        self.assertEqual(
                            program.tensor(index).dtype, self.domains.numpy_dtype
                        )
                    assert_value_preserving_qparams(self, program)
                    assert_cache_parity(
                        self,
                        self.attention,
                        module,
                        args,
                        runtime,
                        label=f"{dtype} {layout} present",
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
