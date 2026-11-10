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
from tico.circle.runtime import CircleReferenceRuntime
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.utils.version import has_transformers_for

from test.support.quantization.kv_cache import (
    assert_cache_parity,
    assert_value_preserving_qparams,
    hidden,
    tensor_qparam,
)
from test.support.quantization.llama_kv_cache import (
    assert_llama_decode_topology,
    load_controlled_qparams,
    make_llama_attention,
    mask,
    prefill_args as build_prefill_args,
    rope,
)

_SKIP_MESSAGE = (
    "required transformers not installed — skipping Llama cache Circle tests"
)


@unittest.skipUnless(has_transformers_for("llama"), _SKIP_MESSAGE)
class TestLlamaKVCacheCircle(unittest.TestCase):
    """Verify branch-local cache domains in serialized tiny Llama attention."""

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

    def _exercise_domains(self, dtype: DType, *, equal: bool = False) -> None:
        self.attention, self.domains = make_llama_attention(dtype)
        if equal:
            self.domains.storage_qparam = self.domains.producer_qparam
            self.domains.compute_qparam = self.domains.producer_qparam
            load_controlled_qparams(self.attention, self.domains)
        prefill_module = self.attention.as_export_module(
            "prefill", return_kv=True, require_npu_profile=True
        )
        decode_module = self.attention.as_export_module(
            "decode", return_kv=True, require_npu_profile=True
        )
        for optimize in (False, True):
            label = f"{dtype} equal={equal} O1={optimize}"
            with self.subTest(dtype=str(dtype), equal=equal, circle_optimize=optimize):
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
                    if step == 0 and not equal:
                        assert_llama_decode_topology(self, decode, self.domains)
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

    def test_noninteger_domains_expose_double_rounding(self) -> None:
        self.attention, self.domains = make_llama_attention(DType.uint(8))
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
