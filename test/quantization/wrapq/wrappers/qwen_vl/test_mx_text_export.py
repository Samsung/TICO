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

"""CPU target semantics and serialized NPU layout contracts, not NPU execution."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import tico
import torch
import torch.nn.functional as F
from circle_schema import circle
from tico.circle import CircleDocument
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx
from tico.quantization.passes.fold_quant_ops import FoldQuantOps
from tico.quantization.passes.legalize_mx_affine_domains import run_target_reference
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.utils.mx.mx_ops import quantize_mx

from test.quantization.wrapq.wrappers.qwen_vl.test_quantize_text_attention import (
    _make_text_config,
)
from test.support.quantization.kv_cache import load_qparam
from test.support.quantization.mx_layout import assert_qwen_mx_layout


def _inputs(length, offset=0):
    hidden = torch.linspace(-1.2, 1.1, length * 64).reshape(1, length, 64)
    angles = (torch.arange(length * 32).reshape(1, length, 32) + offset) * 0.07
    return hidden, (angles.cos(), angles.sin())


def _make_attention(*, affine_projection=True, mx_softmax=True):
    from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attention import (
        QuantQwen3VLTextAttention,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    overrides = {}
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        overrides[name + ".act_in"] = mx(axis=-1)
        overrides[name + ".act_out"] = (
            affine(DType.int(16))
            if affine_projection and name != "o_proj"
            else mx(axis=-1)
        )
    for name in ("new_k", "new_v", "present_key", "present_value"):
        overrides[name] = affine(DType.int(16), qscheme=QScheme.PER_TENSOR_ASYMM)
    if mx_softmax:
        overrides["mask_add"] = mx(axis=-1)
        overrides["softmax"] = mx(axis=-1)
    config = PTQConfig(
        activation=affine(DType.int(16)),
        weight=affine(DType.uint(4)),
        overrides=overrides,
    )
    attention = QuantQwen3VLTextAttention(
        Qwen3VLTextAttention(_make_text_config(), layer_idx=0).eval(), qcfg=config
    ).eval()
    attention.enable_calibration()
    # S and C are deliberately unrelated, imported before calibration and locked.
    for observer in (attention.obs_new_k, attention.obs_new_v):
        load_qparam(observer, 0.03125, 3)
    for observer in (attention.obs_present_key, attention.obs_present_value):
        load_qparam(observer, 0.046875, -5)
    hidden, rope = _inputs(3)
    with torch.no_grad():
        attention(hidden, rope, attention_mask=torch.zeros(1, 1, 3, 3))
    attention.freeze_qparams()
    return attention


@torch.no_grad()
def _target_formula(a, args, *, past=None):
    """Independent native-rounding formula for this explicit PTQ policy.

    Reuse source observer grids, never the legalizer or a folded graph. Norm and
    RoPE stay affine; FC and MX-softmax PV add native MX output rounding.
    """
    hidden, (cos, sin), mask = args
    fq = lambda x, name: getattr(a, "obs_" + name).fake_quant(x)

    def linear(wrapper, x):
        module = wrapper.wrapped
        x = module.obs_act_in.fake_quant(x)
        weight = module.obs_weight.fake_quant(module.module.weight)
        out = F.linear(x, weight, module.module.bias)
        return module.obs_act_out.fake_quant(quantize_mx(out, "int8", -1))

    hidden = fq(hidden, "hidden")
    batch, length, _ = hidden.shape
    q, k, v = [
        linear(getattr(a, name + "_proj"), hidden)
        .reshape(batch, length, -1, a.head_dim)
        .transpose(1, 2)
        for name in ("q", "k", "v")
    ]
    q, k = a.q_norm(q), a.k_norm(k)
    cos, sin = fq(cos, "cos").unsqueeze(1), fq(sin, "sin").unsqueeze(1)

    def rope(x, name):
        x1, x2 = x.chunk(2, dim=-1)
        half = fq(
            torch.cat(
                (fq(-fq(x2, name + "_x2"), name + "_neg"), fq(x1, name + "_x1")), -1
            ),
            name + "_cat",
        )
        return fq(
            fq(x * cos, name + "_cos") + fq(half * sin, name + "_sin"), name + "_rot"
        )

    q, k = rope(q, "q"), rope(k, "k")
    stored = fq(k, "new_k"), fq(v, "new_v")
    ck, cv = fq(k, "present_key"), fq(v, "present_value")
    if past is not None:
        sk, sv = fq(past[0], "new_k"), fq(past[1], "new_v")
        ck = torch.cat((fq(sk, "present_key"), ck), 2)
        cv = torch.cat((fq(sv, "present_value"), cv), 2)
    ck, cv = fq(ck, "present_key"), fq(cv, "present_value")
    mask = fq(mask, "causal_mask").squeeze(1)
    heads = []
    for head in range(a.num_heads):
        kv = head // a.kv_rep
        logits = fq(torch.bmm(q[:, head], ck[:, kv].transpose(1, 2)), "logits")
        logits = fq(
            quantize_mx(
                quantize_mx(logits, "int8", -1) + quantize_mx(mask, "int8", -1),
                "int8",
                -1,
            ),
            "mask_add",
        )
        probability = fq(torch.softmax(logits, -1), "softmax")
        # BMM reduction axis is 1 on V. This rounding is separate from C.
        value = quantize_mx(cv[:, kv], "int8", 1)
        out = quantize_mx(torch.bmm(probability, value), "int8", 2)
        heads.append(fq(out, "attn_out"))
    merged = fq(torch.stack(heads, 2), "attn_out_h").reshape(batch, length, -1)
    return linear(a.o_proj, merged), *stored


@unittest.skipUnless(has_transformers_for("qwen3-vl"), "requires Qwen3-VL Transformers")
class TestMxTextExport(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(20261117)

    def _convert(self, module, args, optimize):
        snapshots: list[torch.export.ExportedProgram] = []
        original = FoldQuantOps.call

        def capture(pass_instance, ep):
            if not snapshots:
                snapshots.append(copy.deepcopy(ep))
            return original(pass_instance, ep)

        with patch.object(FoldQuantOps, "call", capture):
            model = tico.convert(
                module.eval(),
                args,
                strict=False,
                config=tico.CompileConfigV1(
                    move_affine_quantize_before_layout=True,
                    legalize_mx_affine_domains=True,
                    circle_optimize=optimize,
                ),
            )
        doc = CircleDocument.from_bytes(model.circle_binary)
        doc.verify()
        return doc, snapshots[0]

    def test_prefill_decode_and_no_cache_target_contract(self):
        a = _make_attention()
        frozen = copy.deepcopy(a.state_dict())
        hidden, rope = _inputs(3)
        prefill_args = hidden, rope, torch.zeros(1, 1, 3, 3)
        expected = _target_formula(a, prefill_args)
        for optimize in (False, True):
            for return_kv in (False, True):
                with self.subTest(mode="prefill", return_kv=return_kv, O1=optimize):
                    module = a.as_export_module("prefill", return_kv=return_kv)
                    doc, pre_fold = self._convert(module, prefill_args, optimize)
                    assert_qwen_mx_layout(self, doc)
                    actual = run_target_reference(pre_fold, *prefill_args)
                    actual = actual if return_kv else (actual,)
                    for value, reference in zip(actual, expected):
                        torch.testing.assert_close(value, reference, atol=0, rtol=0)
                    self.assertEqual(len(actual), 3 if return_kv else 1)
            hidden, rope = _inputs(1, 96)
            past = expected[-2:]
            decode_args = hidden, rope, torch.zeros(1, 1, 1, 4), past
            with self.subTest(mode="decode", O1=optimize):
                doc, pre_fold = self._convert(
                    a.as_export_module("decode", return_kv=True), decode_args, optimize
                )
                assert_qwen_mx_layout(self, doc)
                reference = _target_formula(a, decode_args[:3], past=past)
                actual = run_target_reference(pre_fold, *decode_args)
                for value, ref in zip(actual, reference):
                    torch.testing.assert_close(value, ref, atol=0, rtol=0)
                sg = doc.model.subgraphs[0]
                for index in (*sg.inputs[-2:], *sg.outputs[-2:]):
                    tensor = sg.tensors[index]
                    self.assertEqual(tensor.type, circle.TensorType.TensorType.INT16)
                    np.testing.assert_array_equal(tensor.quantization.scale, [0.03125])
                    np.testing.assert_array_equal(tensor.quantization.zeroPoint, [3])
                # Both independent grids survive O1, including cache read conversion.
                scales = {
                    float(t.quantization.scale[0])
                    for t in sg.tensors
                    if t.quantization is not None
                    and t.quantization.scale is not None
                    and len(t.quantization.scale) == 1
                }
                self.assertIn(0.03125, scales)
                self.assertIn(0.046875, scales)
        for value, original in zip(
            torch.utils._pytree.tree_leaves(a.state_dict()),
            torch.utils._pytree.tree_leaves(frozen),
        ):
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, original, atol=0, rtol=0)
            else:
                self.assertEqual(value, original)
        restored = _make_attention()
        restored.load_state_dict(frozen)
        from tico.quantization.wrapq.lifecycle import validate_frozen_model

        validate_frozen_model(restored)

    def test_legacy_mx_value_projection_shared_layout_is_not_retagged(self):
        # Historical all-linear-MX config: a shared V split defeats private
        # boundary movement. Keep its MX rounding; do not retag during export.
        a = _make_attention(affine_projection=False, mx_softmax=False)
        hidden, rope = _inputs(3)
        args = hidden, rope, torch.zeros(1, 1, 3, 3)
        original = a.v_proj.wrapped.obs_act_out
        for mode in ("prefill", "decode"):
            inputs: tuple[object, ...] = args
            if mode == "decode":
                hidden, rope = _inputs(1, 96)
                past = tuple(torch.zeros(1, 2, 3, 32) for _ in range(2))
                inputs = hidden, rope, torch.zeros(1, 1, 1, 4), past
            doc, _ = self._convert(
                a.as_export_module(mode, return_kv=True), inputs, False
            )
            sg = doc.model.subgraphs[0]
            mx_layouts = [
                op
                for op in sg.operators
                if doc.model.operatorCodes[op.opcodeIndex].builtinCode
                == circle.BuiltinOperator.BuiltinOperator.RESHAPE
                and sg.tensors[op.inputs[0]].type == circle.TensorType.TensorType.MXINT8
            ]
            self.assertTrue(
                mx_layouts, "baseline shared V layout must remain observable"
            )
        self.assertIs(a.v_proj.wrapped.obs_act_out, original)
        self.assertEqual(original.axis, -1)
