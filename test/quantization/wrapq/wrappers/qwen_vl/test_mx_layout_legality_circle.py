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

"""MX-quantized Qwen3-VL modules must export no MX-typed layout ops to Circle.

The wrapper layout brackets (``q_view``/``k_view``/``v_view``,
``pre_o_proj_*``, ``qkv_reshape``, ``pre_proj_*``, ``post_norm_view``) keep
view/transpose/reshape inside the affine domain, so RESHAPE/TRANSPOSE
operators never carry MXINT8/MXFP4 tensors even when the surrounding linear
activations are MX-quantized.
"""

import unittest

import tico
import tico.quantization
import torch

from tico.circle import CircleDocument
from tico.circle.analysis import effective_builtin_code
from tico.circle.runtime.program import builtin_operator_name, tensor_type_name
from tico.quantization.config.ptq import PTQConfig
from tico.quantization.config.specs import affine, mx
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.utils.version import has_transformers_for
from tico.quantization.wrapq.wrappers.qwen_vl.quant_text_attention import (
    QuantQwen3VLTextAttention,
)
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_attention import (
    QuantQwen3VLVisionAttention,
)
from tico.quantization.wrapq.wrappers.qwen_vl.quant_vision_model import (
    QuantQwen3VLVisionModel,
)

_SKIP_MESSAGE = (
    "required transformers not installed — skipping Qwen3-VL MX Circle tests"
)
_MX_TYPE_NAMES = {"MXINT8", "MXFP4"}
_LAYOUT_OP_NAMES = {"RESHAPE", "TRANSPOSE"}


def _mx_linear_io(*names: str) -> dict:
    """Overrides putting both activation boundaries of the given linears in MX."""
    return {name: {"act_in": mx("int8"), "act_out": mx("int8")} for name in names}


def _assert_layout_ops_affine(
    test_case: unittest.TestCase,
    circle_binary: bytes,
    *,
    expected_layout_ops: set,
) -> None:
    """All RESHAPE/TRANSPOSE operands must be non-MX; MX must flow elsewhere."""
    document = CircleDocument.from_bytes(circle_binary)
    model = document.model
    subgraph = model.subgraphs[0]

    mx_tensors = [
        tensor.name
        for tensor in subgraph.tensors
        if tensor_type_name(tensor.type) in _MX_TYPE_NAMES
    ]
    test_case.assertTrue(mx_tensors, "MX domain never reached the Circle graph")

    seen = set()
    for operator in subgraph.operators:
        name = builtin_operator_name(effective_builtin_code(model, operator))
        if name not in _LAYOUT_OP_NAMES:
            continue
        seen.add(name)
        for index in (*operator.inputs, *operator.outputs):
            if index < 0:  # optional operand slot
                continue
            tensor = subgraph.tensors[index]
            test_case.assertNotIn(
                tensor_type_name(tensor.type),
                _MX_TYPE_NAMES,
                f"{name} operand {tensor.name!r} is MX-typed",
            )
    test_case.assertTrue(
        expected_layout_ops <= seen,
        f"expected layout ops {expected_layout_ops}, found {seen}",
    )


def _convert_to_circle(module, args) -> bytes:
    converted = tico.convert(
        module.eval(),
        args,
        strict=False,
        config=tico.CompileConfigV1(circle_optimize=False),
    )
    return converted.circle_binary


class _TextAttentionExport(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(self, hidden, cosine, sine, mask):
        output, _ = self.attention(
            hidden, (cosine, sine), attention_mask=mask, use_cache=False
        )
        return output


class _VisionAttentionExport(torch.nn.Module):
    def __init__(self, attention, split_sizes):
        super().__init__()
        self.attention = attention
        self.split_sizes = split_sizes

    def forward(self, hidden, cu_seqlens, cosine, sine):
        return self.attention(
            hidden,
            cu_seqlens,
            position_embeddings=(cosine, sine),
            attention_split_sizes=self.split_sizes,
        )


@unittest.skipUnless(has_transformers_for("qwen3-vl"), _SKIP_MESSAGE)
class TestQwen3VLMXLayoutLegalityCircle(unittest.TestCase):
    """MX activations + affine layout brackets => MX-free layout ops in Circle."""

    def setUp(self) -> None:
        torch.manual_seed(20261112)

    # -- tiny module builders -------------------------------------------------

    @staticmethod
    def _text_attention(qcfg: PTQConfig) -> QuantQwen3VLTextAttention:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLTextConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

        config = Qwen3VLTextConfig.from_dict(
            {
                "hidden_size": 64,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 32,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "max_position_embeddings": 64,
            }
        )
        config._attn_implementation = "eager"
        return QuantQwen3VLTextAttention(
            Qwen3VLTextAttention(config, layer_idx=0).eval(), qcfg=qcfg
        ).eval()

    @staticmethod
    def _tiny_vision():
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

        cfg = Qwen3VLConfig(
            vision_config={
                "hidden_size": 64,
                "num_heads": 2,
                "depth": 2,
                "temporal_patch_size": 2,
                "patch_size": 16,
                "out_hidden_size": 64,
                "spatial_merge_size": 2,
                "deepstack_visual_indexes": [0, 1],
            },
            text_config={
                "hidden_size": 64,
                "intermediate_size": 256,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 32,
                "num_hidden_layers": 1,
                "attention_bias": False,
                "attention_dropout": 0.0,
                "max_position_embeddings": 1024,
                "vocab_size": 1000,
                "use_cache": False,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
            },
            image_token_id=998,
            video_token_id=999,
        )
        return Qwen3VLModel(cfg).eval().visual

    # -- tests ----------------------------------------------------------------

    def test_text_attention_unrolled(self):
        qcfg = PTQConfig(
            activation=affine(DType.int(16)),
            weight=affine(DType.uint(8)),
            overrides=_mx_linear_io("q_proj", "k_proj", "v_proj", "o_proj"),
            model_args={"attention": {"layout": "unrolled"}},
        )
        attention = self._text_attention(qcfg)

        seq = 2
        hidden = torch.randn(1, seq, 64)
        positions = torch.arange(seq, dtype=torch.float32).reshape(1, seq, 1)
        frequencies = torch.linspace(0.05, 0.25, 32).reshape(1, 1, 32)
        cosine = (positions * frequencies).cos()
        sine = (positions * frequencies).sin()
        mask = torch.zeros(1, 1, seq, seq)

        attention.enable_calibration()
        with torch.no_grad():
            attention(hidden, (cosine, sine), attention_mask=mask, use_cache=False)
        attention.freeze_qparams()

        program = _convert_to_circle(
            _TextAttentionExport(attention), (hidden, cosine, sine, mask)
        )
        _assert_layout_ops_affine(
            self, program, expected_layout_ops={"RESHAPE", "TRANSPOSE"}
        )

    def test_vision_attention(self):
        visual = self._tiny_vision()
        grid_thw = torch.tensor([[1, 8, 8]], dtype=torch.long)
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        position_embeddings = (
            QuantQwen3VLVisionModel._precompute_rope_position_embeddings(
                merge_size=int(visual.spatial_merge_size),
                rope_inv_freq=visual.rotary_pos_emb.inv_freq,
                grid_thw=grid_thw,
            )
        )
        seq_len = int(cu_seqlens[-1].item())

        qcfg = PTQConfig(
            activation=affine(DType.int(16)),
            weight=affine(DType.uint(8)),
            overrides={
                **_mx_linear_io("qkv"),
                "proj": {"act_in": mx("int8")},
            },
        )
        attention = QuantQwen3VLVisionAttention(
            visual.blocks[0].attn.eval(), qcfg=qcfg
        ).eval()

        attention.enable_calibration()
        with torch.no_grad():
            for _ in range(2):
                attention(
                    torch.randn(seq_len, 64),
                    cu_seqlens,
                    position_embeddings=position_embeddings,
                    attention_split_sizes=(seq_len,),
                )
        attention.freeze_qparams()

        cosine, sine = position_embeddings
        program = _convert_to_circle(
            _VisionAttentionExport(attention, (seq_len,)),
            (torch.randn(seq_len, 64), cu_seqlens, cosine, sine),
        )
        _assert_layout_ops_affine(
            self, program, expected_layout_ops={"RESHAPE", "TRANSPOSE"}
        )

    def test_vision_patch_merger(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionPatchMerger,
        )

        cfg = Qwen3VLVisionConfig(
            hidden_size=32,
            spatial_merge_size=2,
            out_hidden_size=64,
        )
        qcfg = PTQConfig(
            activation=affine(DType.int(16)),
            weight=affine(DType.uint(8)),
            overrides={
                "linear_fc1": {"act_in": mx("int8")},
                "linear_fc2": {"act_in": mx("int8")},
            },
        )
        merger = tico.quantization.prepare(
            Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=False).eval(),
            qcfg,
        )

        with torch.no_grad():
            for _ in range(2):
                merger(torch.randn(8, cfg.hidden_size))
        quantized = tico.quantization.convert(merger)

        program = _convert_to_circle(quantized, (torch.randn(8, cfg.hidden_size),))
        _assert_layout_ops_affine(self, program, expected_layout_ops={"RESHAPE"})


if __name__ == "__main__":
    unittest.main()
