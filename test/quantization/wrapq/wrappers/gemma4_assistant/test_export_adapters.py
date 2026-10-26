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

"""Tests for the Gemma4 assistant static core export adapter."""

import copy
import unittest
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM,
    )


import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization import convert, prepare
from tico.quantization.config.gemma4_assistant_builders import (
    build_gemma4_assistant_ptq_config,
)
from tico.quantization.config.specs import affine
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.qscheme import QScheme

_SKIP_MSG = "required transformers Gemma4 assistant modules are not installed"


def _has_gemma4_assistant() -> bool:
    try:
        from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (  # noqa: F401
            Gemma4AssistantForCausalLM,
        )
    except Exception:
        return False
    return True


if _has_gemma4_assistant():
    from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4_assistant import (
        make_tiny_gemma4_assistant_model,
    )
    from tico.quantization.wrapq.wrappers.gemma4_assistant.export_adapters import (
        Gemma4AssistantCoreExportAdapter,
    )
    from tico.quantization.wrapq.wrappers.gemma4_assistant.static_inputs import (
        canonicalize_gemma4_assistant_static_inputs,
        GEMMA4_ASSISTANT_CORE_INPUT_NAMES,
        Gemma4AssistantStaticShapeConfig,
    )
    from tico.quantization.wrapq.wrappers.gemma4_assistant.utils import (
        assistant_layer_type_head_dim,
    )
else:
    make_tiny_gemma4_assistant_model = None  # type: ignore[assignment]
    Gemma4AssistantCoreExportAdapter = None  # type: ignore[assignment, misc]
    canonicalize_gemma4_assistant_static_inputs = None  # type: ignore[assignment]
    GEMMA4_ASSISTANT_CORE_INPUT_NAMES = None  # type: ignore[assignment]
    Gemma4AssistantStaticShapeConfig = None  # type: ignore[assignment, misc]
    assistant_layer_type_head_dim = None  # type: ignore[assignment]


def _make_dynamic_inputs(model: "Gemma4AssistantForCausalLM", kv_len: int = 10) -> dict:
    text_cfg = model.config.get_text_config()
    kv_heads = int(text_cfg.num_key_value_heads)
    full_head_dim = assistant_layer_type_head_dim(text_cfg, "full_attention")
    sliding_head_dim = assistant_layer_type_head_dim(text_cfg, "sliding_attention")
    return {
        "inputs_embeds": torch.randn(1, 1, 2 * int(model.config.backbone_hidden_size)),
        "position_ids": torch.tensor([[kv_len - 1]]),
        "attention_mask": torch.ones(1, kv_len, dtype=torch.long),
        "shared_kv_states": {
            "full_attention": (
                torch.randn(1, kv_heads, kv_len, full_head_dim),
                torch.randn(1, kv_heads, kv_len, full_head_dim),
            ),
            "sliding_attention": (
                torch.randn(1, kv_heads, kv_len, sliding_head_dim),
                torch.randn(1, kv_heads, kv_len, sliding_head_dim),
            ),
        },
        "use_cache": False,
    }


@unittest.skipUnless(_has_gemma4_assistant(), _SKIP_MSG)
class TestGemma4AssistantCoreExportAdapter(unittest.TestCase):
    """Tensor-only ABI, mode enforcement, and torch.export graph checks."""

    def setUp(self):
        torch.manual_seed(2031)
        self.fp_model = make_tiny_gemma4_assistant_model()
        self.shape = Gemma4AssistantStaticShapeConfig(
            full_kv_length=16, sliding_kv_length=16
        )
        self.inputs = _make_dynamic_inputs(self.fp_model, kv_len=8)

    def _static_inputs(self):
        return canonicalize_gemma4_assistant_static_inputs(
            inputs_embeds=self.inputs["inputs_embeds"],
            position_ids=self.inputs["position_ids"],
            attention_mask=self.inputs["attention_mask"],
            shared_kv_states=self.inputs["shared_kv_states"],
            shape=self.shape,
            model_or_config=self.fp_model.config,
            rotary_emb=self.fp_model.model.rotary_emb,
        )

    def _converted(self):
        prepared = prepare(
            copy.deepcopy(self.fp_model),
            build_gemma4_assistant_ptq_config(num_hidden_layers=2),
        )
        with torch.no_grad():
            prepared(**self.inputs)
        return convert(prepared)

    def test_unsupported_shared_compute_policy_is_rejected_before_tracing(self):
        """A direct core adapter must validate its reused attention boundaries."""
        qcfg = build_gemma4_assistant_ptq_config(num_hidden_layers=2)
        qcfg.set_override(
            "model.layers.0.self_attn.present_key",
            affine(DType.int(16), qscheme=QScheme.PER_CHANNEL_SYMM, channel_axis=1),
        )
        prepared = prepare(copy.deepcopy(self.fp_model), qcfg)
        with torch.no_grad():
            prepared(**self.inputs)
        # The direct wrapper freeze path also supports creating export adapters;
        # it must receive the same pre-tracing policy validation as convert().
        prepared.freeze_qparams()
        with self.assertRaisesRegex(ValueError, "per-tensor affine"):
            Gemma4AssistantCoreExportAdapter(prepared)

    def test_legacy_past_overrides_require_explicit_migration(self):
        """The assistant cannot silently retain an independent read grid."""
        for role in ("past_key", "past_value"):
            with self.subTest(role=role):
                qcfg = build_gemma4_assistant_ptq_config(num_hidden_layers=2)
                qcfg.set_override(
                    f"model.layers.0.self_attn.{role}", affine(DType.uint(8))
                )
                with self.assertRaisesRegex(ValueError, "past_.*present_"):
                    prepare(copy.deepcopy(self.fp_model), qcfg)

    def test_unconverted_model_is_rejected(self):
        """The core export requires a converted (QUANT-mode) assistant."""
        prepared = prepare(
            copy.deepcopy(self.fp_model),
            build_gemma4_assistant_ptq_config(num_hidden_layers=2),
        )
        with self.assertRaisesRegex(RuntimeError, "converted"):
            Gemma4AssistantCoreExportAdapter(prepared)

    def test_adapter_matches_quant_eager_forward(self):
        """The flattened static graph must equal the quantized eager path."""
        converted = self._converted()
        adapter = Gemma4AssistantCoreExportAdapter(converted)
        static = self._static_inputs()
        with torch.no_grad():
            eager_static = converted(
                inputs_embeds=static.assistant_input,
                attention_mask=static.attention_mask_mapping(),
                shared_kv_states=static.shared_kv_mapping(),
                position_embeddings=static.position_embeddings_mapping(),
            )
            eager_dynamic = converted(**self.inputs)
            projected, hidden, centroid_logits = adapter(*static.as_tuple())

        # Same canonical inputs → the adapter must be bit-exact with the
        # eager wrapper.
        torch.testing.assert_close(
            projected, eager_static.last_hidden_state, atol=0.0, rtol=0.0
        )
        # Dynamic vs padded execution reorders float accumulation, which may
        # flip individual fake-quant rounding decisions; allow a few int16
        # quantization steps but nothing larger.
        torch.testing.assert_close(
            projected, eager_dynamic.last_hidden_state, atol=2e-2, rtol=0.0
        )
        self.assertEqual(
            tuple(projected.shape),
            (1, 1, converted.wrapped.backbone_hidden_size),
        )
        self.assertEqual(tuple(hidden.shape), (1, 1, converted.wrapped.hidden_size))
        self.assertEqual(
            tuple(centroid_logits.shape),
            (1, 1, converted.wrapped.masked_embedding.num_centroids),
        )

    def test_torch_export_is_tensor_only(self):
        """torch.export must succeed and contain no host-only dynamic ops."""
        converted = self._converted()
        adapter = Gemma4AssistantCoreExportAdapter(converted).eval()
        static = self._static_inputs()
        exported = torch.export.export(adapter, static.as_tuple(), strict=False)

        # Non-strict export lifts parameters/buffers as placeholders; only
        # the user inputs define the runtime ABI.
        user_inputs = exported.graph_signature.user_inputs
        self.assertEqual(len(user_inputs), len(GEMMA4_ASSISTANT_CORE_INPUT_NAMES))

        forbidden_fragments = (
            "aten.item",
            "aten._local_scalar_dense",
            "aten.scatter",
            "aten.topk",
            "aten.index_select",
            "aten.nonzero",
        )
        call_targets = {
            str(node.target)
            for node in exported.graph.nodes
            if node.op == "call_function"
        }
        for fragment in forbidden_fragments:
            self.assertFalse(
                any(fragment in target for target in call_targets),
                f"forbidden op {fragment} found in the core export graph",
            )

        output_node = next(node for node in exported.graph.nodes if node.op == "output")
        self.assertEqual(len(output_node.args[0]), 3)

        with torch.no_grad():
            reference = adapter(*static.as_tuple())
            replayed = exported.module()(*static.as_tuple())
        for expected, actual in zip(reference, replayed):
            torch.testing.assert_close(expected, actual)

    def test_circle_preserves_external_float_kv_and_compute_conversion(self):
        """The serialized assistant accepts FP target KV and quantizes it once."""
        converted = self._converted()
        adapter = Gemma4AssistantCoreExportAdapter(converted).eval()
        inputs = self._static_inputs().as_tuple()
        originals = tuple(value.clone() for value in inputs)
        with torch.no_grad():
            expected = adapter(*inputs)
            circle = tico.convert(adapter, inputs, strict=False)
        document = CircleDocument.from_bytes(circle.circle_binary)
        document.verify()
        runtime = CircleReferenceRuntime(document)
        program = runtime.program
        attentions = {
            layer.wrapped.self_attn.wrapped.layer_type: layer.wrapped.self_attn.wrapped
            for layer in converted.wrapped.model.layers
        }
        for input_offset, layer_type, role in (
            (1, "full_attention", "key"),
            (2, "full_attention", "value"),
            (3, "sliding_attention", "key"),
            (4, "sliding_attention", "value"),
        ):
            input_index = program.inputs[input_offset]
            source = program.tensor(input_index)
            self.assertEqual(source.dtype, np.dtype(np.float32))
            self.assertIsNone(source.quantization)
            conversions = [
                operator
                for operator in program.operators
                if operator.name == "QUANTIZE" and operator.inputs == (input_index,)
            ]
            self.assertEqual(len(conversions), 1)
            target = program.tensor(conversions[0].outputs[0])
            observer = getattr(attentions[layer_type], f"obs_present_{role}")
            scale, zero_point = observer.compute_qparams()
            self.assertEqual(target.dtype, np.dtype(np.int16))
            assert target.quantization is not None
            self.assertEqual(target.quantization.scale, (float(scale),))
            self.assertEqual(target.quantization.zero_point, (int(zero_point),))

        actual = runtime.run(
            [value.numpy() for value in inputs], mode=ExecutionMode.FAKE_QUANTIZE
        ).outputs
        for result, reference, index in zip(actual, expected, program.outputs):
            # CPU serialization changes float accumulation order. Bound the
            # two-layer core to four output-grid steps, with no relative slack.
            quantization = program.tensor(index).quantization
            assert quantization is not None
            np.testing.assert_allclose(
                result,
                reference.numpy(),
                atol=4 * quantization.scale[0],
                rtol=0,
            )
        for value, original in zip(inputs, originals):
            torch.testing.assert_close(value, original, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
