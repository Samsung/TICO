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

"""Real Gemma4 top-level smoke export ABI and frozen-state regressions."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tico
import torch
from tico.circle import CircleDocument
from tico.circle.runtime import CircleReferenceRuntime, ExecutionMode
from tico.quantization.recipes.debug.wrapper_smoke.cases.gemma4 import (
    Gemma4ForCausalLMCase,
    Gemma4ForConditionalGenerationCase,
)
from torch.utils._pytree import tree_flatten


def _assert_state_equal(test, before, after):
    """Compare nested frozen snapshots, including non-tensor state metadata."""
    if isinstance(before, torch.Tensor):
        torch.testing.assert_close(before, after, atol=0, rtol=0, equal_nan=True)
    elif isinstance(before, dict):
        test.assertEqual(before.keys(), after.keys())
        for key in before:
            _assert_state_equal(test, before[key], after[key])
    elif isinstance(before, (tuple, list)):
        test.assertEqual(len(before), len(after))
        for first, second in zip(before, after):
            _assert_state_equal(test, first, second)
    else:
        test.assertEqual(before, after)


class _PLECausalCase(Gemma4ForCausalLMCase):
    def _make_text_config(self, cfg, **kwargs):
        kwargs["hidden_size_per_layer_input"] = 4
        return super()._make_text_config(cfg, **kwargs)


@unittest.skipUnless(
    Gemma4ForCausalLMCase().availability().available,
    "installed Transformers does not provide Gemma4 modules",
)
class TestGemma4TopLevelSmokeExport(unittest.TestCase):
    """The external-embedding ABI retains every text layer and logits path."""

    def _check_export(self, case, *, serialize=True):
        model, _ = case.build({})
        prepared = case.prepare_model(model, {}).eval()
        with torch.no_grad():
            for sample in case.calibration_inputs(prepared, {}):
                case.forward(prepared, sample)
            quantized = case.convert_model(prepared, {}).eval()
            sample = case.eval_input(quantized, {})
            state = copy.deepcopy(quantized.state_dict())
            modes = {
                name: (module.training, getattr(module, "_mode", None))
                for name, module in quantized.named_modules()
            }
            invocation = case.export_invocation(quantized, sample, {})
            # Before the fix this raises: embeds/mask dictionaries were bound
            # to input_ids and pixel_values_videos on an HF-style signature.
            eager = invocation.module(
                *invocation.inputs.args, **dict(invocation.inputs.kwargs)
            )
            self.assertEqual(eager.shape, (1, case.seq_len, case.text_cfg.vocab_size))
            self.assertEqual(
                len(invocation.module.layers), case.text_cfg.num_hidden_layers
            )
            embeds, ple, masks, _ = invocation.inputs.args
            text = case._text_owner(quantized)
            token_ple = (
                text.get_per_layer_inputs(sample.kwargs["input_ids"], embeds)
                if ple is not None
                else None
            )
            expected = quantized(
                inputs_embeds=embeds,
                per_layer_inputs=token_ple,
                attention_mask=masks,
                use_cache=False,
            ).logits
            torch.testing.assert_close(eager, expected, atol=0, rtol=0)
            if serialize:
                with tempfile.TemporaryDirectory() as output:
                    path = Path(output) / "logits.circle"
                    converted = tico.convert(invocation.module, invocation.inputs.args)
                    converted.save(path)
                    self.assertGreater(path.stat().st_size, 0)
                    document = CircleDocument.load(path)
                    document.verify()
                    runtime = CircleReferenceRuntime(document)
                    # Match torch.export's tensor-leaf order, including nested
                    # mask/RoPE dictionaries; None is a specialized constant.
                    leaves, _ = tree_flatten(invocation.inputs.args)
                    inputs = [
                        value for value in leaves if isinstance(value, torch.Tensor)
                    ]
                    self.assertEqual(len(inputs), len(runtime.program.inputs))
                    result = runtime.run(inputs, mode=ExecutionMode.FAKE_QUANTIZE)
                    self.assertEqual(len(result.outputs), 1)
                    self.assertEqual(result.outputs[0].shape, tuple(eager.shape))
                    self.assertTrue(np.isfinite(result.outputs[0]).all())
                    # Reference arithmetic is tested separately from storage and
                    # ABI: the serialized output must retain the source logits grid.
                    graph = document.model.subgraphs[0]
                    actual_grid = graph.tensors[graph.outputs[0]].quantization
                    logits = quantized.wrapped.obs_logits.require_frozen()
                    assert logits.scale is not None and logits.zero_point is not None
                    np.testing.assert_array_equal(
                        actual_grid.scale, logits.scale.numpy().reshape(-1)
                    )
                    np.testing.assert_array_equal(
                        actual_grid.zeroPoint, logits.zero_point.numpy().reshape(-1)
                    )
            _assert_state_equal(self, state, quantized.state_dict())
            self.assertEqual(
                modes,
                {
                    name: (module.training, getattr(module, "_mode", None))
                    for name, module in quantized.named_modules()
                },
            )

    def test_ple_core_matches_external_eager_logits(self):
        """Host-projected PLE retains its producer grid and softcapping semantics."""
        self._check_export(_PLECausalCase(), serialize=False)

    def test_unobserved_external_embedding_route_is_rejected(self):
        """A token-only causal calibration must not admit the external ABI."""
        case = Gemma4ForCausalLMCase()
        model, _ = case.build({})
        prepared = case.prepare_model(model, {}).eval()
        with torch.no_grad():
            for sample in case.calibration_inputs(prepared, {}):
                prepared(**dict(sample.kwargs), use_cache=True)
            quantized = case.convert_model(prepared, {}).eval()
            with self.assertRaisesRegex(RuntimeError, "external_embeddings"):
                case.export_invocation(quantized, case.eval_input(quantized, {}), {})

    def test_causal_lm_real_export(self):
        """Causal LM retains all decoder layers, norm, LM head and softcapping."""
        self._check_export(Gemma4ForCausalLMCase())

    def test_conditional_text_logits_real_export(self):
        """Text-only export uses its adapter without admitting a vision branch."""
        self._check_export(Gemma4ForConditionalGenerationCase())


if __name__ == "__main__":
    unittest.main()
