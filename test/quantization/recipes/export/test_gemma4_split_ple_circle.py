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

"""Shared host lookup -> real Circle PLE -> real Circle decoder composition."""

import tempfile
import unittest
from pathlib import Path

import tico

import torch
from tico.circle import CircleDocument
from tico.circle.runtime import ExecutionMode
from tico.quantization.recipes.export.gemma4_text_pipeline import Gemma4CircleTextStage
from tico.quantization.wrapq.wrappers.gemma4.ple_embedding_host import (
    Gemma4PLEEmbeddingHostTable,
    save_gemma4_ple_embedding_artifact,
)
from tico.quantization.wrapq.wrappers.llama.export_adapters import (
    register_fake_quant_meta_kernels_for_dynamic_export,
)
from tico.utils.signature import ModelInputSpec
from torch import nn

from test.quantization.recipes.debug.test_static_gemma4_finalize import (
    cache_snapshot,
    make_batch,
    make_runtime,
)
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import _has_gemma4
from test.support.quantization.gemma_kv_cache import tensor_qparams


class _LookupRange(nn.Module):
    def __init__(self, host, start, end):
        super().__init__()
        self.host = host
        self.start, self.end = start, end

    def forward(self, input_ids):
        return self.host(input_ids, layer_start=self.start, layer_end=self.end)


class _ProjectionPhases(nn.Module):
    def __init__(self, prefill, decode):
        super().__init__()
        self.prefill, self.decode = prefill, decode

    def forward(self, inputs_embeds, per_layer_token_inputs):
        stage = self.decode if inputs_embeds.shape[1] == 1 else self.prefill
        return stage(inputs_embeds, per_layer_token_inputs)


@unittest.skipUnless(_has_gemma4(), "Transformers Gemma4 is required")
class TestGemma4SplitPLECircle(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        register_fake_quant_meta_kernels_for_dynamic_export()

    def _circle(self, root, name, module, args, kwargs, quantized):
        # Read shapes from the actual serialized model, then exercise the normal
        # saved-stage binder and explicit CPU reference execution.
        program = torch.export.export(module.eval(), args, kwargs=kwargs, strict=False)
        converted = tico.convert_from_exported_program(program)
        path = root / f"{name}.circle"
        path.write_bytes(converted.circle_binary)
        document = CircleDocument.load(path)
        document.verify()
        spec = ModelInputSpec(converted.circle_binary)
        return Gemma4CircleTextStage(
            path,
            input_shapes=dict(zip(spec.names, (list(x) for x in spec.shapes))),
            output_shapes=[
                list(document.model.subgraphs[0].tensors[i].shape)
                for i in document.model.subgraphs[0].outputs
            ],
            execution_mode=(
                ExecutionMode.FAKE_QUANTIZE if quantized else ExecutionMode.NATIVE
            ),
        )

    def _connected(self, quantized):
        runtime = make_runtime(quantized=quantized, visual_tokens=2)
        batch = make_batch(3, visual_tokens=2)
        calls = {}
        handles = []
        for phase, layers in (
            ("prefill", runtime.prefill_layers),
            ("decode", runtime.decode_layers),
        ):
            for index, module in enumerate(layers):

                def capture(module, args, kwargs, key=(phase, index)):
                    if key not in calls:
                        calls[key] = torch.utils._pytree.tree_map(
                            lambda value: (
                                value.detach().clone()
                                if isinstance(value, torch.Tensor)
                                else value
                            ),
                            (args, kwargs),
                        )

                handles.append(
                    module.register_forward_pre_hook(capture, with_kwargs=True)
                )
        with torch.no_grad(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            eager = runtime.prefill(batch, last_token_only=False)
            eager_cache = cache_snapshot(runtime)
            eager_steps = [
                runtime.decode_one(torch.tensor([[token]])).clone() for token in (17, 8)
            ]
            for handle in handles:
                handle.remove()
            ids = batch["llm_input_ids"]
            h0 = runtime.mm_fusion(runtime.token_embedding(ids), batch["pixel_values"])
            full_tokens = runtime.ple_embedding(ids)
            prefix_lookup, prefix_module, suffix_lookup, suffix_module = (
                runtime._get_split_ple_stages()
            )
            prefix = self._circle(
                root,
                "ple_prefix",
                prefix_module,
                (h0, prefix_lookup(ids)),
                {},
                quantized,
            )
            suffix = self._circle(
                root,
                "ple_suffix",
                suffix_module,
                (h0[:, 2:3], suffix_lookup(ids[:, 2:3])),
                {},
                quantized,
            )
            full = self._circle(
                root,
                "ple_full",
                runtime.ple_projection,
                (h0, full_tokens),
                {},
                quantized,
            )
            decode = self._circle(
                root,
                "ple_decode",
                runtime.ple_projection,
                (h0[:, 2:3], full_tokens[:, 2:3]),
                {},
                quantized,
            )
            # This single .pt artifact owns the table for every phase/range.
            table_path = root / "ple_embedding.pt"
            save_gemma4_ple_embedding_artifact(runtime.ple_embedding, table_path)
            host = Gemma4PLEEmbeddingHostTable.from_artifact(table_path)
            runtime.ple_embedding = host
            runtime.ple_projection = _ProjectionPhases(full, decode)
            runtime._selective_ple_embedding = host
            k, layer_count = runtime.shared_kv_suffix_start, len(runtime.prefill_layers)
            runtime._split_ple_stages = (
                _LookupRange(host, 0, k),
                prefix,
                _LookupRange(host, k, layer_count),
                suffix,
            )
            for phase, modules in (
                ("prefill", runtime.prefill_layers),
                ("decode", runtime.decode_layers),
            ):
                for index in range(len(modules)):
                    args, kwargs = calls[(phase, index)]
                    modules[index] = self._circle(
                        root,
                        f"decoder_{phase}_{index}",
                        modules[index],
                        args,
                        kwargs,
                        quantized,
                    )
                    if quantized:
                        producer = prefix if index < k else suffix
                        consumer = modules[index]
                        ple_index = consumer.input_spec.names.index("per_layer_input")
                        self.assertEqual(
                            tensor_qparams(producer.runtime.output_tensors[0]),
                            tensor_qparams(consumer.runtime.input_tensors[ple_index]),
                        )
            runtime.lm_head = self._circle(
                root, "lm_head", runtime.lm_head, (h0[:, :1],), {}, quantized
            )
            installed_split = runtime._split_ple_stages
            installed_projection = runtime.ple_projection
            outcomes = []
            for last_token_only, split in ((False, False), (True, False), (True, True)):
                logits = runtime.prefill(
                    batch, last_token_only=last_token_only, ple_prefill_split=split
                )
                cache = cache_snapshot(runtime)
                steps = []
                for token in (17, 8):
                    steps.append(
                        (
                            runtime.decode_one(torch.tensor([[token]])).clone(),
                            cache_snapshot(runtime),
                            runtime.past_len,
                        )
                    )
                outcomes.append((logits, cache, steps))
                self.assertIs(runtime.ple_embedding, host)
                self.assertIs(runtime._selective_ple_embedding, host)
                self.assertIs(runtime._split_ple_stages, installed_split)
                self.assertIs(runtime.ple_projection, installed_projection)
            baseline, baseline_cache, baseline_steps = outcomes[0]
            torch.testing.assert_close(baseline, eager, atol=1e-5, rtol=1e-5)
            for result, cache, steps in outcomes:
                torch.testing.assert_close(result, baseline, atol=1e-5, rtol=1e-5)
                for pair, expected, original in zip(cache, baseline_cache, eager_cache):
                    for value, reference, eager_value in zip(pair, expected, original):
                        torch.testing.assert_close(value, reference, atol=0, rtol=0)
                        torch.testing.assert_close(
                            value, eager_value, atol=1e-5, rtol=1e-5
                        )
                for index, (
                    (logits, cache, past_len),
                    (expected, reference_cache, reference_len),
                ) in enumerate(zip(steps, baseline_steps)):
                    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)
                    torch.testing.assert_close(
                        logits, eager_steps[index], atol=1e-5, rtol=1e-5
                    )
                    self.assertEqual(past_len, reference_len)
                    for pair, reference in zip(cache, reference_cache):
                        for value, expected_value in zip(pair, reference):
                            torch.testing.assert_close(
                                value, expected_value, atol=0, rtol=0
                            )
            self.assertEqual(runtime.past_len, 5)

    def test_fp_circle_producer_decoder_and_continuation(self):
        self._connected(False)

    def test_frozen_uint8_circle_producer_decoder_and_continuation(self):
        self._connected(True)


if __name__ == "__main__":
    unittest.main()
