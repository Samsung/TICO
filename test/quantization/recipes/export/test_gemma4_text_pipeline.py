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

"""Saved decoder manifests execute real static Circle graphs without a checkpoint."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast, ClassVar

import torch

from tico.circle import CircleDocument
from tico.quantization.recipes.export.circle import convert_and_save_circle
from tico.quantization.recipes.export.gemma4 import (
    _gemma4_text_pipeline_manifest,
    _make_decode_attention_mask,
    _make_position_embeddings,
    _make_prefill_attention_mask,
)
from tico.quantization.recipes.export.gemma4_text_pipeline import (
    load_gemma4_text_pipeline,
)
from tico.quantization.wrapq.wrappers.gemma4.quant_text_model import (
    QuantGemma4TextModel,
)
from torch import nn

from test.quantization.recipes.export.test_export_gemma4 import FakeText
from test.quantization.wrapq.wrappers.gemma4.test_quant_text_model import _has_gemma4


class _Decoder(nn.Module):
    """Small independent attention with tokenwise residual/PLE/RoPE work."""

    def __init__(self, shared):
        super().__init__()
        self.shared = shared

    def forward(
        self,
        hidden_states,
        attention_mask,
        position_embeddings,
        past_key_value=None,
        per_layer_input=None,
        shared_key_value=None,
    ):
        cos, sin = position_embeddings
        query = hidden_states[..., :4] * cos + sin
        if self.shared:
            key, value = shared_key_value
        else:
            new_key = query.unsqueeze(1)
            new_value = hidden_states[..., 4:].unsqueeze(1)
            if past_key_value is None:
                key, value = new_key, new_value
            else:
                key = torch.cat((past_key_value[0], new_key), dim=2)
                value = torch.cat((past_key_value[1], new_value), dim=2)
        scores = torch.matmul(query.unsqueeze(1), key.transpose(-1, -2))
        mask = attention_mask.reshape(1, 1, query.shape[1], key.shape[2])
        attention = torch.matmul(torch.softmax(scores * 0.5 + mask, -1), value).squeeze(
            1
        )
        ple = torch.cat((per_layer_input,) * 4, dim=-1)
        hidden = torch.tanh(hidden_states + ple + torch.cat((attention, attention), -1))
        return hidden if self.shared else (hidden, new_key, new_value)


class TestGemma4TextPipeline(unittest.TestCase):
    temp: ClassVar[tempfile.TemporaryDirectory[str]]
    root: ClassVar[Path]
    manifest: ClassVar[dict[str, Any]]
    producer: ClassVar[_Decoder]
    consumer: ClassVar[_Decoder]
    head: ClassVar[nn.Module]

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(71)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.manifest = _gemma4_text_pipeline_manifest(
            cast(QuantGemma4TextModel, FakeText()), max_seq_len=4, artifact_tag="f32"
        )
        cls.producer = _Decoder(False).eval()
        cls.consumer = _Decoder(True).eval()
        cls.head = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 32, bias=False)).eval()
        for decode, qlen in ((False, 4), (True, 1)):
            hidden = torch.randn(1, qlen, 8)
            mask = torch.zeros((1, 1, 4) if decode else (1, 1, 4, 4))
            kwargs = {
                "attention_mask": mask,
                "position_embeddings": (
                    torch.ones(1, qlen, 4),
                    torch.zeros(1, qlen, 4),
                ),
                "per_layer_input": torch.randn(1, qlen, 2),
            }
            phase = "decode" if decode else "prefill"
            for index, module in enumerate((cls.producer, cls.consumer)):
                layer_kwargs = dict(kwargs)
                if index == 1:
                    layer_kwargs["shared_key_value"] = (
                        torch.randn(1, 1, 4, 4),
                        torch.randn(1, 1, 4, 4),
                    )
                elif decode:
                    layer_kwargs["past_key_value"] = (
                        torch.randn(1, 1, 3, 4),
                        torch.randn(1, 1, 3, 4),
                    )
                convert_and_save_circle(
                    module,
                    (hidden,),
                    cls.root / f"decoder_layer_{phase}_{index}.f32.circle",
                    kwargs=layer_kwargs,
                )

        # Preserve the exported adapter's hidden_states argument name.
        class Head(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, hidden_states):
                return self.module(hidden_states)

        convert_and_save_circle(
            Head(cls.head), (torch.randn(1, 1, 8),), cls.root / "lm_head.f32.circle"
        )
        (cls.root / "text_pipeline.json").write_text(json.dumps(cls.manifest))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_saved_consumer_alias_executes_original_prefill_query(self):
        pipeline = load_gemma4_text_pipeline(
            self.root, expected_max_seq_len=4, expected_num_layers=2
        )
        self.assertEqual(pipeline.producer_indices, {1: 0})
        self.assertEqual(pipeline.suffix_start, 1)
        self.assertIs(pipeline.finalize_layers[0], pipeline.decode_layers[1])
        self.assertEqual(len(list(self.root.glob("*.circle"))), 5)
        with self.assertRaisesRegex(TypeError, "FP32"):
            pipeline.lm_head(torch.zeros(1, 1, 8, dtype=torch.float64))
        for length in (1, 2, 3):
            with self.subTest(length=length), torch.no_grad():
                hidden = torch.randn(1, 4, 8)
                ple = torch.randn(1, 4, 2)
                rope = (torch.randn(1, 4, 4), torch.randn(1, 4, 4))
                rows, columns = torch.arange(4)[:, None], torch.arange(4)[None, :]
                mask = torch.where((columns <= rows) & (columns < length), 0.0, -120.0)[
                    None, None
                ]
                kwargs = dict(
                    attention_mask=mask, position_embeddings=rope, per_layer_input=ple
                )
                prefix, key, value = pipeline.prefill_layers[0](hidden, **kwargs)
                full = pipeline.prefill_layers[1](
                    prefix, shared_key_value=(key, value), **kwargs
                )
                p = length - 1
                single_kwargs = dict(
                    attention_mask=mask[:, :, p : p + 1].squeeze(1),
                    position_embeddings=tuple(x[:, p : p + 1] for x in rope),
                    per_layer_input=ple[:, p : p + 1],
                    shared_key_value=(key, value),
                )
                single = pipeline.finalize_layers[0](
                    prefix[:, p : p + 1], **single_kwargs
                )
                torch.testing.assert_close(
                    single, full[:, p : p + 1], atol=1e-6, rtol=1e-5
                )
                golden = self.consumer(prefix[:, p : p + 1], **single_kwargs)
                torch.testing.assert_close(single, golden, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(
                    pipeline.lm_head(single), self.head(golden), atol=1e-6, rtol=1e-5
                )

    def test_missing_or_incompatible_manifests_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "re-export"):
                load_gemma4_text_pipeline(tmp)
        with self.assertRaisesRegex(ValueError, "capacity"):
            load_gemma4_text_pipeline(self.root, expected_max_seq_len=8)
        with self.assertRaisesRegex(ValueError, "layer count"):
            load_gemma4_text_pipeline(self.root, expected_num_layers=3)
        for key, value, message in (
            ("suffix_start", 0, "shared-consumer"),
            ("rope_convention", "hf", "pre_negated_sin"),
            ("max_seq_len", 3, "input contracts"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for artifact in self.root.glob("*.circle"):
                    shutil.copyfile(artifact, root / artifact.name)
                manifest = dict(self.manifest)
                manifest[key] = value
                (root / "text_pipeline.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    load_gemma4_text_pipeline(root)


class _NamedHead(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, hidden_states):
        return self.module(hidden_states)


@unittest.skipUnless(_has_gemma4(), "transformers Gemma4 modules are required")
class TestGemma4SavedDecoderRuntime(unittest.TestCase):
    def _connected(self, quantized):
        from test.quantization.recipes.debug.test_static_gemma4_finalize import (
            cache_snapshot,
            make_batch,
            make_runtime,
        )

        runtime = make_runtime(quantized=quantized)
        manifest = _gemma4_text_pipeline_manifest(
            runtime.text_model, max_seq_len=10, artifact_tag="q" if quantized else "f32"
        )
        batch = make_batch(4)
        with torch.no_grad(), tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hidden = runtime.token_embedding(batch["llm_input_ids"])
            ple = runtime._build_per_layer_inputs(batch["llm_input_ids"], hidden)
            masks, rope = runtime.build_prefill_masks_and_rope(
                batch["llm_input_ids"], batch["attention_mask"]
            )
            eager = runtime.prefill_decoder(
                hidden, ple, masks, rope, 4, last_token_only=False
            )
            eager_cache = cache_snapshot(runtime)
            eager_steps = []
            for token in (17, 8):
                eager_steps.append(runtime.decode_one(torch.tensor([[token]])).clone())

            for index, layer in enumerate(manifest["layers"]):
                for decode, modules, qlen in (
                    (False, runtime.prefill_layers, 10),
                    (True, runtime.decode_layers, 1),
                ):
                    mask = (
                        _make_decode_attention_mask(
                            max_seq_len=10, sliding_window=layer["sliding_window"]
                        )
                        if decode
                        else _make_prefill_attention_mask(
                            max_seq_len=10, sliding_window=layer["sliding_window"]
                        )
                    )
                    kwargs = dict(
                        attention_mask=mask,
                        position_embeddings=_make_position_embeddings(
                            seq_len=qlen, head_dim=layer["head_dim"]
                        ),
                        per_layer_input=ple[:, :qlen, index],
                    )
                    if layer["is_shared"] or decode:
                        shape = (
                            1,
                            layer["num_kv_heads"],
                            10 if layer["is_shared"] else 9,
                            layer["head_dim"],
                        )
                        kwargs[
                            (
                                "shared_key_value"
                                if layer["is_shared"]
                                else "past_key_value"
                            )
                        ] = (torch.randn(shape), torch.randn(shape))
                    name = layer["decode_artifact" if decode else "prefill_artifact"]
                    convert_and_save_circle(
                        modules[index], (hidden[:, :qlen],), root / name, kwargs=kwargs
                    )
            convert_and_save_circle(
                _NamedHead(runtime.lm_head),
                (hidden[:, :1],),
                root / manifest["lm_head_artifact"],
            )
            (root / "text_pipeline.json").write_text(json.dumps(manifest))
            runtime.use_text_pipeline(str(root))

            baseline = runtime.prefill_decoder(
                hidden, ple, masks, rope, 4, last_token_only=False
            )
            baseline_cache = cache_snapshot(runtime)
            baseline_steps = []
            for token in (17, 8):
                baseline_steps.append(
                    (
                        runtime.decode_one(torch.tensor([[token]])).clone(),
                        cache_snapshot(runtime),
                    )
                )
            optimized = runtime.prefill_decoder(
                hidden, ple, masks, rope, 4, last_token_only=True
            )
            # 1e-5 covers FP32 reduction scheduling changes for Q=S vs Q=1.
            # Quantized checks use the exact same frozen grids, with no refreeze.
            torch.testing.assert_close(baseline, eager, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(optimized, baseline, atol=1e-5, rtol=1e-5)
            for actual, expected, eager_pair in zip(
                cache_snapshot(runtime), baseline_cache, eager_cache
            ):
                for value, reference, original in zip(actual, expected, eager_pair):
                    torch.testing.assert_close(value, reference, atol=0, rtol=0)
                    torch.testing.assert_close(value, original, atol=1e-5, rtol=1e-5)
            self.assertEqual(runtime.past_len, 4)
            for step, (token, (expected, expected_cache)) in enumerate(
                zip((17, 8), baseline_steps)
            ):
                actual = runtime.decode_one(torch.tensor([[token]]))
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(
                    actual, eager_steps[step], atol=1e-5, rtol=1e-5
                )
                for actual_pair, expected_pair in zip(
                    cache_snapshot(runtime), expected_cache
                ):
                    for value, reference in zip(actual_pair, expected_pair):
                        torch.testing.assert_close(value, reference, atol=0, rtol=0)
                self.assertEqual(runtime.past_len, 5 + step)
            if quantized:
                # A stale producer decode artifact can have compatible shapes
                # but write a different grid into the persistent cache. Reject
                # it before the first invocation, including producer outputs.
                path = root / manifest["layers"][0]["decode_artifact"]
                document = CircleDocument.load(path)
                graph = document.model.subgraphs[0]
                key = graph.tensors[int(graph.outputs[1])]
                self.assertIsNotNone(key.quantization)
                self.assertGreater(len(key.quantization.scale), 0)
                key.quantization.scale = [
                    float(scale) * 2 for scale in key.quantization.scale
                ]
                document.save(path)
                with self.assertRaisesRegex(ValueError, "output qparams"):
                    load_gemma4_text_pipeline(root)

    def test_fp_export_load_prefill_and_two_decode_steps(self):
        self._connected(False)

    def test_frozen_quantized_export_load_prefill_and_two_decode_steps(self):
        self._connected(True)
