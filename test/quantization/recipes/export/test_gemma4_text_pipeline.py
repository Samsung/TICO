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
from tico.circle.analysis import effective_builtin_code
from tico.circle.runtime.program import builtin_operator_name
from tico.quantization.recipes.export.circle import convert_and_save_circle
from tico.quantization.recipes.export.gemma4 import (
    _gemma4_text_pipeline_manifest,
    _make_decode_attention_mask,
    _make_position_embeddings,
    _make_prefill_attention_mask,
)
from tico.quantization.recipes.export.gemma4_text_pipeline import (
    Gemma4CircleTextStage,
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
            mask = torch.zeros((1, 1, 2) if decode else (1, 1, 4, 4))
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
                        torch.randn(1, 1, 2 if decode else 4, 4),
                        torch.randn(1, 1, 2 if decode else 4, 4),
                    )
                elif decode:
                    layer_kwargs["past_key_value"] = (
                        torch.randn(1, 1, 1, 4),
                        torch.randn(1, 1, 1, 4),
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
        self.assertEqual(self.manifest["schema_version"], 2)
        self.assertEqual(self.manifest["decode_kv_policy"], "compact_sliding_v1")
        self.assertEqual(
            [layer["decode_capacity"] for layer in self.manifest["layers"]], [2, 2]
        )
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
                mask = torch.where(
                    (columns <= rows) & (columns > rows - 2) & (columns < length),
                    0.0,
                    -120.0,
                )[None, None]
                kwargs = dict(
                    attention_mask=mask, position_embeddings=rope, per_layer_input=ple
                )
                prefix, key, value = pipeline.prefill_layers[0](hidden, **kwargs)
                full = pipeline.prefill_layers[1](
                    prefix, shared_key_value=(key, value), **kwargs
                )
                p = length - 1
                indices = torch.arange(max(0, p - 1), p + 1)
                compact_key = torch.zeros(1, 1, 2, 4)
                compact_value = torch.zeros_like(compact_key)
                compact_mask = torch.full((1, 1, 2), -120.0)
                compact_key[:, :, : len(indices)] = key[:, :, indices]
                compact_value[:, :, : len(indices)] = value[:, :, indices]
                compact_mask[:, :, : len(indices)] = mask[:, :, p, indices]
                single_kwargs = dict(
                    attention_mask=compact_mask,
                    position_embeddings=tuple(x[:, p : p + 1] for x in rope),
                    per_layer_input=ple[:, p : p + 1],
                    shared_key_value=(compact_key, compact_value),
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

    def test_compact_decode_internal_attention_axes(self):
        """The serialized attention work must retain the smaller key axis."""
        for index in (0, 1):
            document = CircleDocument.load(
                self.root / f"decoder_layer_decode_{index}.f32.circle"
            )
            graph = document.model.subgraphs[0]
            softmax_outputs = []
            products = []
            for operator in graph.operators:
                name = builtin_operator_name(
                    effective_builtin_code(document.model, operator)
                )
                if name == "SOFTMAX":
                    softmax_outputs.append(graph.tensors[int(operator.outputs[0])])
                elif name == "BATCH_MATMUL":
                    products.append(operator)
            self.assertEqual(len(softmax_outputs), 1)
            self.assertEqual(list(softmax_outputs[0].shape), [1, 1, 1, 2])
            self.assertEqual(len(products), 2)
            # Matmul lowering folds the leading batch/head dimensions into BMM.
            self.assertEqual(
                list(graph.tensors[int(products[0].outputs[0])].shape), [1, 1, 2]
            )
            pv = products[1]
            self.assertEqual(list(graph.tensors[int(pv.inputs[0])].shape), [1, 1, 2])
            self.assertEqual(list(graph.tensors[int(pv.inputs[1])].shape), [1, 2, 4])

    def test_legacy_full_capacity_bundle_keeps_original_abi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for artifact in self.root.glob("*.circle"):
                shutil.copyfile(artifact, root / artifact.name)
            manifest = json.loads(json.dumps(self.manifest))
            manifest["schema_version"] = 1
            manifest["finalize_contract"] = "shared_consumer_decode_alias_v1"
            del manifest["decode_kv_policy"]
            for layer in manifest["layers"]:
                del layer["decode_capacity"]
            for index, module in enumerate((self.producer, self.consumer)):
                kwargs = dict(
                    attention_mask=torch.zeros(1, 1, 4),
                    position_embeddings=(torch.ones(1, 1, 4), torch.zeros(1, 1, 4)),
                    per_layer_input=torch.zeros(1, 1, 2),
                )
                shape = (1, 1, 4 if index else 3, 4)
                kwargs["shared_key_value" if index else "past_key_value"] = (
                    torch.zeros(shape),
                    torch.zeros(shape),
                )
                convert_and_save_circle(
                    module,
                    (torch.randn(1, 1, 8),),
                    root / f"decoder_layer_decode_{index}.f32.circle",
                    kwargs=kwargs,
                )
            explicit_full = json.loads(json.dumps(manifest))
            explicit_full.update(
                schema_version=2,
                finalize_contract="shared_consumer_decode_alias_v2",
                decode_kv_policy="full_capacity_v1",
            )
            for layer in explicit_full["layers"]:
                layer["decode_capacity"] = 4
            (root / "text_pipeline.json").write_text(json.dumps(explicit_full))
            explicit_pipeline = load_gemma4_text_pipeline(
                root, expected_decode_kv_policy="full_capacity_v1"
            )
            self.assertEqual(explicit_pipeline.decode_capacities, (4, 4))
            (root / "text_pipeline.json").write_text(json.dumps(manifest))
            pipeline = load_gemma4_text_pipeline(root)
            self.assertEqual(pipeline.decode_kv_policy, "full_capacity_v1")
            decode_stage = cast(Gemma4CircleTextStage, pipeline.decode_layers[0])
            names = dict(
                zip(decode_stage.input_spec.names, decode_stage.input_spec.shapes)
            )
            self.assertEqual(list(names["past_key_value_0"]), [1, 1, 3, 4])
            hidden = torch.randn(1, 1, 8)
            legacy_kwargs: dict[str, Any] = dict(
                attention_mask=torch.zeros(1, 1, 4),
                position_embeddings=(torch.ones(1, 1, 4), torch.zeros(1, 1, 4)),
                per_layer_input=torch.zeros(1, 1, 2),
                past_key_value=(torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4)),
            )
            with torch.no_grad():
                actual = decode_stage(hidden, **legacy_kwargs)
                expected = self.producer(hidden, **legacy_kwargs)
            for result, reference in zip(actual, expected):
                torch.testing.assert_close(result, reference, atol=1e-5, rtol=1e-5)
            shared = tuple(
                torch.cat((past, delta), 2)
                for past, delta in zip(legacy_kwargs["past_key_value"], actual[1:])
            )
            consumer_kwargs = {
                key: value
                for key, value in legacy_kwargs.items()
                if key != "past_key_value"
            }
            consumer_kwargs["shared_key_value"] = shared
            with torch.no_grad():
                actual_consumer = pipeline.decode_layers[1](
                    actual[0], **consumer_kwargs
                )
                expected_consumer = self.consumer(actual[0], **consumer_kwargs)
            torch.testing.assert_close(
                actual_consumer, expected_consumer, atol=1e-5, rtol=1e-5
            )
            with self.assertRaisesRegex(ValueError, "policy"):
                load_gemma4_text_pipeline(
                    root, expected_decode_kv_policy="compact_sliding_v1"
                )

    def test_compact_metadata_cannot_relabel_incompatible_graphs(self):
        for change, message in (
            (lambda manifest: manifest.update(decode_kv_policy="unknown"), "policy"),
            (
                lambda manifest: manifest["layers"][0].update(decode_capacity=4),
                "capacity",
            ),
            (
                lambda manifest: manifest["layers"][1].update(
                    sliding_window=3, decode_capacity=3
                ),
                "producer",
            ),
            (
                lambda manifest: manifest.update(
                    finalize_contract="shared_consumer_decode_alias_v1"
                ),
                "finalize",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for artifact in self.root.glob("*.circle"):
                    shutil.copyfile(artifact, root / artifact.name)
                manifest = json.loads(json.dumps(self.manifest))
                change(manifest)
                (root / "text_pipeline.json").write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    load_gemma4_text_pipeline(root)
        with self.assertRaisesRegex(ValueError, "policy"):
            load_gemma4_text_pipeline(
                self.root, expected_decode_kv_policy="full_capacity_v1"
            )

    def test_loader_validates_each_compact_graph_boundary(self):
        pipeline = load_gemma4_text_pipeline(self.root)
        for index, role, axis in (
            (0, "past_key_value_0", 2),
            (1, "shared_key_value_0", 2),
            (1, "attention_mask", 2),
        ):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for artifact in self.root.iterdir():
                    shutil.copyfile(artifact, root / artifact.name)
                path = root / self.manifest["layers"][index]["decode_artifact"]
                document = CircleDocument.load(path)
                graph = document.model.subgraphs[0]
                decode_stage = cast(
                    Gemma4CircleTextStage, pipeline.decode_layers[index]
                )
                input_index = decode_stage.input_spec.names.index(role)
                tensor = graph.tensors[int(graph.inputs[input_index])]
                tensor.shape[axis] += 1
                document.save(path)
                with self.assertRaisesRegex(ValueError, "input contracts"):
                    load_gemma4_text_pipeline(root)

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
        tokens = (17, 8, 19, 21, 15)
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
            for token in tokens:
                eager_steps.append(runtime.decode_one(torch.tensor([[token]])).clone())

            for index, layer in enumerate(manifest["layers"]):
                for decode, modules, qlen in (
                    (False, runtime.prefill_layers, 10),
                    (True, runtime.decode_layers, 1),
                ):
                    mask = (
                        _make_decode_attention_mask(
                            max_seq_len=layer["decode_capacity"],
                            sliding_window=layer["sliding_window"],
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
                            (layer["decode_capacity"] if decode else 10)
                            - (0 if layer["is_shared"] else 1),
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
            for token in tokens:
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
                zip(tokens, baseline_steps)
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

    def test_fp_export_load_prefill_and_rolling_decode(self):
        self._connected(False)

    def test_frozen_quantized_export_load_prefill_and_rolling_decode(self):
        self._connected(True)
