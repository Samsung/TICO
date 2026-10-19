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

"""Host context and serialized stage contracts for self-contained LLaMA bundles."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from tico.circle import CircleDocument
from tico.circle._schema import object_api_type
from tico.circle.value.dtype import circle_tensor_type_value
from tico.quantization.recipes.export import llama
from tico.quantization.recipes.export.llama_bundle import (
    _save_tokenizer,
    _stage_contract,
    build_llama_bundle_context,
    write_llama_bundle,
)


def _bundle_model():
    layers = []
    for index in range(2):
        layers.append(
            SimpleNamespace(
                wrapped=SimpleNamespace(
                    rope_cos_template=torch.full((1, 6, 4), float(index + 2)),
                    rope_sin_template=torch.arange(24).reshape(1, 6, 4).float() + index,
                    qcfg=SimpleNamespace(attention_mask_fill_value=-100.0 - index),
                )
            )
        )
    return SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=8,
            vocab_size=32,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            pad_token_id=None,
            eos_token_id=[2, 3],
        ),
        model=SimpleNamespace(wrapped=SimpleNamespace(layers=layers)),
    )


def _write_stage(path, input_names, output_count):
    """Serialize an identity-boundary graph with deliberately chosen input order."""
    model = object_api_type("Model")()
    model.version = 3
    model.buffers = [object_api_type("Buffer")()]
    model.operatorCodes = []
    graph = object_api_type("SubGraph")()
    graph.name = "stage"
    graph.tensors = []
    for name in input_names:
        tensor = object_api_type("Tensor")()
        tensor.name = f"model::layer::{name}"
        tensor.shape = [1]
        tensor.type = circle_tensor_type_value("FLOAT32")
        graph.tensors.append(tensor)
    graph.inputs = list(range(len(input_names)))
    graph.outputs = list(range(output_count))
    graph.operators = []
    model.subgraphs = [graph]
    CircleDocument(model).save(path)


class TestLlamaBundleContext(unittest.TestCase):
    def test_context_preserves_actual_per_layer_templates_and_mask_values(self):
        qmodel = _bundle_model()
        metadata, context = build_llama_bundle_context(
            qmodel=qmodel, max_seq_len=4, artifact_tag="f32", tokenizer=None
        )
        self.assertEqual(metadata["execution_mode"], "native")
        self.assertEqual(metadata["eos_token_id"], [2, 3])
        self.assertIsNone(metadata["tokenizer"])
        for index, layer in enumerate(qmodel.model.wrapped.layers):
            for key in ("rope_cos", "rope_sin"):
                np.testing.assert_array_equal(
                    context[f"{key}_{index}"],
                    getattr(layer.wrapped, f"{key}_template")[:, :4].numpy(),
                )
                self.assertEqual(context[f"{key}_{index}"].dtype, np.float32)
        np.testing.assert_array_equal(context["mask_fill_values"], [-100.0, -101.0])
        qmodel.model.wrapped.layers[0].wrapped.rope_cos_template.zero_()
        np.testing.assert_array_equal(context["rope_cos_0"], np.full((1, 4, 4), 2.0))

    def test_generation_ids_take_precedence_and_tokenizer_supplies_missing_pad(self):
        qmodel = _bundle_model()
        qmodel.generation_config = SimpleNamespace(
            eos_token_id=[4, 5], pad_token_id=None
        )
        tokenizer = SimpleNamespace(
            pad_token_id=0, eos_token_id=2, save_pretrained=Mock()
        )
        metadata, _ = build_llama_bundle_context(
            qmodel=qmodel, max_seq_len=4, artifact_tag="q", tokenizer=tokenizer
        )
        self.assertEqual(metadata["execution_mode"], "fake_quantize")
        self.assertEqual(metadata["eos_token_id"], [4, 5])
        self.assertEqual(metadata["pad_token_id"], 0)
        self.assertEqual(metadata["tokenizer"], "tokenizer")

    def test_context_rejects_incomplete_rope_and_nonfinite_mask(self):
        for mutation, message in (
            (
                lambda layer: setattr(layer, "rope_sin_template", torch.zeros(1, 2, 4)),
                "rope_sin_template",
            ),
            (
                lambda layer: setattr(
                    layer.qcfg, "attention_mask_fill_value", float("nan")
                ),
                "mask fill",
            ),
            (
                lambda layer: setattr(layer.qcfg, "attention_mask_fill_value", -1e100),
                "mask fill",
            ),
        ):
            with self.subTest(message=message):
                qmodel = _bundle_model()
                mutation(qmodel.model.wrapped.layers[0].wrapped)
                with self.assertRaisesRegex(ValueError, message):
                    build_llama_bundle_context(
                        qmodel=qmodel, max_seq_len=4, artifact_tag="f32", tokenizer=None
                    )

    def test_public_bundle_rejects_non_generation_capacity_before_export(self):
        with patch.object(llama, "_prepare_llama_export_model") as prepare:
            with self.assertRaisesRegex(ValueError, "at least 2"):
                llama.export_llama_bundle(
                    q_model=torch.nn.Identity(), max_seq_len=1, output_dir="unused"
                )
        prepare.assert_not_called()

    def test_failed_overwrite_invalidates_old_manifest_after_preflight(self):
        qmodel = _bundle_model()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "llama_manifest.json"
            manifest.write_text("old manifest")
            with patch.object(
                llama,
                "_prepare_llama_export_model",
                return_value=(SimpleNamespace(wrapped=qmodel), "f32", False),
            ), patch.object(
                llama,
                "export_token_embedding",
                side_effect=RuntimeError("conversion failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    llama.export_llama_bundle(
                        q_model=torch.nn.Identity(), max_seq_len=4, output_dir=directory
                    )
            self.assertFalse(manifest.exists())

    def test_legacy_per_layer_overwrite_also_invalidates_bundle_manifest(self):
        qmodel = _bundle_model()
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "llama_manifest.json"
            manifest.write_text("old manifest")
            with patch.object(
                llama,
                "_prepare_llama_export_model",
                return_value=(SimpleNamespace(wrapped=qmodel), "f32", False),
            ), patch.object(
                llama,
                "export_token_embedding",
                side_effect=RuntimeError("conversion failed"),
            ):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    llama.export_llama_per_layer(
                        q_model=torch.nn.Identity(), max_seq_len=4, output_dir=directory
                    )
            self.assertFalse(manifest.exists())

    def test_preflight_failure_preserves_existing_bundle_manifest(self):
        qmodel = _bundle_model()
        qmodel.model.wrapped.layers[0].wrapped.rope_cos_template = torch.zeros(1, 2, 4)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "llama_manifest.json"
            manifest.write_text("old manifest")
            with patch.object(
                llama,
                "_prepare_llama_export_model",
                return_value=(SimpleNamespace(wrapped=qmodel), "f32", False),
            ), patch.object(llama, "_export_prepared_llama_per_layer") as export:
                with self.assertRaisesRegex(ValueError, "rope_cos_template"):
                    llama.export_llama_bundle(
                        q_model=torch.nn.Identity(), max_seq_len=4, output_dir=directory
                    )
            self.assertEqual(manifest.read_text(), "old manifest")
            export.assert_not_called()


class TestLlamaBundleManifest(unittest.TestCase):
    def test_manifest_records_serialized_input_order_and_tokenizer(self):
        tokenizer = SimpleNamespace(
            save_pretrained=Mock(), pad_token_id=0, eos_token_id=2
        )
        metadata, context = build_llama_bundle_context(
            qmodel=_bundle_model(),
            max_seq_len=4,
            artifact_tag="f32",
            tokenizer=tokenizer,
        )
        prefill_names = [
            "position_embeddings_1",
            "hidden_states",
            "attention_mask",
            "position_embeddings_0",
        ]
        decode_names = ["past_key_value_1", *prefill_names, "past_key_value_0"]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _write_stage(output_dir / "token_embedding.f32.circle", ["input_ids"], 1)
            _write_stage(output_dir / "lm_head.f32.circle", ["hidden_states"], 1)
            for index in range(2):
                _write_stage(
                    output_dir / f"decoder_layer_prefill_{index}.f32.circle",
                    prefill_names,
                    3,
                )
                _write_stage(
                    output_dir / f"decoder_layer_decode_{index}.f32.circle",
                    decode_names,
                    3,
                )
            path = write_llama_bundle(
                metadata=metadata,
                host_context=context,
                output_dir=output_dir,
                artifact_tag="f32",
                tokenizer=tokenizer,
            )
            manifest = json.loads(path.read_text())
            self.assertEqual(path.name, "llama_manifest.json")
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(len(manifest["stages"]["layers"]), 2)
            for layer in manifest["stages"]["layers"]:
                self.assertEqual(
                    layer["prefill"]["inputs"],
                    ["rope_sin", "hidden_states", "attention_mask", "rope_cos"],
                )
                self.assertEqual(
                    layer["decode"]["inputs"],
                    [
                        "past_value",
                        "rope_sin",
                        "hidden_states",
                        "attention_mask",
                        "rope_cos",
                        "past_key",
                    ],
                )
                self.assertEqual(
                    layer["decode"]["outputs"],
                    ["hidden_states", "new_key", "new_value"],
                )
            with np.load(
                output_dir / manifest["host_context"], allow_pickle=False
            ) as saved:
                self.assertEqual(set(saved.files), set(context))
                for key, value in context.items():
                    np.testing.assert_array_equal(saved[key], value)
            tokenizer.save_pretrained.assert_called_once()
            saved_path = tokenizer.save_pretrained.call_args.args[0]
            self.assertEqual(saved_path.name, "tokenizer")
            self.assertEqual(saved_path.parent.parent, output_dir)
            self.assertTrue((output_dir / "tokenizer").is_dir())

    def test_fresh_tokenizer_save_removes_stale_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            destination = output_dir / "tokenizer"
            destination.mkdir()
            (destination / "tokenizer.json").write_text("old fast tokenizer")

            def save_pretrained(path):
                self.assertTrue((destination / "tokenizer.json").exists())
                (path / "vocab.txt").write_text("new slow tokenizer")

            _save_tokenizer(
                SimpleNamespace(save_pretrained=save_pretrained), output_dir
            )
            self.assertFalse((destination / "tokenizer.json").exists())
            self.assertEqual(
                (destination / "vocab.txt").read_text(), "new slow tokenizer"
            )
            self.assertEqual(list(output_dir.iterdir()), [destination])

    def test_failed_tokenizer_save_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            destination = output_dir / "tokenizer"
            destination.mkdir()
            previous = destination / "tokenizer.json"
            previous.write_text("old tokenizer")
            tokenizer = SimpleNamespace(
                save_pretrained=Mock(side_effect=RuntimeError("tokenizer save failed"))
            )
            with self.assertRaisesRegex(RuntimeError, "tokenizer save failed"):
                _save_tokenizer(tokenizer, output_dir)
            self.assertEqual(previous.read_text(), "old tokenizer")
            self.assertEqual(list(output_dir.iterdir()), [destination])

    def test_stage_contract_rejects_unrecognized_or_duplicate_inputs(self):
        for names, message in (
            (["unexpected"], "Unsupported"),
            (["hidden_states", "hidden_states"], "required roles"),
        ):
            with self.subTest(names=names), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "stage.circle"
                _write_stage(path, names, 1)
                with self.assertRaisesRegex(ValueError, message):
                    _stage_contract(
                        path,
                        expected_inputs=frozenset({"hidden_states"}),
                        outputs=["logits"],
                    )

    def test_stage_contract_rejects_wrong_output_arity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.circle"
            _write_stage(path, ["hidden_states"], 1)
            with self.assertRaisesRegex(ValueError, "requires 3 outputs"):
                _stage_contract(
                    path,
                    expected_inputs=frozenset({"hidden_states"}),
                    outputs=["hidden_states", "new_key", "new_value"],
                )
