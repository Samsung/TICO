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

"""Execute real, saved LLaMA stages without a checkpoint or remote downloads."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import numpy as np

import torch
from tico.quantization.recipes.export.llama import export_llama_bundle
from tico.quantization.recipes.runtime.llama import LlamaCircleRuntime
from tico.quantization.wrapq.utils.version import has_transformers_for


def _make_model(num_hidden_layers: int = 2) -> torch.nn.Module:
    """Use local random weights and nontrivial grouped-query attention."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        config = LlamaConfig.from_dict(
            {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": num_hidden_layers,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 32,
                "max_position_embeddings": 8,
                "attention_dropout": 0.0,
                "attention_bias": False,
                "attn_implementation": "eager",
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "use_cache": False,
                "return_dict": True,
            }
        )
        return LlamaForCausalLM(config).eval()


def _reference_logits(model: torch.nn.Module, tokens: np.ndarray) -> np.ndarray:
    """Recompute the entire sequence independently of the runtime's KV state."""
    with torch.no_grad():
        return (
            model(input_ids=torch.from_numpy(tokens), use_cache=False)
            .logits[:, -1, :]
            .numpy()
        )


def _reference_generate(
    model: torch.nn.Module, tokens: np.ndarray, count: int
) -> np.ndarray:
    """Greedy decoding through the full eager model, without stage adapters."""
    result = tokens.copy()
    for _ in range(count):
        next_token = np.asarray(
            _reference_logits(model, result).argmax(axis=-1), dtype=np.int64
        ).reshape(1, 1)
        result = np.concatenate((result, next_token), axis=1)
    return result


@unittest.skipUnless(has_transformers_for("llama"), "requires LLaMA transformers")
class TestLlamaCircleRuntime(unittest.TestCase):
    """Compare disk-reloaded Circle execution with the complete eager model."""

    directory: tempfile.TemporaryDirectory[str]
    bundle_dir: Path
    manifest_path: Path
    model: torch.nn.Module
    prompt: np.ndarray

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.bundle_dir = Path(cls.directory.name) / "bundle"
        cls.model = _make_model()
        cls.prompt = np.array([[1, 5, 3]], dtype=np.int64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(23)
            cls.manifest_path = export_llama_bundle(
                q_model=copy.deepcopy(cls.model),
                max_seq_len=8,
                output_dir=cls.bundle_dir,
            )

    def setUp(self):
        self.runtime = LlamaCircleRuntime(self.bundle_dir)

    def test_teacher_forced_decode_matches_full_model(self):
        """Several cache updates preserve both logits and attention positions."""
        tokens = self.prompt.copy()
        actual = self.runtime.prefill(torch.from_numpy(tokens))
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, (1, 32))
        np.testing.assert_allclose(
            actual, _reference_logits(self.model, tokens), rtol=1e-4, atol=1e-6
        )
        self.assertEqual(self.runtime.past_len, tokens.shape[1])

        for token in (7, 9, 4):
            next_input = np.array([[token]], dtype=np.int64)
            tokens = np.concatenate((tokens, next_input), axis=1)
            actual = self.runtime.decode_one(next_input)
            np.testing.assert_allclose(
                actual, _reference_logits(self.model, tokens), rtol=1e-4, atol=1e-6
            )
            self.assertEqual(self.runtime.past_len, tokens.shape[1])

    def test_greedy_generation_matches_full_model_at_capacity(self):
        """The final allowed token is generated, with no silent truncation."""
        actual = self.runtime.generate(self.prompt, max_new_tokens=5)
        self.assertEqual(actual.dtype, np.int64)
        np.testing.assert_array_equal(
            actual, _reference_generate(self.model, self.prompt, 5)
        )
        self.assertEqual(actual.shape, (1, 8))

    def test_generation_stops_after_emitting_eos(self):
        next_token = int(_reference_logits(self.model, self.prompt).argmax())
        actual = self.runtime.generate(
            self.prompt, max_new_tokens=4, eos_token_id=next_token
        )
        np.testing.assert_array_equal(
            actual,
            np.concatenate((self.prompt, [[next_token]]), axis=1),
        )

    def test_zero_generation_returns_prompt_without_prefill(self):
        actual = self.runtime.generate(self.prompt, max_new_tokens=0)
        np.testing.assert_array_equal(actual, self.prompt)
        self.assertEqual(self.runtime.past_len, 0)
        self.runtime.prefill(self.prompt)
        other_prompt = np.array([[6, 8]], dtype=np.int64)
        actual = self.runtime.generate(other_prompt, max_new_tokens=0)
        np.testing.assert_array_equal(actual, other_prompt)
        self.assertEqual(self.runtime.past_len, self.prompt.shape[1])
        actual = self.runtime.decode_one(np.array([[7]], dtype=np.int64))
        np.testing.assert_allclose(
            actual,
            _reference_logits(self.model, np.concatenate((self.prompt, [[7]]), axis=1)),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_prefill_replaces_previous_cache_and_reset_requires_prefill(self):
        self.runtime.prefill(self.prompt)
        self.runtime.decode_one(np.array([[11]], dtype=np.int64))
        other_prompt = np.array([[6, 8]], dtype=np.int64)
        actual = self.runtime.prefill(other_prompt)
        np.testing.assert_allclose(
            actual,
            _reference_logits(self.model, other_prompt),
            rtol=1e-4,
            atol=1e-6,
        )
        self.assertEqual(self.runtime.past_len, 2)
        self.runtime.reset_cache()
        self.assertEqual(self.runtime.past_len, 0)
        with self.assertRaisesRegex((RuntimeError, ValueError), "prefill|cache"):
            self.runtime.decode_one(np.array([[1]], dtype=np.int64))

    def test_rejects_invalid_token_contracts(self):
        """Check rank, batch, dtype, vocabulary, and decode sequence length."""
        invalid = (
            np.array([1, 2], dtype=np.int64),
            np.array([[1], [2]], dtype=np.int64),
            np.empty((1, 0), dtype=np.int64),
            np.array([[1]], dtype=np.int32),
            np.array([[1.0]], dtype=np.float32),
            np.array([[-1]], dtype=np.int64),
            np.array([[32]], dtype=np.int64),
        )
        for tokens in invalid:
            with self.subTest(shape=tokens.shape, dtype=tokens.dtype):
                with self.assertRaises((ValueError, TypeError)):
                    self.runtime.prefill(tokens)
        self.runtime.prefill(self.prompt)
        with self.assertRaisesRegex(ValueError, "shape|token|length"):
            self.runtime.decode_one(np.array([[1, 2]], dtype=np.int64))

    def test_capacity_errors_and_full_length_prefill(self):
        """A full context can be scored; further cache growth is rejected."""
        full_prompt: np.ndarray = np.array([[1, 5, 3, 7, 9, 4, 6, 8]], dtype=np.int64)
        self.runtime.prefill(full_prompt[:, :-1])
        final_logits = self.runtime.decode_one(full_prompt[:, -1:])
        np.testing.assert_allclose(
            final_logits,
            _reference_logits(self.model, full_prompt),
            rtol=1e-4,
            atol=1e-6,
        )
        self.assertEqual(self.runtime.past_len, 8)
        with self.assertRaisesRegex(ValueError, "capacity|max_seq|length|context"):
            self.runtime.decode_one(np.array([[1]], dtype=np.int64))
        actual = self.runtime.prefill(full_prompt)
        np.testing.assert_allclose(
            actual,
            _reference_logits(self.model, full_prompt),
            rtol=1e-4,
            atol=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "capacity|max_seq|length|context"):
            self.runtime.decode_one(np.array([[1]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "capacity|max_seq|length|context"):
            self.runtime.prefill(np.ones((1, 9), dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "capacity|max_seq|length|context"):
            self.runtime.generate(self.prompt, max_new_tokens=6)
        with self.assertRaisesRegex(ValueError, "nonnegative|non-negative|negative"):
            self.runtime.generate(self.prompt, max_new_tokens=-1)

    def test_rejects_missing_and_incompatible_bundle(self):
        """Fail before execution when the bundle contract cannot be loaded."""
        for failure in ("manifest", "stage", "schema", "family", "geometry", "roles"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                bundle = Path(tmp) / "bundle"
                shutil.copytree(self.bundle_dir, bundle)
                manifest_path = bundle / self.manifest_path.name
                manifest = json.loads(manifest_path.read_text())
                if failure == "manifest":
                    manifest_path.unlink()
                elif failure == "stage":
                    (bundle / manifest["stages"]["token_embedding"]["file"]).unlink()
                else:
                    if failure == "schema":
                        manifest["schema_version"] = 999
                    elif failure == "family":
                        manifest["model_family"] = "unknown"
                    elif failure == "geometry":
                        manifest["hidden_size"] += 1
                    else:
                        roles = manifest["stages"]["layers"][0]["prefill"]["inputs"]
                        roles.reverse()
                    manifest_path.write_text(json.dumps(manifest))
                with self.assertRaises((ValueError, FileNotFoundError)):
                    LlamaCircleRuntime(bundle)

    def test_bundle_runs_in_clean_process_without_transformers(self):
        """Only saved Circle/NumPy artifacts cross the process boundary."""
        script = textwrap.dedent(
            """\
            import importlib.abc
            import json
            import sys
            import numpy as np

            class BlockModelLoader(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] == 'transformers':
                        raise AssertionError('Bundle execution imported transformers')
                    return None

            sys.meta_path.insert(0, BlockModelLoader())
            from tico.quantization.recipes.runtime.llama import LlamaCircleRuntime
            runtime = LlamaCircleRuntime(sys.argv[1])
            tokens = np.array(json.loads(sys.argv[2]), dtype=np.int64)
            result = runtime.generate(tokens, max_new_tokens=3)
            print('RESULT=' + json.dumps(result.tolist()))
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            shutil.copytree(self.bundle_dir, bundle)
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(bundle),
                    json.dumps(self.prompt.tolist()),
                ],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        serialized = next(
            line.removeprefix("RESULT=")
            for line in result.stdout.splitlines()
            if line.startswith("RESULT=")
        )
        np.testing.assert_array_equal(
            np.array(json.loads(serialized), dtype=np.int64),
            _reference_generate(self.model, self.prompt, 3),
        )

    def test_cli_generation_persists_tokens(self):
        """Exercise argument routing, real bundle inference, and saved JSON."""
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            shutil.copytree(self.bundle_dir, bundle)
            output_json = Path(tmp) / "generation.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tico.quantization.examples.generate",
                    "--bundle",
                    str(bundle),
                    "--input-ids",
                    "1,5,3",
                    "--max-new-tokens",
                    "3",
                    "--output-json",
                    str(output_json),
                ],
                cwd=tmp,
                env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                text=True,
                capture_output=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            saved = json.loads(output_json.read_text())
        self.assertEqual(saved, json.loads(result.stdout))
        self.assertEqual(saved["execution_mode"], "native")
        self.assertEqual(saved["input_token_ids"], self.prompt[0].tolist())
        expected = _reference_generate(self.model, self.prompt, 3)[0].tolist()
        self.assertEqual(saved["token_ids"], expected)
        self.assertEqual(saved["generated_token_ids"], expected[3:])

    def test_cli_generates_text_with_local_tokenizer(self):
        """Encode and decode a real prompt with a completely local tokenizer."""
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        raw = Tokenizer(
            WordLevel(
                vocab={f"token_{index}": index for index in range(32)},
                unk_token="token_0",
            )
        )
        raw.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=raw,
            unk_token="token_0",
            pad_token="token_0",
            eos_token="token_2",
        )
        prompt = "token_1 token_5 token_3"
        self.assertEqual(tokenizer.encode(prompt), self.prompt[0].tolist())
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            shutil.copytree(self.bundle_dir, bundle)
            tokenizer.save_pretrained(bundle / "tokenizer")
            manifest_path = bundle / self.manifest_path.name
            manifest = json.loads(manifest_path.read_text())
            manifest["tokenizer"] = "tokenizer"
            manifest_path.write_text(json.dumps(manifest))
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tico.quantization.examples.generate",
                    "--bundle",
                    str(bundle),
                    "--prompt",
                    prompt,
                    "--max-new-tokens",
                    "3",
                ],
                cwd=tmp,
                env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
                text=True,
                capture_output=True,
                timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = json.loads(result.stdout)
        expected = _reference_generate(self.model, self.prompt, 3)[0].tolist()
        self.assertEqual(output["input_token_ids"], self.prompt[0].tolist())
        self.assertEqual(output["token_ids"], expected)
        self.assertEqual(output["generated_token_ids"], expected[3:])
        self.assertEqual(
            output["text"], tokenizer.decode(expected[3:], skip_special_tokens=True)
        )


@unittest.skipUnless(has_transformers_for("llama"), "requires LLaMA transformers")
class TestFakeQuantizedLlamaCircleRuntime(unittest.TestCase):
    """Preserve frozen quantization semantics through saved stage boundaries."""

    directory: tempfile.TemporaryDirectory[str]
    bundle_dir: Path
    manifest_path: Path
    model: torch.nn.Module
    prompt: np.ndarray

    @classmethod
    def setUpClass(cls):
        from tico.quantization import convert, prepare
        from tico.quantization.config.ptq import PTQConfig

        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.bundle_dir = Path(cls.directory.name) / "bundle"
        cls.prompt = np.array([[1, 5, 3]], dtype=np.int64)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(29)
            prepared = prepare(
                _make_model(num_hidden_layers=1),
                PTQConfig(model_args={"profile": "npu_export"}),
            )
            with torch.no_grad():
                prepared(torch.tensor([[1, 5, 3, 7, 9, 4, 6, 8]]), use_cache=False)
                # Cache inputs and delta outputs have their own observers. Exercise
                # the same prefill/decode lifecycle as the exported stage bundle.
                output = prepared(
                    torch.from_numpy(cls.prompt), use_cache=True, return_dict=True
                )
                for token in (7, 9):
                    output = prepared(
                        torch.tensor([[token]]),
                        past_key_values=output.past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
            cls.model = convert(prepared).eval()
            cls.manifest_path = export_llama_bundle(
                q_model=copy.deepcopy(cls.model),
                max_seq_len=8,
                output_dir=cls.bundle_dir,
            )

    def test_fake_quantized_teacher_forced_logits_and_generation(self):
        """The numerical reference uses the converted model, not the FP model."""
        manifest = json.loads(self.manifest_path.read_text())
        self.assertEqual(manifest["execution_mode"], "fake_quantize")
        runtime = LlamaCircleRuntime(self.bundle_dir)
        tokens = self.prompt.copy()
        actual = runtime.prefill(tokens)
        for token in (7, 9):
            np.testing.assert_allclose(
                actual, _reference_logits(self.model, tokens), rtol=1e-4, atol=1e-6
            )
            next_input = np.array([[token]], dtype=np.int64)
            tokens = np.concatenate((tokens, next_input), axis=1)
            actual = runtime.decode_one(next_input)
        np.testing.assert_allclose(
            actual, _reference_logits(self.model, tokens), rtol=1e-4, atol=1e-6
        )
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(
            runtime.generate(self.prompt, max_new_tokens=3),
            _reference_generate(self.model, self.prompt, 3),
        )

    def test_quantized_artifacts_require_explicit_fake_quantize_mode(self):
        """Integer Circle tensors must not be presented as native float graphs."""
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            shutil.copytree(self.bundle_dir, bundle)
            path = bundle / self.manifest_path.name
            manifest = json.loads(path.read_text())
            manifest["execution_mode"] = "native"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "quantized|dtype"):
                LlamaCircleRuntime(bundle)


if __name__ == "__main__":
    unittest.main()
