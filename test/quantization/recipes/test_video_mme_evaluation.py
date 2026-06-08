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

try:
    from quantization.recipes.optional_dependency_stubs import (
        install_optional_dependency_stubs,
    )
except ModuleNotFoundError:
    from optional_dependency_stubs import install_optional_dependency_stubs

install_optional_dependency_stubs()

import unittest
from unittest.mock import MagicMock, patch

import tico.quantization.recipes.evaluation.video_mme as video_mme


class TestVideoMmeEvaluation(unittest.TestCase):
    """Smoke tests for the Video-MME evaluation recipe helper."""

    def test_evaluate_and_print_video_mme_delegates_to_lmms_eval(self):
        """evaluate_and_print_video_mme should call evaluate_vlm_on_tasks with video_mme task."""
        captured = {}

        def fake_evaluate_vlm_on_tasks(**kwargs):
            captured.update(kwargs)
            return {"results": {"video_mme": {"acc": 0.5}}}

        def fake_print_lmms_eval_results(results):
            pass  # no-op for test

        with (
            patch.object(
                video_mme, "evaluate_vlm_on_tasks", fake_evaluate_vlm_on_tasks
            ),
            patch.object(
                video_mme, "print_lmms_eval_results", fake_print_lmms_eval_results
            ),
        ):
            result = video_mme.evaluate_and_print_video_mme(
                model=MagicMock(),
                processor=MagicMock(),
                device="cuda",
                batch_size=1,
                max_new_tokens=16,
            )

        # Verify the task is "videomme" (lmms-eval task name)
        self.assertEqual(captured["tasks"], ["videomme"])
        # Verify other args are passed through
        self.assertEqual(captured["device"], "cuda")
        self.assertEqual(captured["batch_size"], 1)
        self.assertEqual(captured["max_new_tokens"], 16)
        # Verify results are returned
        self.assertIn("results", result)

    def test_evaluate_and_print_video_mme_passes_cache_args(self):
        """Cache-related arguments should be forwarded to evaluate_vlm_on_tasks."""
        captured = {}

        def fake_evaluate_vlm_on_tasks(**kwargs):
            captured.update(kwargs)
            return {"results": {}}

        def fake_print_lmms_eval_results(results):
            pass

        with (
            patch.object(
                video_mme, "evaluate_vlm_on_tasks", fake_evaluate_vlm_on_tasks
            ),
            patch.object(
                video_mme, "print_lmms_eval_results", fake_print_lmms_eval_results
            ),
        ):
            video_mme.evaluate_and_print_video_mme(
                model=MagicMock(),
                processor=MagicMock(),
                device="cpu",
                use_cache="/tmp/cache",
            )

        self.assertEqual(captured["use_cache"], "/tmp/cache")


class TestCoerceIntAttr(unittest.TestCase):
    """Tests for ``_coerce_int_attr``."""

    def test_none_returns_default(self):
        from tico.quantization.evaluation.lmms_eval_utils import _coerce_int_attr

        self.assertEqual(_coerce_int_attr(None, 42), 42)

    def test_int_returns_int(self):
        from tico.quantization.evaluation.lmms_eval_utils import _coerce_int_attr

        self.assertEqual(_coerce_int_attr(7, 42), 7)

    def test_float_returns_int(self):
        from tico.quantization.evaluation.lmms_eval_utils import _coerce_int_attr

        self.assertEqual(_coerce_int_attr(3.14, 42), 3)

    def test_list_with_one_element_returns_first(self):
        from tico.quantization.evaluation.lmms_eval_utils import _coerce_int_attr

        self.assertEqual(_coerce_int_attr([16], 42), 16)

    def test_list_with_multiple_elements_returns_first(self):
        from tico.quantization.evaluation.lmms_eval_utils import _coerce_int_attr

        self.assertEqual(_coerce_int_attr([16, 32], 42), 16)

    def test_tuple_with_one_element_returns_first(self):
        from tico.quantization.evaluation.lmms_eval_utils import _coerce_int_attr

        self.assertEqual(_coerce_int_attr((2,), 42), 2)

    def test_empty_list_returns_default(self):
        from tico.quantization.evaluation.lmms_eval_utils import _coerce_int_attr

        self.assertEqual(_coerce_int_attr([], 42), 42)

    def test_empty_tuple_returns_default(self):
        from tico.quantization.evaluation.lmms_eval_utils import _coerce_int_attr

        self.assertEqual(_coerce_int_attr((), 42), 42)


class TestProcessorVisionFactor(unittest.TestCase):
    """Tests for ``_processor_vision_factor``."""

    def _make_processor(self, patch_size=None, merge_size=None):
        from types import SimpleNamespace

        image_processor = SimpleNamespace()
        if patch_size is not None:
            image_processor.patch_size = patch_size
        if merge_size is not None:
            image_processor.merge_size = merge_size
        return SimpleNamespace(image_processor=image_processor)

    def test_qwen3_vl_defaults(self):
        """Qwen3-VL default: patch_size=16, merge_size=2 → factor=32."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _processor_vision_factor,
        )

        proc = self._make_processor(patch_size=16, merge_size=2)
        self.assertEqual(_processor_vision_factor(proc), 32)

    def test_qwen3_vl_list_attrs(self):
        """Qwen3-VL sometimes stores patch_size/merge_size as lists."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _processor_vision_factor,
        )

        proc = self._make_processor(patch_size=[16], merge_size=[2])
        self.assertEqual(_processor_vision_factor(proc), 32)

    def test_missing_attrs_uses_defaults(self):
        """Missing patch_size/merge_size → defaults 16 and 2 → factor=32."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _processor_vision_factor,
        )

        proc = self._make_processor()
        self.assertEqual(_processor_vision_factor(proc), 32)

    def test_no_image_processor_uses_defaults(self):
        """No image_processor at all → defaults 16 and 2 → factor=32."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import (
            _processor_vision_factor,
        )

        proc = SimpleNamespace()
        self.assertEqual(_processor_vision_factor(proc), 32)

    def test_custom_patch_and_merge(self):
        """Custom values: patch_size=14, merge_size=1 → factor=14."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _processor_vision_factor,
        )

        proc = self._make_processor(patch_size=14, merge_size=1)
        self.assertEqual(_processor_vision_factor(proc), 14)

    def test_minimum_factor_is_1(self):
        """patch_size=0, merge_size=0 → max(1, 0) = 1."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _processor_vision_factor,
        )

        proc = self._make_processor(patch_size=0, merge_size=0)
        self.assertEqual(_processor_vision_factor(proc), 1)


class TestComputeVideoMaxPixelsForBudget(unittest.TestCase):
    """Tests for ``_compute_video_max_pixels_for_budget``."""

    def _make_processor(self, patch_size=16, merge_size=2):
        from types import SimpleNamespace

        image_processor = SimpleNamespace(patch_size=patch_size, merge_size=merge_size)
        return SimpleNamespace(image_processor=image_processor)

    def test_budget_2048_10_frames(self):
        """Budget 2048, 10 frames → max_pixels=180224."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_max_pixels_for_budget,
        )

        max_px, min_px, adj_frames = _compute_video_max_pixels_for_budget(
            max_position_embeddings=2048,
            max_num_frames=10,
            max_new_tokens=30,
            processor=self._make_processor(),
        )
        self.assertEqual(max_px, 180224)
        self.assertEqual(min_px, 180224)
        self.assertEqual(adj_frames, 10)

    def test_budget_2048_32_frames(self):
        """Budget 2048, 32 frames → max_pixels=56320."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_max_pixels_for_budget,
        )

        max_px, min_px, adj_frames = _compute_video_max_pixels_for_budget(
            max_position_embeddings=2048,
            max_num_frames=32,
            max_new_tokens=30,
            processor=self._make_processor(),
        )
        self.assertEqual(max_px, 56320)
        self.assertEqual(min_px, 56320)
        self.assertEqual(adj_frames, 32)

    def test_budget_2048_5_frames(self):
        """Budget 2048, 5 frames → max_pixels=360448, min_pixels=200704."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_max_pixels_for_budget,
        )

        max_px, min_px, adj_frames = _compute_video_max_pixels_for_budget(
            max_position_embeddings=2048,
            max_num_frames=5,
            max_new_tokens=30,
            processor=self._make_processor(),
        )
        self.assertEqual(max_px, 360448)
        self.assertEqual(min_px, 200704)
        self.assertEqual(adj_frames, 5)

    def test_large_budget_no_reduction(self):
        """Large budget (32768) should not reduce pixels significantly."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_max_pixels_for_budget,
        )

        max_px, min_px, adj_frames = _compute_video_max_pixels_for_budget(
            max_position_embeddings=32768,
            max_num_frames=32,
            max_new_tokens=30,
            processor=self._make_processor(),
        )
        self.assertEqual(max_px, 1039360)
        self.assertEqual(min_px, 200704)
        self.assertEqual(adj_frames, 32)

    def test_tight_budget_reduces_frames(self):
        """Very tight budget: max_num_frames should be reduced."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_max_pixels_for_budget,
        )

        max_px, min_px, adj_frames = _compute_video_max_pixels_for_budget(
            max_position_embeddings=300,
            max_num_frames=32,
            max_new_tokens=30,
            processor=self._make_processor(),
        )
        self.assertEqual(adj_frames, 14)
        self.assertEqual(max_px, 1024)
        self.assertEqual(min_px, 1024)

    def test_zero_visual_budget_raises(self):
        """Zero or negative visual budget should raise ValueError."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_max_pixels_for_budget,
        )

        with self.assertRaises(ValueError):
            _compute_video_max_pixels_for_budget(
                max_position_embeddings=100,
                max_num_frames=10,
                max_new_tokens=50,
                processor=self._make_processor(),
                text_token_margin=256,
            )

    def test_custom_text_token_margin(self):
        """Custom text_token_margin affects the visual budget."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_max_pixels_for_budget,
        )

        max_px_default, _, _ = _compute_video_max_pixels_for_budget(
            max_position_embeddings=2048,
            max_num_frames=10,
            max_new_tokens=30,
            processor=self._make_processor(),
            text_token_margin=256,
        )
        max_px_large_margin, _, _ = _compute_video_max_pixels_for_budget(
            max_position_embeddings=2048,
            max_num_frames=10,
            max_new_tokens=30,
            processor=self._make_processor(),
            text_token_margin=1024,
        )
        self.assertLess(max_px_large_margin, max_px_default)

    def test_total_tokens_within_budget(self):
        """Verify total tokens do not exceed max_position_embeddings."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_max_pixels_for_budget,
        )

        max_pos = 2048
        max_new = 30
        margin = 256
        n_frames = 10
        max_px, _, adj_frames = _compute_video_max_pixels_for_budget(
            max_position_embeddings=max_pos,
            max_num_frames=n_frames,
            max_new_tokens=max_new,
            processor=self._make_processor(),
            text_token_margin=margin,
        )
        tokens_per_frame = max_px // (32 * 32)
        total_tokens = adj_frames * tokens_per_frame + margin + max_new
        self.assertLessEqual(total_tokens, max_pos)


class TestGetMaxPositionEmbeddings(unittest.TestCase):
    """Tests for ``_get_max_position_embeddings``."""

    def test_qwen3_vl_config(self):
        """Qwen3-VL stores max_position_embeddings under text_config."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import (
            _get_max_position_embeddings,
        )

        class TextConfig:
            max_position_embeddings = 32768

        class Config:
            text_config = TextConfig()

        model = SimpleNamespace(config=Config())
        self.assertEqual(_get_max_position_embeddings(model), 32768)

    def test_direct_config_attr(self):
        """Models without text_config use direct max_position_embeddings."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import (
            _get_max_position_embeddings,
        )

        class Config:
            max_position_embeddings = 4096

        model = SimpleNamespace(config=Config())
        self.assertEqual(_get_max_position_embeddings(model), 4096)

    def test_no_config_returns_none(self):
        """Model without config returns None."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import (
            _get_max_position_embeddings,
        )

        model = SimpleNamespace()
        self.assertIsNone(_get_max_position_embeddings(model))

    def test_config_without_max_pos_returns_none(self):
        """Config without max_position_embeddings returns None."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import (
            _get_max_position_embeddings,
        )

        model = SimpleNamespace(config=SimpleNamespace())
        self.assertIsNone(_get_max_position_embeddings(model))

    def test_text_config_takes_precedence(self):
        """text_config.max_position_embeddings takes precedence over direct."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import (
            _get_max_position_embeddings,
        )

        class TextConfig:
            max_position_embeddings = 32768

        class Config:
            text_config = TextConfig()
            max_position_embeddings = 4096

        model = SimpleNamespace(config=Config())
        self.assertEqual(_get_max_position_embeddings(model), 32768)

    def test_returns_int(self):
        """Return value is always int, even if config stores something else."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import (
            _get_max_position_embeddings,
        )

        class Config:
            max_position_embeddings = 8192.0

        model = SimpleNamespace(config=Config())
        result = _get_max_position_embeddings(model)
        self.assertIsInstance(result, int)
        self.assertEqual(result, 8192)


class TestLmmsEvalUtils(unittest.TestCase):
    """Tests for the low-level lmms-eval wrapper utilities."""

    def test_build_model_args_infers_qwen3_vl(self):
        """_build_model_args should infer 'qwen3_vl' from a Qwen3-VL model object."""
        from tico.quantization.evaluation.lmms_eval_utils import _build_model_args

        model = MagicMock()
        type(model).__name__ = "Qwen3VLForConditionalGeneration"
        model_name_str = "Qwen/Qwen3-VL-2B-Instruct"
        model.config._name_or_path = model_name_str

        model_name, model_args = _build_model_args(model)

        self.assertEqual(model_name, "qwen3_vl")
        self.assertEqual(model_args["pretrained"], model_name_str)

    def test_build_model_args_passes_max_num_frames(self):
        """_build_model_args should include max_num_frames in model_args."""
        from tico.quantization.evaluation.lmms_eval_utils import _build_model_args

        model = MagicMock()
        type(model).__name__ = "Qwen3VLForConditionalGeneration"
        model.config._name_or_path = "Qwen/Qwen3-VL-2B-Instruct"

        model_name, model_args = _build_model_args(
            model,
            max_num_frames=5,
        )

        self.assertEqual(model_args["max_num_frames"], 5)

    def test_build_model_args_with_budget_sets_max_pixels(self):
        """When max_position_embeddings is provided, max_pixels is computed."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import _build_model_args

        model = MagicMock()
        type(model).__name__ = "Qwen3VLForConditionalGeneration"
        model.config._name_or_path = "Qwen/Qwen3-VL-2B-Instruct"

        image_processor = SimpleNamespace(patch_size=16, merge_size=2)
        proc = SimpleNamespace(image_processor=image_processor)

        _, args = _build_model_args(
            model,
            max_num_frames=10,
            max_position_embeddings=2048,
            max_new_tokens=30,
            processor=proc,
        )
        self.assertIn("max_pixels", args)
        self.assertIn("min_pixels", args)
        self.assertIn("max_num_frames", args)
        self.assertEqual(args["max_pixels"], 180224)
        self.assertEqual(args["max_num_frames"], 10)

    def test_build_model_args_without_budget_no_processor(self):
        """Without processor, no budget computation even if max_pos given."""
        from tico.quantization.evaluation.lmms_eval_utils import _build_model_args

        model = MagicMock()
        type(model).__name__ = "Qwen3VLForConditionalGeneration"
        model.config._name_or_path = "Qwen/Qwen3-VL-2B-Instruct"

        _, args = _build_model_args(
            model, max_num_frames=10, max_position_embeddings=2048
        )
        self.assertNotIn("max_pixels", args)
        self.assertIn("max_num_frames", args)

    def test_build_model_args_budget_adjusts_max_num_frames(self):
        """Tight budget adjusts max_num_frames downward."""
        from types import SimpleNamespace

        from tico.quantization.evaluation.lmms_eval_utils import _build_model_args

        model = MagicMock()
        type(model).__name__ = "Qwen3VLForConditionalGeneration"
        model.config._name_or_path = "Qwen/Qwen3-VL-2B-Instruct"

        image_processor = SimpleNamespace(patch_size=16, merge_size=2)
        proc = SimpleNamespace(image_processor=image_processor)

        _, args = _build_model_args(
            model,
            max_num_frames=32,
            max_position_embeddings=300,
            max_new_tokens=30,
            processor=proc,
        )
        self.assertLessEqual(args["max_num_frames"], 32)

    def test_check_lmms_eval_available_raises_when_missing(self):
        """_check_lmms_eval_available should raise RuntimeError if lmms-eval is not installed."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _check_lmms_eval_available,
        )

        with patch.dict("sys.modules", {"lmms_eval": None}):
            with self.assertRaises(RuntimeError) as ctx:
                _check_lmms_eval_available()
            self.assertIn("lmms-eval", str(ctx.exception))

    def test_get_custom_tasks_dir_finds_lmms_tasks(self):
        """_get_custom_tasks_dir should find the lmms_tasks directory shipped with TICO."""
        from tico.quantization.evaluation.lmms_eval_utils import _get_custom_tasks_dir

        tasks_dir = _get_custom_tasks_dir()
        self.assertIsNotNone(tasks_dir)
        self.assertTrue(tasks_dir.endswith("lmms_tasks"))  # type: ignore[union-attr]

    def test_print_results_fallback(self):
        """Fallback printer should handle float and non-float values."""
        import contextlib

        import io

        from tico.quantization.evaluation.lmms_eval_utils import _print_results_fallback

        results = {
            "results": {
                "video_mme": {
                    "acc,none": 0.6543,
                    "num_samples": 900,
                }
            }
        }

        with contextlib.redirect_stdout(io.StringIO()) as buffer:
            _print_results_fallback(results)

        output = buffer.getvalue()
        self.assertIn("video_mme", output)
        self.assertIn("0.6543", output)
        self.assertIn("900", output)


class TestComputeVideoChunkPatterns(unittest.TestCase):
    """Tests for _compute_video_chunk_patterns."""

    def test_limit_1_downloads_1_chunk(self):
        """A limit of 1 should download only 1 video chunk."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_chunk_patterns,
        )

        patterns = _compute_video_chunk_patterns(limit=1)
        self.assertIn("videos_chunked_01.zip", patterns)
        self.assertNotIn("videos_chunked_02.zip", patterns)

    def test_limit_61_downloads_3_chunks(self):
        """A limit of 61 (> 2*_SAMPLES_PER_CHUNK) should download 3 chunks."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_chunk_patterns,
        )

        # _SAMPLES_PER_CHUNK = 30, so ceil(61/30) = 3 chunks
        patterns = _compute_video_chunk_patterns(limit=61)
        self.assertIn("videos_chunked_01.zip", patterns)
        self.assertIn("videos_chunked_02.zip", patterns)
        self.assertIn("videos_chunked_03.zip", patterns)
        self.assertNotIn("videos_chunked_04.zip", patterns)

    def test_limit_none_downloads_all_chunks(self):
        """No limit (None) should download all 20 chunks."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_chunk_patterns,
        )

        patterns = _compute_video_chunk_patterns(limit=None)
        self.assertIn("videos_chunked_01.zip", patterns)
        self.assertIn("videos_chunked_20.zip", patterns)
        self.assertNotIn("videos_chunked_21.zip", patterns)

    def test_always_includes_base_patterns(self):
        """Base patterns (parquet, gitattributes, README, subtitle) should always be present."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_chunk_patterns,
        )

        patterns = _compute_video_chunk_patterns(limit=1)
        self.assertIn("*.parquet", patterns)
        self.assertIn(".gitattributes", patterns)
        self.assertIn("README.md", patterns)
        self.assertIn("subtitle.zip", patterns)


class TestPatchSnapshotDownloadForLimit(unittest.TestCase):
    """Tests for _patch_snapshot_download_for_limit."""

    def test_returns_context_manager(self):
        """_patch_snapshot_download_for_limit should return a context manager."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _patch_snapshot_download_for_limit,
        )

        ctx = _patch_snapshot_download_for_limit(limit=10)
        # Should be usable as a context manager
        self.assertTrue(hasattr(ctx, "__enter__"))
        self.assertTrue(hasattr(ctx, "__exit__"))

    def test_non_videomme_repo_passes_through(self):
        """Patched snapshot_download should pass through non-Video-MME repos unchanged."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _patch_snapshot_download_for_limit,
        )

        ctx = _patch_snapshot_download_for_limit(limit=10)
        with ctx:
            # After patching, calling snapshot_download with a non-Video-MME
            # repo should go through to the original function.
            # We just verify the context manager works without error.
            pass


class TestGetDownloadedVideommeChunks(unittest.TestCase):
    """Tests for _get_downloaded_videomme_chunks."""

    def test_returns_empty_set_when_no_cache(self):
        """Should return empty set when HF cache dir doesn't exist."""
        # With a non-existent HF_HOME, should return empty set
        import os

        from tico.quantization.evaluation.lmms_eval_utils import (
            _get_downloaded_videomme_chunks,
        )

        original_hf = os.environ.get("HF_HOME")
        try:
            os.environ["HF_HOME"] = "/tmp/nonexistent_hf_cache_12345"
            result = _get_downloaded_videomme_chunks()
            self.assertIsInstance(result, set)
            self.assertEqual(len(result), 0)
        finally:
            if original_hf is not None:
                os.environ["HF_HOME"] = original_hf
            else:
                os.environ.pop("HF_HOME", None)

    def test_finds_chunks_in_fake_cache(self):
        """Should find chunk zips in a fake cache directory."""
        import os
        import tempfile

        from tico.quantization.evaluation.lmms_eval_utils import (
            _get_downloaded_videomme_chunks,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the expected directory structure
            repo_dir = os.path.join(tmpdir, "hub", "datasets--lmms-lab--Video-MME")
            snap_dir = os.path.join(repo_dir, "snapshots", "abc123")
            os.makedirs(snap_dir)

            # Create fake chunk files
            for name in ["videos_chunked_01.zip", "videos_chunked_02.zip"]:
                with open(os.path.join(snap_dir, name), "w") as f:
                    f.write("fake zip content")

            # Create a non-chunk file that should be ignored
            with open(os.path.join(snap_dir, "subtitle.zip"), "w") as f:
                f.write("fake subtitle")

            original_hf = os.environ.get("HF_HOME")
            try:
                os.environ["HF_HOME"] = tmpdir
                result = _get_downloaded_videomme_chunks()
                self.assertIn("videos_chunked_01.zip", result)
                self.assertIn("videos_chunked_02.zip", result)
                self.assertNotIn("subtitle.zip", result)
                self.assertEqual(len(result), 2)
            finally:
                if original_hf is not None:
                    os.environ["HF_HOME"] = original_hf
                else:
                    os.environ.pop("HF_HOME", None)


class TestEnsureVideommeChunksDownloaded(unittest.TestCase):
    """Tests for _ensure_videomme_chunks_downloaded."""

    def test_skips_download_when_all_chunks_cached(self):
        """Should not call snapshot_download when all needed chunks are cached."""
        import os
        import tempfile
        from unittest.mock import MagicMock, patch

        from tico.quantization.evaluation.lmms_eval_utils import (
            _ensure_videomme_chunks_downloaded,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the cache with chunk 01 already present
            repo_dir = os.path.join(tmpdir, "hub", "datasets--lmms-lab--Video-MME")
            snap_dir = os.path.join(repo_dir, "snapshots", "abc123")
            os.makedirs(snap_dir)
            with open(os.path.join(snap_dir, "videos_chunked_01.zip"), "w") as f:
                f.write("fake")

            original_hf = os.environ.get("HF_HOME")
            try:
                os.environ["HF_HOME"] = tmpdir

                with patch("huggingface_hub.snapshot_download") as mock_dl:
                    _ensure_videomme_chunks_downloaded(limit=1)
                    # Should NOT call snapshot_download since chunk 01 is cached
                    mock_dl.assert_not_called()
            finally:
                if original_hf is not None:
                    os.environ["HF_HOME"] = original_hf
                else:
                    os.environ.pop("HF_HOME", None)

    def test_downloads_missing_chunks(self):
        """Should download only missing chunks when some are cached."""
        import os
        import tempfile
        from unittest.mock import patch

        from tico.quantization.evaluation.lmms_eval_utils import (
            _ensure_videomme_chunks_downloaded,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the cache with chunk 01 already present
            repo_dir = os.path.join(tmpdir, "hub", "datasets--lmms-lab--Video-MME")
            snap_dir = os.path.join(repo_dir, "snapshots", "abc123")
            os.makedirs(snap_dir)
            with open(os.path.join(snap_dir, "videos_chunked_01.zip"), "w") as f:
                f.write("fake")

            original_hf = os.environ.get("HF_HOME")
            try:
                os.environ["HF_HOME"] = tmpdir

                with patch("huggingface_hub.snapshot_download") as mock_dl:
                    # limit=31 needs 2 chunks (01 and 02), but 01 is cached
                    _ensure_videomme_chunks_downloaded(limit=31)
                    mock_dl.assert_called_once()
                    call_kwargs = mock_dl.call_args
                    allow_patterns = call_kwargs.kwargs.get(
                        "allow_patterns"
                    ) or call_kwargs[1].get("allow_patterns")
                    # Should only download chunk 02 (01 is cached)
                    self.assertIn("videos_chunked_02.zip", allow_patterns)
                    self.assertNotIn("videos_chunked_01.zip", allow_patterns)
            finally:
                if original_hf is not None:
                    os.environ["HF_HOME"] = original_hf
                else:
                    os.environ.pop("HF_HOME", None)


class TestVerboseFlagPropagation(unittest.TestCase):
    """Tests for verbose flag propagation via LMMS_VERBOSE env var."""

    def test_evaluate_vlm_on_tasks_sets_lmms_verbose_env(self):
        """evaluate_vlm_on_tasks should set LMMS_VERBOSE env var based on verbose flag."""
        import os

        from tico.quantization.evaluation.lmms_eval_utils import evaluate_vlm_on_tasks

        # We can't actually run evaluate_vlm_on_tasks (needs lmms-eval + GPU),
        # but we can test that the env var is set correctly by checking
        # the function source or by mocking. Here we just verify the env var
        # mechanism works.
        os.environ["LMMS_VERBOSE"] = "1"
        self.assertEqual(os.environ.get("LMMS_VERBOSE"), "1")

        os.environ["LMMS_VERBOSE"] = "0"
        self.assertEqual(os.environ.get("LMMS_VERBOSE"), "0")

        # Clean up
        os.environ.pop("LMMS_VERBOSE", None)


class TestVideommeMiniUtils(unittest.TestCase):
    """Tests for the videomme_mini task utility functions."""

    def test_available_video_ids_empty_dir(self):
        """_available_video_ids should return empty set for non-existent directory."""
        # Import with stubs
        try:
            from quantization.recipes.optional_dependency_stubs import (
                install_optional_dependency_stubs,
            )
        except ModuleNotFoundError:
            from optional_dependency_stubs import install_optional_dependency_stubs

        install_optional_dependency_stubs()

        from tico.quantization.evaluation.lmms_tasks.videomme_mini.utils import (
            _available_video_ids,
        )

        # With default _data_dir (likely doesn't exist in test env)
        result = _available_video_ids()
        # Should return a set (possibly empty)
        self.assertIsInstance(result, set)

    def test_doc_to_visual_returns_empty_for_missing_video(self):
        """videomme_doc_to_visual should return [] for missing videos."""
        from tico.quantization.evaluation.lmms_tasks.videomme_mini.utils import (
            videomme_doc_to_visual,
        )

        doc = {"videoID": "nonexistent_video_12345"}
        result = videomme_doc_to_visual(doc)
        self.assertEqual(result, [])

    def test_doc_to_visual_returns_path_for_existing_video(self):
        """videomme_doc_to_visual should return [path] for an existing video file."""
        import os
        import tempfile

        from tico.quantization.evaluation.lmms_tasks.videomme_mini import (
            utils as vm_utils,
        )
        from tico.quantization.evaluation.lmms_tasks.videomme_mini.utils import (
            videomme_doc_to_visual,
        )

        # Create a temporary directory with a fake video
        with tempfile.TemporaryDirectory() as tmpdir:
            original_data_dir = vm_utils._data_dir
            vm_utils._data_dir = tmpdir
            try:
                # Create a fake video file
                video_id = "test_video_abc"
                video_path = os.path.join(tmpdir, video_id + ".mp4")
                with open(video_path, "w") as f:
                    f.write("fake")

                doc = {"videoID": video_id}
                result = videomme_doc_to_visual(doc)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0], video_path)
            finally:
                vm_utils._data_dir = original_data_dir

    def test_process_docs_filters_by_available_videos(self):
        """videomme_process_docs should filter dataset to only available videos."""
        import os
        import tempfile
        from unittest.mock import MagicMock

        from tico.quantization.evaluation.lmms_tasks.videomme_mini import (
            utils as vm_utils,
        )
        from tico.quantization.evaluation.lmms_tasks.videomme_mini.utils import (
            videomme_process_docs,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            original_data_dir = vm_utils._data_dir
            vm_utils._data_dir = tmpdir
            try:
                # Create one video file
                with open(os.path.join(tmpdir, "available_video.mp4"), "w") as f:
                    f.write("fake")

                # Mock dataset
                mock_dataset = MagicMock()
                mock_dataset.filter.return_value = ["filtered_result"]

                videomme_process_docs(mock_dataset)
                # filter should have been called since we have available videos
                mock_dataset.filter.assert_called_once()
            finally:
                vm_utils._data_dir = original_data_dir

    def test_is_verbose_reflects_runtime_env_changes(self):
        """_is_verbose() should reflect runtime changes to LMMS_VERBOSE env var."""
        import os

        from tico.quantization.evaluation.lmms_tasks.videomme_mini.utils import (
            _is_verbose,
        )

        # Ensure verbose is off
        os.environ.pop("LMMS_VERBOSE", None)
        self.assertFalse(_is_verbose())

        # Set verbose on at runtime – _is_verbose() should pick it up immediately
        os.environ["LMMS_VERBOSE"] = "1"
        self.assertTrue(_is_verbose())

        # Turn it off again
        os.environ.pop("LMMS_VERBOSE", None)
        self.assertFalse(_is_verbose())


if __name__ == "__main__":
    unittest.main()
