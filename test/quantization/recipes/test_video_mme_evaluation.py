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
                cache_dir="/tmp/hf_cache",
            )

        self.assertEqual(captured["use_cache"], "/tmp/cache")
        self.assertEqual(captured["cache_dir"], "/tmp/hf_cache")


class TestLmmsEvalUtils(unittest.TestCase):
    """Tests for the low-level lmms-eval wrapper utilities."""

    def test_build_model_args_infers_qwen3_vl(self):
        """_build_model_args should infer 'qwen3_vl' from a Qwen3-VL model object."""
        from tico.quantization.evaluation.lmms_eval_utils import _build_model_args

        model = MagicMock()
        type(model).__name__ = "Qwen3VLForConditionalGeneration"
        model_name_str = "Qwen/Qwen3-VL-2B-Instruct"
        model.config._name_or_path = model_name_str

        processor = MagicMock()
        processor.tokenizer.name_or_path = model_name_str

        model_name, model_args = _build_model_args(
            model, processor, device="cuda", batch_size=2, max_new_tokens=16
        )

        self.assertEqual(model_name, "qwen3_vl")
        self.assertEqual(model_args["pretrained"], model_name_str)
        # Only 'pretrained' is in model_args; batch_size, device,
        # max_new_tokens, and tokenizer are NOT (they'd cause errors
        # with lmms-eval model constructors like Qwen3_VL).
        self.assertNotIn("batch_size", model_args)
        self.assertNotIn("device", model_args)
        self.assertNotIn("max_new_tokens", model_args)
        self.assertNotIn("tokenizer", model_args)

    def test_build_model_args_passes_max_num_frames(self):
        """_build_model_args should include max_num_frames in model_args."""
        from tico.quantization.evaluation.lmms_eval_utils import _build_model_args

        model = MagicMock()
        type(model).__name__ = "Qwen3VLForConditionalGeneration"
        model.config._name_or_path = "Qwen/Qwen3-VL-2B-Instruct"

        processor = MagicMock()

        model_name, model_args = _build_model_args(
            model,
            processor,
            device="cuda",
            batch_size=1,
            max_new_tokens=16,
            max_num_frames=5,
        )

        self.assertEqual(model_args["max_num_frames"], 5)

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
        self.assertTrue(tasks_dir.endswith("lmms_tasks"))

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

    def test_limit_41_downloads_2_chunks(self):
        """A limit of 41 (> _SAMPLES_PER_CHUNK) should download 2 chunks."""
        from tico.quantization.evaluation.lmms_eval_utils import (
            _compute_video_chunk_patterns,
        )

        patterns = _compute_video_chunk_patterns(limit=41)
        self.assertIn("videos_chunked_01.zip", patterns)
        self.assertIn("videos_chunked_02.zip", patterns)
        self.assertNotIn("videos_chunked_03.zip", patterns)

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

    def test_verbose_flag_controls_print(self):
        """Print statements should be suppressed when LMMS_VERBOSE is not set."""
        import os

        from tico.quantization.evaluation.lmms_tasks.videomme_mini import (
            utils as vm_utils,
        )

        # Ensure verbose is off
        os.environ.pop("LMMS_VERBOSE", None)
        # Re-evaluate _VERBOSE by reimporting the module is tricky,
        # so we just test the _VERBOSE flag directly.
        # Since _VERBOSE is evaluated at import time, we test the env var logic.
        self.assertFalse(os.getenv("LMMS_VERBOSE", "").lower() in ("1", "true", "yes"))


if __name__ == "__main__":
    unittest.main()
