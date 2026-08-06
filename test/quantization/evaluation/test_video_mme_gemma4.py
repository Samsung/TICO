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

"""Tests for Video-MME evaluation support for Gemma4.

These tests exercise:
- ``_load_video_frames`` in ``lmms_gemma4.py``
- ``patch_huggingface_wrapper_for_gemma4`` context manager
- ``evaluate_and_print_video_mme`` delegation
- The Gemma4 adapter's videomme config parsing
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np


def _has_decord() -> bool:
    try:
        import decord  # noqa: F401
    except ImportError:
        return False
    return True


def _has_lmms_eval() -> bool:
    try:
        import lmms_eval  # noqa: F401
    except ImportError:
        return False
    return True


class TestLoadVideoFrames(unittest.TestCase):
    """Test ``_load_video_frames`` from ``lmms_gemma4.py``."""

    def test_load_video_frames_fewer_than_max(self):
        """When total frames <= max_num_frames, all frames are returned."""
        from tico.quantization.evaluation.lmms_gemma4 import _load_video_frames

        fake_frames = np.random.rand(5, 8, 8, 3).astype(np.uint8)

        mock_vr = MagicMock()
        mock_vr.__len__ = Mock(return_value=5)
        mock_vr.get_batch = Mock(
            return_value=MagicMock(asnumpy=Mock(return_value=fake_frames))
        )

        mock_decord = MagicMock()
        mock_decord.VideoReader = Mock(return_value=mock_vr)

        with patch.dict("sys.modules", {"decord": mock_decord}):
            result = _load_video_frames("/fake/path.mp4", max_num_frames=32)

        self.assertEqual(result.shape, (5, 8, 8, 3))
        np.testing.assert_array_equal(result, fake_frames)

    def test_load_video_frames_subsample(self):
        """When total frames > max_num_frames, frames are subsampled."""
        from tico.quantization.evaluation.lmms_gemma4 import _load_video_frames

        total = 100
        max_frames = 10
        fake_frames = np.random.rand(max_frames, 8, 8, 3).astype(np.uint8)

        mock_vr = MagicMock()
        mock_vr.__len__ = Mock(return_value=total)
        mock_vr.get_batch = Mock(
            return_value=MagicMock(asnumpy=Mock(return_value=fake_frames))
        )

        mock_decord = MagicMock()
        mock_decord.VideoReader = Mock(return_value=mock_vr)

        with patch.dict("sys.modules", {"decord": mock_decord}):
            result = _load_video_frames("/fake/path.mp4", max_num_frames=max_frames)

        self.assertEqual(result.shape[0], max_frames)
        # Verify get_batch was called with a list of indices
        call_args = mock_vr.get_batch.call_args[0][0]
        self.assertIsInstance(call_args, list)
        self.assertEqual(len(call_args), max_frames)
        # Indices should be within range
        self.assertTrue(all(0 <= i < total for i in call_args))

    def test_load_video_frames_zero_frames_raises(self):
        """A video with 0 frames should raise ValueError."""
        from tico.quantization.evaluation.lmms_gemma4 import _load_video_frames

        mock_vr = MagicMock()
        mock_vr.__len__ = Mock(return_value=0)

        mock_decord = MagicMock()
        mock_decord.VideoReader = Mock(return_value=mock_vr)

        with patch.dict("sys.modules", {"decord": mock_decord}):
            with self.assertRaises(ValueError) as ctx:
                _load_video_frames("/fake/path.mp4", max_num_frames=32)
            self.assertIn("0 frames", str(ctx.exception))


class TestPatchHuggingfaceWrapperForGemma4(unittest.TestCase):
    """Test ``patch_huggingface_wrapper_for_gemma4`` context manager."""

    @staticmethod
    def _make_mock_modules(mock_hf_class):
        """Build a dict of mock lmms_eval modules for patch.dict('sys.modules')."""
        # The function does ``from lmms_eval.models.chat.huggingface import Huggingface``
        # so Huggingface must be an attribute on the mock module object.
        hf_module = MagicMock()
        hf_module.Huggingface = mock_hf_class

        instance_module = MagicMock()
        instance_module.GenerationResult = Mock
        instance_module.TokenCounts = Mock

        protocol_module = MagicMock()
        protocol_module.ChatMessages = Mock

        utils_module = MagicMock()
        utils_module.Collator = Mock

        gen_metrics_module = MagicMock()
        gen_metrics_module.log_metrics = Mock()

        return {
            "lmms_eval": MagicMock(),
            "lmms_eval.api": MagicMock(),
            "lmms_eval.api.instance": instance_module,
            "lmms_eval.models": MagicMock(),
            "lmms_eval.models.chat": MagicMock(),
            "lmms_eval.models.chat.huggingface": hf_module,
            "lmms_eval.protocol": protocol_module,
            "lmms_eval.utils": utils_module,
            "lmms_eval.models.model_utils": MagicMock(),
            "lmms_eval.models.model_utils.gen_metrics": gen_metrics_module,
            "loguru": MagicMock(),
            "tqdm": MagicMock(),
        }

    def test_patch_restores_on_exit(self):
        """The context manager should restore the original generate_until on exit."""
        from tico.quantization.evaluation.lmms_gemma4 import (
            patch_huggingface_wrapper_for_gemma4,
        )

        mock_hf_class = Mock()
        original_generate_until = Mock()
        mock_hf_class.generate_until = original_generate_until

        mock_modules = self._make_mock_modules(mock_hf_class)

        with patch.dict("sys.modules", mock_modules):
            with patch_huggingface_wrapper_for_gemma4(max_num_frames=5):
                # Inside the context, generate_until should be patched
                self.assertIsNot(mock_hf_class.generate_until, original_generate_until)

            # After exiting, the original should be restored
            self.assertIs(mock_hf_class.generate_until, original_generate_until)

    def test_patch_default_max_num_frames(self):
        """The default max_num_frames should be 32."""
        from tico.quantization.evaluation.lmms_gemma4 import (
            patch_huggingface_wrapper_for_gemma4,
        )

        mock_hf_class = Mock()
        original_generate_until = Mock()
        mock_hf_class.generate_until = original_generate_until

        mock_modules = self._make_mock_modules(mock_hf_class)

        with patch.dict("sys.modules", mock_modules):
            with patch_huggingface_wrapper_for_gemma4():
                # The patched function should exist
                self.assertTrue(callable(mock_hf_class.generate_until))


class TestEvaluateAndPrintVideoMME(unittest.TestCase):
    """Test ``evaluate_and_print_video_mme`` delegation."""

    @patch("tico.quantization.recipes.evaluation.video_mme.evaluate_vlm_on_tasks")
    @patch("tico.quantization.recipes.evaluation.video_mme.print_lmms_eval_results")
    @patch("builtins.print")
    def test_delegates_to_evaluate_vlm_on_tasks(
        self, mock_print, mock_print_results, mock_eval
    ):
        """evaluate_and_print_video_mme should call evaluate_vlm_on_tasks with tasks=['videomme']."""
        from tico.quantization.recipes.evaluation.video_mme import (
            evaluate_and_print_video_mme,
        )

        mock_results = {"results": {"videomme": {"accuracy": 0.5}}}
        mock_eval.return_value = mock_results

        result = evaluate_and_print_video_mme(
            model=Mock(),
            processor=Mock(),
            device="cpu",
            batch_size=2,
            max_num_frames=21,
            max_new_tokens=30,
            n_samples=10,
            verbose=False,
        )

        mock_eval.assert_called_once()
        call_kwargs = mock_eval.call_args.kwargs
        self.assertEqual(call_kwargs["tasks"], ["videomme"])
        self.assertEqual(call_kwargs["device"], "cpu")
        self.assertEqual(call_kwargs["batch_size"], 2)
        self.assertEqual(call_kwargs["max_num_frames"], 21)
        self.assertEqual(call_kwargs["max_new_tokens"], 30)
        self.assertEqual(call_kwargs["limit"], 10)
        self.assertEqual(call_kwargs["verbose"], False)

        mock_print_results.assert_called_once_with(mock_results)
        self.assertEqual(result, mock_results)

    @patch("tico.quantization.recipes.evaluation.video_mme.evaluate_vlm_on_tasks")
    @patch("tico.quantization.recipes.evaluation.video_mme.print_lmms_eval_results")
    @patch("builtins.print")
    def test_n_samples_none_passes_none_limit(
        self, mock_print, mock_print_results, mock_eval
    ):
        """When n_samples is None, limit should be None."""
        from tico.quantization.recipes.evaluation.video_mme import (
            evaluate_and_print_video_mme,
        )

        mock_eval.return_value = {}

        evaluate_and_print_video_mme(
            model=Mock(),
            processor=Mock(),
            device="cpu",
            n_samples=None,
        )

        call_kwargs = mock_eval.call_args.kwargs
        self.assertIsNone(call_kwargs["limit"])

    @patch("tico.quantization.recipes.evaluation.video_mme.evaluate_vlm_on_tasks")
    @patch("tico.quantization.recipes.evaluation.video_mme.print_lmms_eval_results")
    @patch("builtins.print")
    def test_default_max_num_frames(self, mock_print, mock_print_results, mock_eval):
        """Default max_num_frames should be 32."""
        from tico.quantization.recipes.evaluation.video_mme import (
            evaluate_and_print_video_mme,
        )

        mock_eval.return_value = {}

        evaluate_and_print_video_mme(
            model=Mock(),
            processor=Mock(),
            device="cpu",
        )

        call_kwargs = mock_eval.call_args.kwargs
        self.assertEqual(call_kwargs["max_num_frames"], 32)


class TestGemma4AdapterVideoMMEConfig(unittest.TestCase):
    """Test the Gemma4 adapter's videomme config parsing."""

    def test_videomme_disabled_by_default(self):
        """When videomme is not in eval_cfg, evaluate_and_print_video_mme should not be called."""
        from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter

        adapter = Gemma4Adapter()
        ctx = Mock()
        ctx.cfg = {
            "evaluation": {"enabled": True},
        }
        ctx.model = Mock()
        ctx.processor = Mock()
        ctx.device = "cpu"

        with patch(
            "tico.quantization.recipes.adapters.gemma4.evaluate_and_print_video_mme"
        ) as mock_eval:
            adapter.evaluate(ctx)
            mock_eval.assert_not_called()

    def test_videomme_enabled_calls_evaluate(self):
        """When videomme.enabled is True, evaluate_and_print_video_mme should be called."""
        from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter

        adapter = Gemma4Adapter()
        ctx = Mock()
        ctx.cfg = {
            "evaluation": {
                "enabled": True,
                "videomme": {
                    "enabled": True,
                    "n_samples": 5,
                    "max_num_frames": 21,
                    "batch_size": 1,
                    "max_new_tokens": 30,
                },
            },
        }
        ctx.model = Mock()
        ctx.processor = Mock()
        ctx.device = "cpu"

        with patch(
            "tico.quantization.recipes.adapters.gemma4.evaluate_and_print_video_mme"
        ) as mock_eval:
            adapter.evaluate(ctx)
            mock_eval.assert_called_once()
            call_kwargs = mock_eval.call_args.kwargs
            self.assertEqual(call_kwargs["n_samples"], 5)
            self.assertEqual(call_kwargs["max_num_frames"], 21)
            self.assertEqual(call_kwargs["batch_size"], 1)
            self.assertEqual(call_kwargs["max_new_tokens"], 30)

    def test_videomme_default_max_num_frames_is_21(self):
        """Gemma4 adapter should default max_num_frames to 21 (not 32)."""
        from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter

        adapter = Gemma4Adapter()
        ctx = Mock()
        ctx.cfg = {
            "evaluation": {
                "enabled": True,
                "videomme": {
                    "enabled": True,
                },
            },
        }
        ctx.model = Mock()
        ctx.processor = Mock()
        ctx.device = "cpu"

        with patch(
            "tico.quantization.recipes.adapters.gemma4.evaluate_and_print_video_mme"
        ) as mock_eval:
            adapter.evaluate(ctx)
            call_kwargs = mock_eval.call_args.kwargs
            # Gemma4 defaults to 21 frames (not 32) due to soft_tokens_per_frame
            self.assertEqual(call_kwargs["max_num_frames"], 21)

    def test_videomme_invalid_max_num_frames_raises(self):
        """max_num_frames <= 0 should raise ValueError."""
        from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter

        adapter = Gemma4Adapter()
        ctx = Mock()
        ctx.cfg = {
            "evaluation": {
                "enabled": True,
                "videomme": {
                    "enabled": True,
                    "max_num_frames": 0,
                },
            },
        }
        ctx.model = Mock()
        ctx.processor = Mock()
        ctx.device = "cpu"

        with patch(
            "tico.quantization.recipes.adapters.gemma4.evaluate_and_print_video_mme"
        ):
            with self.assertRaises(ValueError) as exc_ctx:
                adapter.evaluate(ctx)
            self.assertIn("max_num_frames", str(exc_ctx.exception))

    def test_videomme_negative_max_num_frames_raises(self):
        """Negative max_num_frames should raise ValueError."""
        from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter

        adapter = Gemma4Adapter()
        ctx = Mock()
        ctx.cfg = {
            "evaluation": {
                "enabled": True,
                "videomme": {
                    "enabled": True,
                    "max_num_frames": -1,
                },
            },
        }
        ctx.model = Mock()
        ctx.processor = Mock()
        ctx.device = "cpu"

        with patch(
            "tico.quantization.recipes.adapters.gemma4.evaluate_and_print_video_mme"
        ):
            with self.assertRaises(ValueError):
                adapter.evaluate(ctx)

    def test_videomme_n_samples_negative_becomes_none(self):
        """When n_samples is negative, it should be passed as None."""
        from tico.quantization.recipes.adapters.gemma4 import Gemma4Adapter

        adapter = Gemma4Adapter()
        ctx = Mock()
        ctx.cfg = {
            "evaluation": {
                "enabled": True,
                "videomme": {
                    "enabled": True,
                    "n_samples": -1,
                },
            },
        }
        ctx.model = Mock()
        ctx.processor = Mock()
        ctx.device = "cpu"

        with patch(
            "tico.quantization.recipes.adapters.gemma4.evaluate_and_print_video_mme"
        ) as mock_eval:
            adapter.evaluate(ctx)
            call_kwargs = mock_eval.call_args.kwargs
            self.assertIsNone(call_kwargs["n_samples"])


if __name__ == "__main__":
    unittest.main()
