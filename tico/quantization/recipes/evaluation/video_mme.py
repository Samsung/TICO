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

"""Recipe helper for Video-MME evaluation via ``lmms-eval``.

This module provides :func:`evaluate_and_print_video_mme`, a thin wrapper
that delegates all benchmark-specific logic (video frame extraction, prompt
formatting, answer extraction, scoring) to the ``lmms-eval`` library.

The ``lmms-eval`` and ``decord`` packages must be installed separately::

    pip install lmms-eval decord
"""

from typing import Any

from tico.quantization.evaluation.lmms_eval_utils import (
    evaluate_vlm_on_tasks,
    print_lmms_eval_results,
)


def evaluate_and_print_video_mme(
    *,
    model: Any,
    processor: Any,
    device: str,
    batch_size: int = 1,
    max_num_frames: int = 32,
    max_new_tokens: int = 30,
    n_samples: int | None = None,
    use_cache: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a VLM on the Video-MME benchmark and print results.

    This function calls :func:`evaluate_vlm_on_tasks` with
    ``tasks=["videomme"]`` (or ``tasks=["videomme_mini"]`` when *mini*
    is ``True``) and prints the results in a formatted table.

    The *n_samples* parameter restricts the number of samples evaluated,
    which also limits how many videos are downloaded.  For example,
    ``n_samples=10`` will only download only first zip file and evaluate
    the first 10 samples. This uses a custom ``videomme_mini`` task
    YAML (shipped with TICO) that specifies ``test_split: test``.

    Args:
        model: A Hugging Face VLM (e.g. ``Qwen3_VLForConditionalGeneration``).
        processor: The matching ``AutoProcessor`` for the model.
        device: Device string for inference (e.g. ``"cuda"``, ``"cpu"``).
        batch_size: Batch size for generation.  Defaults to 1.
        max_num_frames: The maximum number of frames that will be extracted from the video uniformly.
        max_new_tokens: Maximum number of tokens to generate per sample.
        n_samples: If set, only evaluate the first *n_samples* samples.  This
            also limits how many videos are downloaded.  Defaults to
            ``None`` (download and evaluate all samples).
        use_cache: Optional path to an ``lmms-eval`` results cache directory.
        verbose: Whether to print detailed evaluation logs.

    Returns:
        A dictionary of evaluation results as returned by
        ``lmms_eval.evaluator.simple_evaluate``.

    Raises:
        RuntimeError: If ``lmms-eval`` is not installed.
    """

    results = evaluate_vlm_on_tasks(
        model=model,
        processor=processor,
        tasks=["videomme"],
        device=device,
        batch_size=batch_size,
        max_num_frames=max_num_frames,
        max_new_tokens=max_new_tokens,
        limit=n_samples,
        use_cache=use_cache,
        verbose=verbose,
    )

    print(f"\n=== Video-MME Evaluation (limit={n_samples}) ===")
    print_lmms_eval_results(results)

    return results
