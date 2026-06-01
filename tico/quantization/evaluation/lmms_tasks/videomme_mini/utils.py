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

# Video-MME mini task utilities.
#
# This module re-exports most functions from the upstream ``lmms-eval``
# Video-MME task but overrides two key functions:
#
# * ``videomme_doc_to_visual`` – handles missing video files gracefully
#   (returns an empty list instead of calling ``sys.exit``).
#
# * ``videomme_process_docs`` – filters the dataset to only include
#   samples whose video files have been downloaded and extracted.
#   This is essential when using ``allow_patterns`` to download only a
#   subset of the video zip files (e.g. only ``videos_chunked_01.zip``).

import os
import sys

from lmms_eval.tasks.videomme.utils import (  # noqa: F401
    CATEGORIES,
    SUB_CATEGORIES,
    TASK_CATEGORIES,
    VIDEO_TYPE,
    videomme_aggregate_results,
    videomme_process_results,
)

# We override videomme_doc_to_text below to add debug printing,
# so we do NOT re-export it from upstream.

# Resolve the cache directory the same way the upstream utils.py does.
_hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
_hf_home = os.path.expanduser(_hf_home)
_cache_dir = os.path.join(_hf_home, "videomme")
_data_dir = os.path.join(_cache_dir, "data")

# Verbose flag: set LMMS_VERBOSE=1 (or pass --verbose to the quantization
# script) to enable debug printing.  By default, printing is suppressed.
# NOTE: We use a helper function instead of a module-level variable so that
# changes to the LMMS_VERBOSE environment variable at runtime are reflected
# immediately (e.g. when evaluate_vlm_on_tasks sets it before calling
# simple_evaluate).


def _is_verbose() -> bool:
    return os.getenv("LMMS_VERBOSE", "").lower() in ("1", "true", "yes")


def _available_video_ids() -> set[str]:
    """Return the set of video IDs that have been downloaded and extracted.

    A video ID is the stem of the video filename (e.g. ``"HwnB8aCn8yE"``
    from ``"HwnB8aCn8yE.mp4"``).
    """
    if not os.path.isdir(_data_dir):
        return set()
    return {
        os.path.splitext(f)[0]
        for f in os.listdir(_data_dir)
        if f.endswith((".mp4", ".MP4", ".mkv"))
    }


def videomme_process_docs(dataset):
    """Filter the dataset to only include samples with available videos.

    When using ``allow_patterns`` to download only a subset of the video
    zip files (e.g. only ``videos_chunked_01.zip``), many samples in the
    parquet metadata will reference videos that have not been downloaded.
    This filter removes those samples so that evaluation only runs on
    available data.
    """
    available = _available_video_ids()
    if not available:
        # If no videos are extracted yet, raise RuntimeError.
        raise RuntimeError("There are no avaiable videos from extracted zip files")
    return dataset.filter(lambda x: x["videoID"] in available)


def videomme_doc_to_visual(doc):
    """Return the video path for a sample, or an empty list if missing.

    The upstream ``videomme_doc_to_visual`` calls ``sys.exit()`` when a
    video file is not found.  This override returns an empty list instead,
    allowing evaluation to continue with the samples that *are* available.

    NOTE: Frame extraction is NOT done here.  The video path is returned
    as-is, and frame sampling is handled by the lmms-eval model wrapper
    (e.g. ``Qwen3_VL._subsample_video_inputs``) which uses the
    ``max_num_frames`` parameter.
    """
    video_path = os.path.join(_data_dir, doc["videoID"] + ".mp4")
    if os.path.exists(video_path):
        if _is_verbose():
            print(
                f"[INFO] doc_to_visual: returning video path: {video_path}",
                file=sys.stderr,
            )
        return [video_path]
    # Try alternate extensions
    for ext in (".MP4", ".mkv"):
        alt_path = os.path.join(_data_dir, doc["videoID"] + ext)
        if os.path.exists(alt_path):
            if _is_verbose():
                print(
                    f"[INFO] doc_to_visual: returning video path: {alt_path}",
                    file=sys.stderr,
                )
            return [alt_path]
    # Video not available – skip gracefully
    if _is_verbose():
        print(
            f"[INFO] Video not found: {video_path}, skipping sample",
            file=sys.stderr,
        )
    return []


# Track which samples have already been printed to avoid duplicate output.
# lmms-eval calls doc_to_text multiple times per sample (task construction,
# batching, generation), so we deduplicate by (videoID, question).
_printed_prompts: set[tuple[str, str]] = set()


def _reset_printed_prompts() -> None:
    """Clear the set of already-printed prompt keys."""
    _printed_prompts.clear()


def videomme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build the text prompt for a sample and print it for debugging.

    This is an override of the upstream ``videomme_doc_to_text`` that adds
    a ``print()`` call so you can see the final prompt sent to the model.
    Deduplicates by (videoID, question) to avoid printing the same sample
    multiple times (lmms-eval calls this function several times per sample).
    """
    from lmms_eval.tasks.videomme.utils import (
        videomme_doc_to_text as _upstream_doc_to_text,
    )

    full_prompt = _upstream_doc_to_text(doc, lmms_eval_specific_kwargs)

    # Print the final prompt for debugging (once per unique sample, only when verbose)
    if _is_verbose():
        video_id = doc.get("videoID", "<unknown>")
        question = doc.get("question", "")
        key = (video_id, question)
        if key not in _printed_prompts:
            _printed_prompts.add(key)
            print(
                f"\n{'='*60}\n"
                f"[PROMPT] videoID={video_id}\n"
                f"{'-'*60}\n"
                f"{full_prompt}\n"
                f"{'='*60}",
                file=sys.stderr,
            )

    return full_prompt
