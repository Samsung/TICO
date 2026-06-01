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

"""VLM evaluation utilities using the ``lmms-eval`` library.

The primary entry point is :func:`evaluate_vlm_on_tasks`, which accepts a
Hugging Face VLM + processor pair and delegates all benchmark-specific
logic (dataset loading, prompt formatting, answer extraction, scoring) to
``lmms-eval``.

Supported tasks include but are not limited to:

- ``video_mme`` – Video-MME video understanding benchmark
- ``mmstar`` – MMStar benchmark
- ``mathvista`` – MathVista benchmark
- ``mmbench`` – MMBench benchmark

Any task registered in ``lmms-eval`` can be passed via the *tasks*
parameter.
"""

import os
from typing import Any


def _check_lmms_eval_available() -> None:
    """Raise a clear error if ``lmms-eval`` is not installed."""
    try:
        import decord  # noqa: F401
        import lmms_eval  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "lmms-eval is required for VLM evaluation via lmms-eval. "
            "Install it with:  pip install lmms-eval decord"
        ) from exc


def _patch_lmms_eval_broken_tasks() -> None:
    """Work around packaging bugs in installed ``lmms-eval`` versions.

    Some versions of ``lmms-eval`` (notably v0.7.1) ship task directories
    that are missing YAML template files (``_default_template_yaml``,
    ``_default_template_*_yaml``, etc.) expected by :class:`TaskManager`
    during initialization.  This causes ``FileNotFoundError`` crashes that
    prevent *any* task from running.

    This function monkey-patches ``lmms_eval.utils.load_yaml_config`` so
    that missing YAML files are treated as empty configs instead of
    raising an exception.  The patch is idempotent and only applied once.
    """
    try:
        import lmms_eval.utils as lmms_utils
    except ImportError:
        return

    # Only patch once
    if getattr(lmms_utils, "_tico_patched", False):
        return

    _original_load_yaml_config = lmms_utils.load_yaml_config

    def _patched_load_yaml_config(yaml_path=None, mode="simple", **kwargs):
        try:
            return _original_load_yaml_config(yaml_path=yaml_path, mode=mode, **kwargs)
        except FileNotFoundError:
            # Return an empty config for missing YAML files so that
            # TaskManager initialization does not crash.
            return {}

    lmms_utils.load_yaml_config = _patched_load_yaml_config
    lmms_utils._tico_patched = True


def _compute_video_chunk_patterns(limit: int | None) -> list[str]:
    """Compute ``allow_patterns`` for Video-MME based on sample limit.

    Video-MME stores videos as ~20 zip files (~5 GB each, ~30 videos each)
    in the HuggingFace repository.  ``snapshot_download`` downloads **all**
    repo files by default, totalling ~100 GB.  When *limit* is set we only
    need a few samples, so downloading everything is wasteful.

    This function returns an ``allow_patterns`` list that includes the
    parquet metadata, subtitle zip, and enough video chunk zips to cover
    the requested number of samples.

    Args:
        limit: Maximum number of samples to evaluate.  ``None`` means
            download all video chunks.

    Returns:
        List of ``allow_patterns`` strings for ``snapshot_download``.
    """
    _SAMPLES_PER_CHUNK = 30  # approximate number of samples per zip
    _MAX_CHUNKS = 20  # maximum number of video chunk zips

    base_patterns = [
        "*.parquet",
        ".gitattributes",
        "README.md",
        "subtitle.zip",
    ]

    if limit is None:
        # No limit → download all video chunks
        video_patterns = [
            f"videos_chunked_{i:02d}.zip" for i in range(1, _MAX_CHUNKS + 1)
        ]
    else:
        # Calculate how many chunks we need based on the limit.
        # Each chunk contains ~30 videos, so we need ceil(limit / 30) chunks.
        import math

        num_chunks = min(math.ceil(limit / _SAMPLES_PER_CHUNK), _MAX_CHUNKS)
        num_chunks = max(num_chunks, 1)  # at least 1 chunk
        video_patterns = [
            f"videos_chunked_{i:02d}.zip" for i in range(1, num_chunks + 1)
        ]

    return base_patterns + video_patterns


def _get_downloaded_videomme_chunks() -> set[str]:
    """Check which Video-MME video chunk zips are already in the HF cache.

    Scans the HuggingFace cache directory for the Video-MME dataset repo
    and returns the set of chunk zip filenames that have already been
    downloaded (e.g. ``{"videos_chunked_01.zip", "videos_chunked_02.zip"}``).

    Returns:
        A set of zip filenames found in the cache, or an empty set if the
        cache directory does not exist yet.
    """
    import re

    try:
        import huggingface_hub
    except ImportError:
        return set()

    # The HF cache stores dataset repos under:
    #   <HF_HOME>/hub/datasets--lmms-lab--Video-MME/
    # Inside, the actual files are under blobs/ (content-addressed) but
    # snapshot directories contain symlinks.  We scan snapshots for the
    # zip filenames.
    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    repo_dir = os.path.join(hf_home, "hub", "datasets--lmms-lab--Video-MME")

    if not os.path.isdir(repo_dir):
        return set()

    downloaded = set()
    _chunk_re = re.compile(r"^videos_chunked_\d+\.zip$")
    for root, dirs, files in os.walk(repo_dir):
        for f in files:
            if _chunk_re.match(f):
                # Check that the file has actual content (not just a pointer)
                fpath = os.path.join(root, f)
                if os.path.getsize(fpath) > 0:
                    downloaded.add(f)

    return downloaded


def _ensure_videomme_chunks_downloaded(limit: int | None) -> None:
    """Download any missing Video-MME zip chunks needed for the given limit.

    This function proactively downloads only the chunks that are not already
    in the HuggingFace cache.  It is incremental: if you previously ran with
    ``limit=30`` (1 chunk) and now run with ``limit=60`` (2 chunks), only
    the second chunk is downloaded.

    When *limit* is ``None``, all 20 chunks are downloaded (if not already
    cached).

    This must be called **before** ``simple_evaluate`` so that the chunks
    are available when lmms-eval's dataset loading runs.

    Args:
        limit: Maximum number of samples to evaluate.  ``None`` means
            download all chunks.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return  # huggingface_hub not available; nothing to download

    needed_patterns = _compute_video_chunk_patterns(limit)
    # Separate base patterns (parquet, etc.) from video chunk zips
    needed_chunks = {p for p in needed_patterns if p.startswith("videos_chunked_")}
    needed_base = [p for p in needed_patterns if not p.startswith("videos_chunked_")]

    # Check which chunks are already in the cache
    already_downloaded = _get_downloaded_videomme_chunks()

    # Compute which chunks are missing
    missing_chunks = needed_chunks - already_downloaded

    if not missing_chunks and already_downloaded:
        print(
            f"[INFO] All {len(needed_chunks)} needed Video-MME chunks already "
            f"downloaded ({len(already_downloaded)} in cache)."
        )
        return

    # Download missing chunks + base patterns (parquet, subtitle, etc.)
    download_patterns = needed_base + sorted(missing_chunks)

    if missing_chunks:
        print(
            f"[INFO] Downloading {len(missing_chunks)} missing Video-MME chunks: "
            f"{sorted(missing_chunks)}  "
            f"({len(already_downloaded)} already in cache)"
        )

    snapshot_download(
        "lmms-lab/Video-MME",
        allow_patterns=download_patterns,
        repo_type="dataset",
    )


def _patch_snapshot_download_for_limit(limit: int | None):
    """Patch ``huggingface_hub.snapshot_download`` to limit Video-MME downloads.

    Video-MME stores videos as ~20 zip files (~5 GB each) in the HuggingFace
    repository.  ``snapshot_download`` downloads **all** repo files by default,
    totalling ~100 GB.  When *limit* is set we only need a few samples, so
    downloading everything is wasteful.

    This patch intercepts ``snapshot_download`` for the Video-MME repo and
    adds ``allow_patterns`` to download only the parquet metadata + enough
    video chunk zips to cover the requested number of samples.  Each chunk
    contains ~30 videos (~5 GB), so:

    - ``limit ≤ 30``  → 1 chunk  (~5 GB)
    - ``limit ≤ 60``  → 2 chunks (~10 GB)
    - ``limit ≤ 90``  → 3 chunks (~15 GB)
    - etc.

    The ``videomme_mini`` task's ``process_docs`` then filters the dataset to
    only include samples whose videos have been extracted.

    The patch is automatically undone when the returned context manager exits.

    Args:
        limit: Maximum number of samples to evaluate.  ``None`` means no
            limit (download all chunks).

    Returns:
        A context manager that restores the original ``snapshot_download``
        on exit.
    """
    import contextlib

    try:
        import huggingface_hub
    except ImportError:

        @contextlib.contextmanager
        def _noop():
            yield

        return _noop()

    _original_module_fn = huggingface_hub.snapshot_download

    _VIDEOMME_REPO_IDS = {
        "lmms-lab/Video-MME",
        "lmms-lab/video-mme",
    }

    # Compute allow_patterns based on the limit.
    _zip_patterns = _compute_video_chunk_patterns(limit)

    def _patched_snapshot_download(repo_id, *args, **kwargs):
        """Patched snapshot_download that limits Video-MME downloads."""
        if repo_id not in _VIDEOMME_REPO_IDS:
            return _original_module_fn(repo_id, *args, **kwargs)

        # For Video-MME: download only parquet + needed zip files
        # NOTE: The parameter is ``allow_patterns``
        allow_patterns = kwargs.get("allow_patterns")
        if allow_patterns is None:
            kwargs["allow_patterns"] = _zip_patterns

        return _original_module_fn(repo_id, *args, **kwargs)

    # Apply the monkey-patch to the module-level function.
    huggingface_hub.snapshot_download = _patched_snapshot_download

    # Additionally, ``lmms_eval.api.task`` does
    # ``from huggingface_hub import snapshot_download`` at import time,
    # creating a LOCAL reference that bypasses the module-level attribute.
    _lmms_task_refs = {}
    try:
        import lmms_eval.api.task as _lmms_task

        if hasattr(_lmms_task, "snapshot_download"):
            _lmms_task_refs[_lmms_task] = _lmms_task.snapshot_download
            _lmms_task.snapshot_download = _patched_snapshot_download
    except ImportError:
        pass

    @contextlib.contextmanager
    def _restore():
        try:
            yield
        finally:
            huggingface_hub.snapshot_download = _original_module_fn
            for _mod, _orig_fn in _lmms_task_refs.items():
                _mod.snapshot_download = _orig_fn

    return _restore()


def _patch_from_pretrained_to_reuse_model(model, processor):
    """Patch ``from_pretrained`` to return the already-loaded model and processor.

    ``lmms-eval`` always instantiates the model from scratch via
    ``from_pretrained(pretrained, ...)``, which downloads ``model.safetensors``
    and loads it into GPU memory a second time.  This wastes time, disk space,
    and GPU memory (likely causing OOM).

    This function monkey-patches ``from_pretrained`` on the relevant
    ``transformers`` model classes so that the first call returns the
    already-loaded *model* object, and patches ``AutoProcessor.from_pretrained``
    and ``AutoTokenizer.from_pretrained`` to return the already-loaded
    *processor* and its ``.tokenizer`` attribute.

    The patches are automatically undone when the returned context manager exits.

    Args:
        model: The already-loaded Hugging Face model.
        processor: The already-loaded Hugging Face processor.

    Returns:
        A context manager that restores the original ``from_pretrained``
        methods on exit.
    """
    import contextlib

    from transformers import AutoProcessor, AutoTokenizer

    # --- Ensure the model has a .config attribute ---
    # When the model is a PTQ wrapper (has .wrapped), lmms-eval's
    # Qwen3_VL.__init__ accesses self._model.config, which would fail
    # because the wrapper class doesn't define it.  We dynamically add
    # it as an instance attribute that delegates to the inner model's
    # config.  This avoids modifying ptq_wrapper.py.
    _inner = getattr(model, "wrapped", None)
    if _inner is not None and not hasattr(model, "config"):
        model.config = _inner.config

    # --- Patch model class from_pretrained ---
    # Collect all model classes that might be used by lmms-eval
    _model_classes = []
    try:
        from transformers import Qwen3VLForConditionalGeneration as _Q3

        _model_classes.append(_Q3)
    except ImportError:
        pass
    try:
        from transformers import Qwen3VLMoeForConditionalGeneration as _Q3M

        _model_classes.append(_Q3M)
    except ImportError:
        pass
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _Q25

        _model_classes.append(_Q25)
    except ImportError:
        pass
    try:
        from transformers import Qwen2VLForConditionalGeneration as _Q2

        _model_classes.append(_Q2)
    except ImportError:
        pass

    _original_model_fps = {cls: cls.from_pretrained for cls in _model_classes}

    # Replace from_pretrained on each class with a function that returns
    # the already-loaded model.

    for cls in _model_classes:
        # We must capture the original unbound function for the fallback.
        _orig_fn = cls.from_pretrained

        def _make_patched_fp(orig_fn):
            def _patched_from_pretrained(cls_arg, *args, **kwargs):
                return model

            return _patched_from_pretrained

        cls.from_pretrained = classmethod(_make_patched_fp(_orig_fn))

    # --- Patch AutoProcessor.from_pretrained ---
    _original_processor_fp = AutoProcessor.from_pretrained

    def _patched_processor_fp(cls_arg, *args, **kwargs):
        return processor

    AutoProcessor.from_pretrained = classmethod(_patched_processor_fp)

    # --- Patch AutoTokenizer.from_pretrained ---
    _original_tokenizer_fp = AutoTokenizer.from_pretrained
    _tokenizer = getattr(processor, "tokenizer", None)

    def _patched_tokenizer_fp(cls_arg, *args, **kwargs):
        return _tokenizer

    AutoTokenizer.from_pretrained = classmethod(_patched_tokenizer_fp)

    @contextlib.contextmanager
    def _restore():
        try:
            yield
        finally:
            # Restore original from_pretrained methods
            for cls, orig_fp in _original_model_fps.items():
                cls.from_pretrained = orig_fp
            AutoProcessor.from_pretrained = _original_processor_fp
            AutoTokenizer.from_pretrained = _original_tokenizer_fp

    return _restore()


def _patch_subsample_video_inputs_for_debug():
    """
    This monkey-patches the ``_subsample_video_inputs`` method on the
    lmms-eval ``Qwen3_VL`` model wrapper to print debug information.

    The patch is automatically undone when the returned context manager exits.

    Returns:
        A context manager that restores the original method on exit.
    """
    import contextlib

    try:
        from lmms_eval.models.simple.qwen3_vl import Qwen3_VL
    except ImportError:
        return contextlib.nullcontext()

    if not hasattr(Qwen3_VL, "_subsample_video_inputs"):
        return contextlib.nullcontext()
    _original_subsample = Qwen3_VL._subsample_video_inputs

    def _patched_subsample(self, video_inputs, video_metadatas=None):
        if video_inputs is not None:
            for i, vi in enumerate(video_inputs):
                total = vi.shape[0]
                max_nf = self.max_num_frames
                print(
                    f"[INFO] _subsample_video_inputs: video {i}: "
                    f"total_frames={total}, max_num_frames={max_nf}, "
                    f"will subsample to ~{min(total, max_nf)} frames",
                    flush=True,
                )
        return _original_subsample(self, video_inputs, video_metadatas)

    Qwen3_VL._subsample_video_inputs = _patched_subsample

    @contextlib.contextmanager
    def _restore():
        try:
            yield
        finally:
            Qwen3_VL._subsample_video_inputs = _original_subsample

    return _restore()


def evaluate_vlm_on_tasks(
    model: Any,
    processor: Any,
    tasks: list[str] | str,
    device: str = "cuda",
    batch_size: int = 1,
    max_num_frames: int = 32,
    max_new_tokens: int | None = None,
    limit: int | None = None,
    use_cache: str | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate a VLM on one or more ``lmms-eval`` tasks.

    This function calls :func:`lmms_eval.evaluator.simple_evaluate`, delegating
    all benchmark-specific logic (dataset loading, prompt formatting, answer
    extraction, scoring) to the library.

    The model is passed via ``model_args`` as a dict so that ``lmms-eval``
    instantiates the correct model wrapper from its registry.  The
    ``model`` parameter should be a string name recognised by
    ``lmms-eval`` (e.g. ``"qwen3_vl"``).  If a *model* object is
    passed instead, it is auto-detected and the appropriate ``lmms-eval``
    model name is inferred from the model class.

    Args:
        model: A Hugging Face VLM (e.g. ``Qwen3_VLForConditionalGeneration``)
            **or** a string model name recognised by ``lmms-eval`` (e.g.
            ``"qwen3_vl"``).  If a model object is passed, the function
            attempts to infer the ``lmms-eval`` model name automatically.
            If the model has a ``.wrapped`` attribute (fake-quant wrapper),
            the inner model is used automatically.
        processor: The matching ``AutoProcessor`` for the model.  Its
            ``.tokenizer`` attribute is passed to the ``lmms-eval`` model
            wrapper.
        tasks: A single task name (e.g. ``"video_mme"``) or a list of
            task names.  Any task registered in ``lmms-eval`` is accepted.
        device: Device string for inference (e.g. ``"cuda"``, ``"cpu"``).
        batch_size: Batch size for generation.  Defaults to 1.
        max_new_tokens: Maximum number of tokens to generate per sample.
        use_cache: Optional path to an ``lmms-eval`` results cache directory.
            When set, results are cached and can be reused across runs.
        verbose: Whether to print detailed evaluation logs.
        **kwargs: Additional keyword arguments forwarded to
            ``lmms_eval.evaluator.simple_evaluate``.

    Returns:
        A dictionary of evaluation results as returned by
        ``lmms_eval.evaluator.simple_evaluate``.  The structure includes
        ``"results"`` (per-task metrics) and ``"configs"`` (task
        configuration details).

    Raises:
        RuntimeError: If ``lmms-eval`` is not installed.
    """
    _check_lmms_eval_available()
    _patch_lmms_eval_broken_tasks()

    import contextlib

    from lmms_eval.evaluator import simple_evaluate

    if isinstance(model, str):
        model_name = model
        model_args_dict = {}
        if max_num_frames is not None:
            model_args_dict["max_num_frames"] = max_num_frames
        _model_ctx = contextlib.nullcontext()
    else:
        # Unwrap fake-quant wrapper if present
        inner_model = getattr(model, "wrapped", model)
        # Determine the lmms-eval model name and build model_args
        model_name, model_args_dict = _build_model_args(inner_model, max_num_frames)
        # Patch ``from_pretrained`` so that ``lmms-eval`` reuses the
        # already-loaded model and processor instead of downloading and
        # loading them a second time.
        _model_ctx = _patch_from_pretrained_to_reuse_model(model, processor)

    # Propagate verbose flag to custom task utils via environment variable.
    # The videomme_mini utils.py checks LMMS_VERBOSE to decide whether to
    # print prompts, video paths, etc.
    os.environ["LMMS_VERBOSE"] = "1" if verbose else "0"

    print(f"evaluate_vlm_on_tasks: model={model_name}, max_num_frames={max_num_frames}")
    print(f"model_args: {model_args_dict}")

    # Normalise tasks to a list
    if isinstance(tasks, str):
        tasks_list = [t.strip() for t in tasks.split(",") if t.strip()]
    else:
        tasks_list = list(tasks)

    # Partial Video-MME downloads require videomme_mini, which filters samples by
    # local video availability.
    if limit is not None:
        tasks_list = [
            "videomme_mini" if task in ("videomme", "video_mme") else task
            for task in tasks_list
        ]

    # Convert model_args dict to the key=value string format expected by
    # ``simple_evaluate`` → ``create_from_arg_string`` →
    # ``simple_parse_args_string``.
    model_args_str = ",".join(f"{k}={v}" for k, v in model_args_dict.items())

    eval_kwargs: dict[str, Any] = {
        "model": model_name,
        "model_args": model_args_str,
        "tasks": tasks_list,
        "batch_size": batch_size,
        "device": device,
        "verbosity": "INFO" if verbose else "WARNING",
    }

    if limit is not None:
        eval_kwargs["limit"] = limit
    if use_cache is not None:
        eval_kwargs["use_cache"] = use_cache
    if max_new_tokens is not None:
        gen_kwargs = f"max_new_tokens={max_new_tokens}"
        eval_kwargs["gen_kwargs"] = gen_kwargs

    # Auto-register custom task directories (e.g. videomme_mini) by
    # creating a ``TaskManager`` with ``include_path`` pointing to the
    # ``lmms_tasks`` directory shipped with TICO.  ``simple_evaluate``
    # does not accept ``include_path`` directly, but it does accept a
    # ``task_manager`` parameter.
    custom_tasks_dir = _get_custom_tasks_dir()
    if custom_tasks_dir is not None and "task_manager" not in kwargs:
        from lmms_eval.tasks import TaskManager

        task_manager = TaskManager(
            verbosity=eval_kwargs.get("verbosity", "DEBUG"),
            include_path=custom_tasks_dir,
            model_name=model_name,
        )
        eval_kwargs["task_manager"] = task_manager

    # Forward any additional kwargs (may override task_manager)
    eval_kwargs.update(kwargs)

    # Patch _subsample_video_inputs to add debug logging
    _subsample_ctx = _patch_subsample_video_inputs_for_debug()

    # Proactively download any missing Video-MME video chunk zips before
    # lmms-eval runs.  This is incremental: if some chunks are already in
    # the HF cache from a previous run (possibly with a smaller limit), only
    # the newly needed chunks are downloaded.
    _is_videomme = any(
        t in ("videomme", "video_mme", "videomme_mini") for t in tasks_list
    )
    if _is_videomme:
        _ensure_videomme_chunks_downloaded(limit)  # type: ignore[arg-type]

        # Reset the videomme_mini utils module's _printed_prompts set so
        # that prompts are printed again for this evaluation run.
        try:
            from tico.quantization.evaluation.lmms_tasks.videomme_mini.utils import (
                _reset_printed_prompts,
            )

            _reset_printed_prompts()
        except ImportError:
            pass

    # When *limit* is set, also patch ``huggingface_hub.snapshot_download``
    # as a safety net to prevent lmms-eval from downloading all chunks.
    _download_ctx = None
    if _is_videomme and limit is not None and isinstance(limit, int) and limit > 0:
        _download_ctx = _patch_snapshot_download_for_limit(limit)

    with _model_ctx, _subsample_ctx:
        if _download_ctx is not None:
            with _download_ctx:
                results = simple_evaluate(**eval_kwargs)
        else:
            results = simple_evaluate(**eval_kwargs)

    return results


def _get_custom_tasks_dir() -> str | None:
    """Return the path to TICO's custom ``lmms_tasks`` directory.

    This directory contains custom task YAML files (e.g.
    ``videomme_mini``) that are not part of the upstream ``lmms-eval``
    package.  The path is passed to ``simple_evaluate`` via the
    ``include_path`` parameter so that custom tasks can be discovered.

    Returns:
        The absolute path to the ``lmms_tasks`` directory, or ``None``
        if it does not exist.
    """
    import pathlib

    tasks_dir = pathlib.Path(__file__).parent / "lmms_tasks"
    if tasks_dir.is_dir():
        return str(tasks_dir)
    return None


def _build_model_args(
    model: Any,
    max_num_frames: int = 32,
) -> tuple[str, dict[str, Any]]:
    """Build the ``model`` name and ``model_args`` dict for ``simple_evaluate``.

    In ``lmms-eval`` v0.7+, ``simple_evaluate`` takes a model name string
    (from the registry) and a ``model_args`` dict that is forwarded to the
    model constructor.  This helper infers the correct model name from the
    Python model object and builds the args dict.

    Returns:
        A ``(model_name, model_args_dict)`` tuple.
    """
    model_cls_name = type(model).__name__.lower()

    # Map model class names to lmms-eval registry names
    if "qwen3" in model_cls_name and (
        "vl" in model_cls_name or "visual" in model_cls_name
    ):
        lmms_model_name = "qwen3_vl"
    elif "qwen2" in model_cls_name and (
        "vl" in model_cls_name or "visual" in model_cls_name
    ):
        lmms_model_name = "qwen2_5_vl"
    elif "qwen" in model_cls_name and (
        "vl" in model_cls_name or "visual" in model_cls_name
    ):
        lmms_model_name = "qwen2_5_vl"
    elif "llava" in model_cls_name:
        lmms_model_name = "llava_hf"
    elif "internvl" in model_cls_name:
        lmms_model_name = "internvl_hf"
    else:
        # Fallback: use the generic huggingface model
        lmms_model_name = "huggingface"

    # Get the Hugging Face model id from the model config if available
    pretrained = None
    if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
        pretrained = model.config._name_or_path

    model_args_dict: dict[str, Any] = {}

    if pretrained is not None:
        model_args_dict["pretrained"] = pretrained

    # Pass max_num_frames to the lmms-eval model wrapper.
    # The Qwen3_VL wrapper accepts this in __init__ and uses it in
    # _subsample_video_inputs to control how many frames are sampled
    # from each video.
    if max_num_frames is not None:
        model_args_dict["max_num_frames"] = max_num_frames

    return lmms_model_name, model_args_dict


def print_lmms_eval_results(results: dict[str, Any]) -> None:
    """Print ``lmms-eval`` results in a formatted table.

    This uses ``lmms_eval``'s built-in ``make_table`` utility when
    available, and falls back to a simple custom printer.

    Args:
        results: The results dictionary returned by
            :func:`evaluate_vlm_on_tasks`.
    """
    try:
        from lmms_eval.utils import make_table

        print(make_table(results))
    except ImportError:
        _print_results_fallback(results)


def _print_results_fallback(results: dict[str, Any]) -> None:
    """Print results when ``make_table`` is not available.

    Args:
        results: The results dictionary returned by
            :func:`evaluate_vlm_on_tasks`.
    """
    task_results = results.get("results", {})
    for task_name, metrics in task_results.items():
        print(f"\n=== {task_name} ===")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name:<40} {value:.4f}")
            else:
                print(f"  {metric_name:<40} {value}")
