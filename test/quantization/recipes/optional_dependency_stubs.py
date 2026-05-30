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

"""
Test-only stubs for optional third-party dependencies.

The recipe modules import evaluation/data helpers that depend on packages such as
`datasets` and `lm_eval`. Unit tests in this directory patch the actual
evaluation/data calls, so requiring those optional packages at import time would
make the tests heavier than necessary.
"""

import importlib
import sys
import types

_STUB_MARKER = "__tico_optional_dependency_stub__"


def install_optional_dependency_stubs() -> None:
    """Install lightweight stubs for optional recipe dependencies."""
    _install_datasets_stub()
    _install_lm_eval_stub()
    _install_decord_stub()
    _install_lmms_eval_stub()


def _has_attrs(module: types.ModuleType, names: tuple[str, ...]) -> bool:
    """Return True if a module has all required attributes."""
    return all(hasattr(module, name) for name in names)


def _is_our_stub(module: types.ModuleType | None) -> bool:
    """Return True if a module was installed by this stub helper."""
    return bool(module is not None and getattr(module, _STUB_MARKER, False))


def _try_import_optional_module(module_name: str) -> types.ModuleType | None:
    """
    Import an optional module if it is available.

    Missing optional modules are tolerated. Import failures caused by missing
    transitive dependencies are re-raised so broken real installations are not
    silently hidden by test stubs.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        top_level_name = module_name.partition(".")[0]
        if exc.name in {module_name, top_level_name}:
            return None
        raise


def _mark_as_stub(module: types.ModuleType) -> None:
    """Mark a module as installed by this helper."""
    setattr(module, _STUB_MARKER, True)


def _mark_as_package(module: types.ModuleType) -> None:
    """Make a ModuleType behave like a package for nested imports."""
    if not hasattr(module, "__path__"):
        module.__path__ = []  # type: ignore[attr-defined]


def _install_datasets_stub() -> None:
    """Install a minimal datasets module when the real package is unavailable."""
    required_attrs = ("Dataset", "IterableDataset", "load_dataset")
    existing_module = sys.modules.get("datasets")

    if (
        existing_module is not None
        and not _is_our_stub(existing_module)
        and _has_attrs(existing_module, required_attrs)
    ):
        return

    real_module = _try_import_optional_module("datasets")
    if real_module is not None and _has_attrs(real_module, required_attrs):
        return

    module = sys.modules.get("datasets")
    if module is None or not _is_our_stub(module):
        module = types.ModuleType("datasets")
        sys.modules["datasets"] = module

    _mark_as_stub(module)

    class Dataset:
        """Minimal datasets.Dataset stub for import-time compatibility."""

        @classmethod
        def from_dict(cls, *args, **kwargs):
            """Create a minimal Dataset stub instance."""
            return cls()

    class IterableDataset:
        """Minimal datasets.IterableDataset stub for import-time compatibility."""

        @classmethod
        def from_generator(cls, *args, **kwargs):
            """Create a minimal IterableDataset stub instance."""
            return cls()

    def load_dataset(*args, **kwargs):
        """Fail clearly if a test accidentally reaches real dataset loading."""
        raise RuntimeError(
            "The datasets package is stubbed in recipe tests. "
            "Patch the recipe data/evaluation helper instead of loading data."
        )

    setattr(module, "Dataset", Dataset)
    setattr(module, "IterableDataset", IterableDataset)
    setattr(module, "load_dataset", load_dataset)


def _install_lm_eval_stub() -> None:
    """Install minimal lm_eval modules when the real package is unavailable."""
    existing_module = sys.modules.get("lm_eval")

    if (
        existing_module is not None
        and not _is_our_stub(existing_module)
        and hasattr(existing_module, "evaluator")
    ):
        return

    real_module = _try_import_optional_module("lm_eval")
    if real_module is not None and not _is_our_stub(real_module):
        real_evaluator_module = _try_import_optional_module("lm_eval.evaluator")
        if real_evaluator_module is not None:
            setattr(real_module, "evaluator", real_evaluator_module)
            return

    lm_eval_module = sys.modules.get("lm_eval")
    if lm_eval_module is None or not _is_our_stub(lm_eval_module):
        lm_eval_module = types.ModuleType("lm_eval")
        sys.modules["lm_eval"] = lm_eval_module

    _mark_as_stub(lm_eval_module)
    _mark_as_package(lm_eval_module)

    evaluator_module = types.ModuleType("lm_eval.evaluator")
    _mark_as_stub(evaluator_module)

    def simple_evaluate(*args, **kwargs):
        """Fail clearly if a test accidentally runs real lm-eval."""
        raise RuntimeError(
            "lm_eval.evaluator.simple_evaluate is stubbed in recipe tests. "
            "Patch evaluate_lm_tasks/evaluate_llm_on_tasks instead."
        )

    setattr(evaluator_module, "simple_evaluate", simple_evaluate)
    sys.modules["lm_eval.evaluator"] = evaluator_module
    setattr(lm_eval_module, "evaluator", evaluator_module)

    utils_module = types.ModuleType("lm_eval.utils")
    _mark_as_stub(utils_module)

    def make_table(results):
        """Return a stable string representation for patched evaluation results."""
        return str(results)

    setattr(utils_module, "make_table", make_table)
    sys.modules["lm_eval.utils"] = utils_module
    setattr(lm_eval_module, "utils", utils_module)

    models_module = types.ModuleType("lm_eval.models")
    _mark_as_stub(models_module)
    _mark_as_package(models_module)

    huggingface_module = types.ModuleType("lm_eval.models.huggingface")
    _mark_as_stub(huggingface_module)

    class HFLM:
        """Minimal HFLM stub for import-time compatibility."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    setattr(huggingface_module, "HFLM", HFLM)
    setattr(models_module, "huggingface", huggingface_module)
    sys.modules["lm_eval.models"] = models_module
    sys.modules["lm_eval.models.huggingface"] = huggingface_module
    setattr(lm_eval_module, "models", models_module)


def _install_decord_stub() -> None:
    """Install a minimal decord module when the real package is unavailable."""
    existing_module = sys.modules.get("decord")

    if (
        existing_module is not None
        and not _is_our_stub(existing_module)
        and hasattr(existing_module, "VideoReader")
    ):
        return

    real_module = _try_import_optional_module("decord")
    if real_module is not None and not _is_our_stub(real_module):
        return

    decord_module = sys.modules.get("decord")
    if decord_module is None or not _is_our_stub(decord_module):
        decord_module = types.ModuleType("decord")
        sys.modules["decord"] = decord_module

    _mark_as_stub(decord_module)
    _mark_as_package(decord_module)

    class VideoReader:
        """Minimal VideoReader stub for import-time compatibility."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "decord.VideoReader is stubbed in recipe tests. "
                "Patch video decoding or install decord for integration tests."
            )

    def cpu(index: int = 0):
        """Return a minimal CPU context placeholder."""
        return ("cpu", index)

    def gpu(index: int = 0):
        """Return a minimal GPU context placeholder."""
        return ("gpu", index)

    bridge_module = types.ModuleType("decord.bridge")
    _mark_as_stub(bridge_module)

    def set_bridge(name: str) -> None:
        """No-op bridge setter for import-time compatibility."""
        return None

    setattr(bridge_module, "set_bridge", set_bridge)

    setattr(decord_module, "VideoReader", VideoReader)
    setattr(decord_module, "cpu", cpu)
    setattr(decord_module, "gpu", gpu)
    setattr(decord_module, "bridge", bridge_module)
    setattr(decord_module, "__version__", "0.0.0-stub")

    sys.modules["decord.bridge"] = bridge_module


def _install_lmms_eval_stub() -> None:
    """Install minimal lmms_eval modules when the real package is unavailable.

    The Video-MME recipe tests patch the actual evaluation calls, but imported
    modules still reference lmms_eval symbols at import time.
    """
    existing_module = sys.modules.get("lmms_eval")

    if (
        existing_module is not None
        and not _is_our_stub(existing_module)
        and hasattr(existing_module, "evaluator")
        and hasattr(existing_module, "tasks")
    ):
        return

    real_module = _try_import_optional_module("lmms_eval")
    if real_module is not None and not _is_our_stub(real_module):
        real_evaluator_module = _try_import_optional_module("lmms_eval.evaluator")
        real_tasks_module = _try_import_optional_module("lmms_eval.tasks")
        if real_evaluator_module is not None and real_tasks_module is not None:
            setattr(real_module, "evaluator", real_evaluator_module)
            setattr(real_module, "tasks", real_tasks_module)
            return

    lmms_eval_module = sys.modules.get("lmms_eval")
    if lmms_eval_module is None or not _is_our_stub(lmms_eval_module):
        lmms_eval_module = types.ModuleType("lmms_eval")
        sys.modules["lmms_eval"] = lmms_eval_module

    _mark_as_stub(lmms_eval_module)
    _mark_as_package(lmms_eval_module)

    # lmms_eval.evaluator
    evaluator_module = types.ModuleType("lmms_eval.evaluator")
    _mark_as_stub(evaluator_module)

    def simple_evaluate(*args, **kwargs):
        """Fail clearly if a test accidentally runs real lmms-eval."""
        raise RuntimeError(
            "lmms_eval.evaluator.simple_evaluate is stubbed in recipe tests. "
            "Patch evaluate_vlm_on_tasks instead of running real lmms-eval."
        )

    setattr(evaluator_module, "simple_evaluate", simple_evaluate)
    sys.modules["lmms_eval.evaluator"] = evaluator_module
    setattr(lmms_eval_module, "evaluator", evaluator_module)

    # lmms_eval.utils
    utils_module = types.ModuleType("lmms_eval.utils")
    _mark_as_stub(utils_module)

    def make_table(results):
        """Return a stable string representation for patched evaluation results."""
        return str(results)

    def load_yaml_config(*args, **kwargs):
        """Return an empty YAML config for import-time compatibility."""
        return {}

    setattr(utils_module, "make_table", make_table)
    setattr(utils_module, "load_yaml_config", load_yaml_config)
    sys.modules["lmms_eval.utils"] = utils_module
    setattr(lmms_eval_module, "utils", utils_module)

    # lmms_eval.tasks
    tasks_module = types.ModuleType("lmms_eval.tasks")
    _mark_as_stub(tasks_module)
    _mark_as_package(tasks_module)

    class TaskManager:
        """Minimal TaskManager stub for import-time compatibility."""

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    setattr(tasks_module, "TaskManager", TaskManager)
    sys.modules["lmms_eval.tasks"] = tasks_module
    setattr(lmms_eval_module, "tasks", tasks_module)

    # lmms_eval.tasks.videomme
    videomme_module = types.ModuleType("lmms_eval.tasks.videomme")
    _mark_as_stub(videomme_module)
    _mark_as_package(videomme_module)

    sys.modules["lmms_eval.tasks.videomme"] = videomme_module
    setattr(tasks_module, "videomme", videomme_module)

    # lmms_eval.tasks.videomme.utils
    videomme_utils_module = types.ModuleType("lmms_eval.tasks.videomme.utils")
    _mark_as_stub(videomme_utils_module)

    setattr(videomme_utils_module, "CATEGORIES", [])
    setattr(videomme_utils_module, "SUB_CATEGORIES", [])
    setattr(videomme_utils_module, "TASK_CATEGORIES", [])
    setattr(videomme_utils_module, "VIDEO_TYPE", [])

    def videomme_aggregate_results(results):
        """Minimal aggregate stub."""
        if not results:
            return 0.0
        return sum(float(x) for x in results) / len(results)

    def videomme_process_results(doc, results):
        """Minimal process-results stub."""
        return {"videomme_perception_score": 0.0}

    def videomme_doc_to_text(doc, lmms_eval_specific_kwargs=None):
        """Minimal prompt builder used by videomme_mini utils tests."""
        question = doc.get("question", "")
        options = doc.get("options", [])
        option_text = "\n".join(str(option) for option in options)
        if option_text:
            return f"{question}\n{option_text}"
        return question

    setattr(
        videomme_utils_module,
        "videomme_aggregate_results",
        videomme_aggregate_results,
    )
    setattr(
        videomme_utils_module,
        "videomme_process_results",
        videomme_process_results,
    )
    setattr(videomme_utils_module, "videomme_doc_to_text", videomme_doc_to_text)

    sys.modules["lmms_eval.tasks.videomme.utils"] = videomme_utils_module
    setattr(videomme_module, "utils", videomme_utils_module)

    # Optional path used by _patch_subsample_video_inputs_for_debug().
    # Keeping this available makes that helper return a context manager instead
    # of depending on a real lmms_eval installation.
    models_module = types.ModuleType("lmms_eval.models")
    _mark_as_stub(models_module)
    _mark_as_package(models_module)

    simple_models_module = types.ModuleType("lmms_eval.models.simple")
    _mark_as_stub(simple_models_module)
    _mark_as_package(simple_models_module)

    qwen3_vl_module = types.ModuleType("lmms_eval.models.simple.qwen3_vl")
    _mark_as_stub(qwen3_vl_module)

    class Qwen3_VL:
        """Minimal Qwen3_VL stub for debug monkey-patch tests."""

        max_num_frames = 0

        def _subsample_video_inputs(self, video_inputs, video_metadatas=None):
            return video_inputs

    setattr(qwen3_vl_module, "Qwen3_VL", Qwen3_VL)

    sys.modules["lmms_eval.models"] = models_module
    sys.modules["lmms_eval.models.simple"] = simple_models_module
    sys.modules["lmms_eval.models.simple.qwen3_vl"] = qwen3_vl_module

    setattr(lmms_eval_module, "models", models_module)
    setattr(models_module, "simple", simple_models_module)
    setattr(simple_models_module, "qwen3_vl", qwen3_vl_module)
