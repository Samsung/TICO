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

    setattr(module, _STUB_MARKER, True)

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

    setattr(lm_eval_module, _STUB_MARKER, True)

    evaluator_module = types.ModuleType("lm_eval.evaluator")
    setattr(evaluator_module, _STUB_MARKER, True)

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
    setattr(utils_module, _STUB_MARKER, True)

    def make_table(results):
        """Return a stable string representation for patched evaluation results."""
        return str(results)

    setattr(utils_module, "make_table", make_table)
    sys.modules["lm_eval.utils"] = utils_module
    setattr(lm_eval_module, "utils", utils_module)

    models_module = types.ModuleType("lm_eval.models")
    setattr(models_module, _STUB_MARKER, True)

    huggingface_module = types.ModuleType("lm_eval.models.huggingface")
    setattr(huggingface_module, _STUB_MARKER, True)

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
