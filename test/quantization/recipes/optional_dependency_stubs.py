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

import sys
import types


def install_optional_dependency_stubs() -> None:
    """Install lightweight stubs for optional recipe dependencies."""
    _install_datasets_stub()
    _install_lm_eval_stub()


def _install_datasets_stub() -> None:
    """Install a minimal datasets module when the real package is unavailable."""
    if "datasets" in sys.modules and all(
        hasattr(sys.modules["datasets"], name)
        for name in ("Dataset", "IterableDataset", "load_dataset")
    ):
        return

    module = sys.modules.get("datasets")
    if module is None:
        module = types.ModuleType("datasets")
        sys.modules["datasets"] = module

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
    if "lm_eval" in sys.modules and hasattr(sys.modules["lm_eval"], "evaluator"):
        return

    lm_eval_module = sys.modules.get("lm_eval")
    if lm_eval_module is None:
        lm_eval_module = types.ModuleType("lm_eval")
        sys.modules["lm_eval"] = lm_eval_module

    evaluator_module = types.ModuleType("lm_eval.evaluator")

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

    def make_table(results):
        """Return a stable string representation for patched evaluation results."""
        return str(results)

    setattr(utils_module, "make_table", make_table)
    sys.modules["lm_eval.utils"] = utils_module
    setattr(lm_eval_module, "utils", utils_module)

    models_module = types.ModuleType("lm_eval.models")
    huggingface_module = types.ModuleType("lm_eval.models.huggingface")

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
