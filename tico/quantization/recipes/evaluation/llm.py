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

from typing import Any

from tico.quantization.evaluation.optional_deps import require_attr

_DATASETS_INSTALL_HINT = "pip install datasets"
_LM_EVAL_INSTALL_HINT = "pip install lm-eval"


def evaluate_perplexity(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    cache_dir: str | None,
    max_seq_len: int,
    dataset_name: str = "Salesforce/wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
) -> float:
    """Evaluate text perplexity on a Hugging Face dataset.

    The ``datasets`` package is imported lazily so that importing recipe modules
    does not require optional evaluation dependencies unless this function is
    actually used.
    """
    load_dataset = require_attr(
        "datasets",
        "load_dataset",
        feature="LLM perplexity evaluation",
        install_hint=_DATASETS_INSTALL_HINT,
    )

    from tico.quantization.wrapq.utils.metrics import perplexity

    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=split,
        cache_dir=cache_dir,
    )
    enc = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    return float(
        perplexity(model, enc, device, max_length=max_seq_len, stride=max_seq_len)
    )


def evaluate_lm_tasks(
    *,
    model: Any,
    tokenizer: Any,
    tasks: str,
    max_seq_len: int,
) -> Any:
    """Evaluate text-only LM tasks through ``lm_eval``.

    ``lm_eval`` is imported lazily and reports a clear error only when this
    evaluation path is requested.
    """
    make_table = require_attr(
        "lm_eval.utils",
        "make_table",
        feature="lm_eval task evaluation",
        install_hint=_LM_EVAL_INSTALL_HINT,
    )

    from tico.quantization.evaluation.script.llm_tasks_eval import evaluate_llm_on_tasks

    results = evaluate_llm_on_tasks(model, tokenizer, tasks, max_length=max_seq_len)
    print(make_table(results))
    return results
