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

from collections.abc import Mapping, Sequence
from typing import Any


SELECTED_TASKS_KEY = "selected_tasks"


def _normalize_evaluation_targets(
    values: Sequence[Any],
    *,
    source: str,
    allow_empty: bool,
) -> tuple[str, ...]:
    """Normalize and validate an ordered sequence of evaluation target names."""
    targets: list[str] = []
    seen: set[str] = set()

    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(
                f"{source}[{index}] must be a string. got {type(value).__name__}"
            )

        target = value.strip()
        if not target:
            raise ValueError(f"{source} must not contain empty target names.")
        if target in seen:
            raise ValueError(f"{source} contains duplicate target {target!r}.")

        seen.add(target)
        targets.append(target)

    if not targets and not allow_empty:
        raise ValueError(f"{source} must contain at least one evaluation target.")

    return tuple(targets)


def parse_evaluation_targets(raw: str) -> list[str]:
    """Parse a comma-separated CLI value into canonical evaluation targets."""
    if not isinstance(raw, str):
        raise TypeError(f"--tasks must be a string. got {type(raw).__name__}")

    return list(
        _normalize_evaluation_targets(
            raw.split(","),
            source="--tasks",
            allow_empty=False,
        )
    )


def get_selected_evaluation_targets(
    eval_cfg: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """Return the exclusive target allow-list, or None when it is not configured."""
    if SELECTED_TASKS_KEY not in eval_cfg:
        return None

    raw_targets = eval_cfg[SELECTED_TASKS_KEY]
    if raw_targets is None:
        return None
    if isinstance(raw_targets, (str, bytes)) or not isinstance(raw_targets, Sequence):
        raise TypeError(
            "evaluation.selected_tasks must be a sequence of target names or null."
        )

    return _normalize_evaluation_targets(
        raw_targets,
        source="evaluation.selected_tasks",
        allow_empty=True,
    )


def validate_adapter_evaluation_config(
    adapter: Any,
    cfg: Mapping[str, Any],
) -> None:
    """Validate evaluation selection without breaking legacy duck-typed adapters."""
    validator = getattr(adapter, "validate_evaluation_config", None)
    if validator is not None:
        if not callable(validator):
            raise TypeError(
                "Model adapter validate_evaluation_config must be callable."
            )
        validator(cfg)
        return

    eval_cfg = cfg.get("evaluation")
    if eval_cfg is None:
        return
    if not isinstance(eval_cfg, Mapping):
        raise TypeError("evaluation must be a mapping.")
    if not eval_cfg.get("enabled", False):
        return

    selected_targets = get_selected_evaluation_targets(eval_cfg)
    if selected_targets is None:
        return

    family = getattr(adapter, "family", type(adapter).__name__)
    raise TypeError(
        f"Model adapter {family!r} must implement validate_evaluation_config() "
        "to use evaluation.selected_tasks."
    )


def should_run_evaluation(
    eval_cfg: Mapping[str, Any],
    target_name: str,
    *,
    default_enabled: bool,
) -> bool:
    """Return whether a top-level evaluation target should run."""
    selected_targets = get_selected_evaluation_targets(eval_cfg)
    if selected_targets is not None:
        return target_name in selected_targets
    return default_enabled


def get_mapping_evaluation_config(
    eval_cfg: Mapping[str, Any],
    config_key: str,
) -> Mapping[str, Any]:
    """Return a nested evaluation mapping, using an empty mapping when omitted."""
    value = eval_cfg.get(config_key)
    if value is None or value is False:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"evaluation.{config_key} must be a mapping.")
    return value


def should_run_mapping_evaluation(
    eval_cfg: Mapping[str, Any],
    target_name: str,
    *,
    config_key: str | None = None,
) -> bool:
    """Select a mapping-backed target while preserving legacy enabled behavior."""
    selected_targets = get_selected_evaluation_targets(eval_cfg)
    if selected_targets is not None:
        return target_name in selected_targets

    key = config_key or target_name
    target_cfg = get_mapping_evaluation_config(eval_cfg, key)
    return bool(target_cfg.get("enabled", False))
