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

"""Resolve semantic dataset roles and prevent benchmark leakage."""

from __future__ import annotations

import warnings

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


CALIBRATION_ROLE = "calibration"
EVALUATION_ROLE = "evaluation"
FEW_SHOT_ROLE = "few_shot"
EVALUATION_CONTEXT_ROLE = "evaluation_context"

_DATA_USE_ROLES = frozenset(
    {
        CALIBRATION_ROLE,
        EVALUATION_ROLE,
        FEW_SHOT_ROLE,
        EVALUATION_CONTEXT_ROLE,
    }
)
_WILDCARD_CONFIGS = frozenset({"", "*", "all"})


class DatasetUsageError(ValueError):
    """Report an unsafe or invalid dataset use before any data is loaded."""


@dataclass(frozen=True)
class DatasetPolicy:
    """Describe the supported roles and split policy for one logical dataset."""

    key: str
    canonical_id: str
    aliases: tuple[str, ...]
    allowed_roles: frozenset[str]
    default_splits: Mapping[str, str]
    calibration_safe_splits: frozenset[str] | None = None
    default_config: str | None = None


@dataclass(frozen=True)
class DatasetUsage:
    """Represent one fully resolved dataset use in a quantization recipe."""

    requested_id: str
    canonical_id: str
    config: str | None
    split: str
    role: str
    consumer: str
    n_samples: int | None = None
    targets_included: bool = False
    policy_key: str | None = None

    @property
    def identity(self) -> tuple[str, str | None, str]:
        """Return the normalized identity used for overlap comparisons."""
        return (
            _normalize_name(self.canonical_id),
            _normalize_optional(self.config),
            _normalize_name(self.split),
        )

    def to_config(self) -> dict[str, Any]:
        """Return a YAML-serializable provenance record."""
        return {
            "requested_id": self.requested_id,
            "canonical_id": self.canonical_id,
            "config": self.config,
            "split": self.split,
            "role": self.role,
            "consumer": self.consumer,
            "n_samples": self.n_samples,
            "targets_included": self.targets_included,
        }

    def describe(self) -> str:
        """Return a compact human-readable source description."""
        config = self.config if self.config is not None else "<default>"
        samples = (
            "all" if self.n_samples is None or self.n_samples < 0 else self.n_samples
        )
        return (
            f"{self.canonical_id}, config={config}, split={self.split}, "
            f"role={self.role}, consumer={self.consumer}, samples={samples}, "
            f"targets_included={self.targets_included}"
        )


def _policy(
    *,
    key: str,
    canonical_id: str,
    aliases: Sequence[str],
    allowed_roles: Sequence[str],
    default_splits: Mapping[str, str],
    calibration_safe_splits: Sequence[str] | None = None,
    default_config: str | None = None,
) -> DatasetPolicy:
    """Build an immutable dataset policy with normalized collection fields."""
    return DatasetPolicy(
        key=key,
        canonical_id=canonical_id,
        aliases=tuple(aliases),
        allowed_roles=frozenset(allowed_roles),
        default_splits=dict(default_splits),
        calibration_safe_splits=(
            None
            if calibration_safe_splits is None
            else frozenset(calibration_safe_splits)
        ),
        default_config=default_config,
    )


DATASET_POLICIES: dict[str, DatasetPolicy] = {
    "vqav2": _policy(
        key="vqav2",
        canonical_id="HuggingFaceM4/VQAv2",
        aliases=("vqav2", "HuggingFaceM4/VQAv2", "lmms-lab/VQAv2"),
        allowed_roles=(CALIBRATION_ROLE, EVALUATION_ROLE),
        default_splits={CALIBRATION_ROLE: "train", EVALUATION_ROLE: "validation"},
        calibration_safe_splits=("train",),
    ),
    "textvqa": _policy(
        key="textvqa",
        canonical_id="lmms-lab-encoder/textvqa",
        aliases=(
            "textvqa",
            "HuggingFaceM4/TextVQA",
            "lmms-lab/textvqa",
            "lmms-lab-encoder/textvqa",
        ),
        allowed_roles=(CALIBRATION_ROLE, EVALUATION_ROLE),
        default_splits={CALIBRATION_ROLE: "train", EVALUATION_ROLE: "validation"},
        calibration_safe_splits=("train",),
    ),
    "coco": _policy(
        key="coco",
        canonical_id="lmms-lab-encoder/COCO-Caption2017",
        aliases=(
            "coco",
            "lmms-lab/COCO-Caption2017",
            "lmms-lab-encoder/COCO-Caption2017",
        ),
        allowed_roles=(EVALUATION_ROLE,),
        default_splits={EVALUATION_ROLE: "val"},
        calibration_safe_splits=(),
    ),
    "llava_bench": _policy(
        key="llava_bench",
        canonical_id="lmms-lab/llava-bench-in-the-wild",
        aliases=(
            "llava_bench",
            "llava-bench",
            "lmms-lab/llava-bench-in-the-wild",
        ),
        allowed_roles=(EVALUATION_ROLE,),
        default_splits={EVALUATION_ROLE: "train"},
        calibration_safe_splits=(),
    ),
    "wikitext2": _policy(
        key="wikitext2",
        canonical_id="Salesforce/wikitext",
        aliases=("wikitext2", "wikitext", "Salesforce/wikitext"),
        allowed_roles=(CALIBRATION_ROLE, EVALUATION_ROLE),
        default_splits={CALIBRATION_ROLE: "train", EVALUATION_ROLE: "test"},
        calibration_safe_splits=("train",),
        default_config="wikitext-2-raw-v1",
    ),
    "alpaca": _policy(
        key="alpaca",
        canonical_id="tatsu-lab/alpaca",
        aliases=("alpaca", "tatsu-lab/alpaca"),
        allowed_roles=(CALIBRATION_ROLE,),
        default_splits={CALIBRATION_ROLE: "train"},
        calibration_safe_splits=("train",),
    ),
    "mmlu": _policy(
        key="mmlu",
        canonical_id="cais/mmlu",
        aliases=("mmlu", "cais/mmlu"),
        allowed_roles=(CALIBRATION_ROLE, EVALUATION_ROLE, FEW_SHOT_ROLE),
        default_splits={
            CALIBRATION_ROLE: "auxiliary_train",
            EVALUATION_ROLE: "test",
            FEW_SHOT_ROLE: "dev",
        },
        calibration_safe_splits=("auxiliary_train",),
        default_config="all",
    ),
    "videomme": _policy(
        key="videomme",
        canonical_id="lmms-eval/Video-MME",
        aliases=(
            "videomme",
            "videomme_text",
            "video-mme",
            "lmms-eval/Video-MME",
            "lmms-lab/Video-MME",
        ),
        allowed_roles=(EVALUATION_ROLE,),
        default_splits={EVALUATION_ROLE: "test"},
        calibration_safe_splits=(),
    ),
    "mmmu": _policy(
        key="mmmu",
        canonical_id="MMMU/MMMU",
        aliases=("mmmu", "MMMU/MMMU"),
        allowed_roles=(EVALUATION_ROLE, FEW_SHOT_ROLE, EVALUATION_CONTEXT_ROLE),
        default_splits={
            EVALUATION_ROLE: "validation",
            FEW_SHOT_ROLE: "dev",
            EVALUATION_CONTEXT_ROLE: "test",
        },
        calibration_safe_splits=(),
    ),
    "mmmu_pro_vision": _policy(
        key="mmmu_pro_vision",
        canonical_id="MMMU/MMMU_Pro",
        aliases=("mmmu_pro_vision", "mmmu-pro-vision", "MMMU/MMMU_Pro"),
        allowed_roles=(EVALUATION_ROLE,),
        default_splits={EVALUATION_ROLE: "test"},
        calibration_safe_splits=(),
        default_config="vision",
    ),
    "hellaswag": _policy(
        key="hellaswag",
        canonical_id="Rowan/hellaswag",
        aliases=("hellaswag", "Rowan/hellaswag"),
        allowed_roles=(EVALUATION_ROLE,),
        default_splits={EVALUATION_ROLE: "validation"},
        calibration_safe_splits=(),
    ),
}


def _normalize_name(value: str) -> str:
    """Normalize a dataset identifier, role, config, or split for comparison."""
    return str(value).strip().casefold()


def _normalize_optional(value: str | None) -> str | None:
    """Normalize an optional string while preserving the absence of a value."""
    if value is None:
        return None
    return _normalize_name(value)


def _build_alias_index() -> dict[str, str]:
    """Build the normalized alias-to-policy lookup table."""
    aliases: dict[str, str] = {}
    for key, policy in DATASET_POLICIES.items():
        for alias in (key, policy.canonical_id, *policy.aliases):
            normalized = _normalize_name(alias)
            previous = aliases.get(normalized)
            if previous is not None and previous != key:
                raise RuntimeError(
                    f"Dataset alias {alias!r} is assigned to both "
                    f"{previous!r} and {key!r}."
                )
            aliases[normalized] = key
    return aliases


_ALIAS_TO_POLICY = _build_alias_index()


def get_dataset_policy(dataset: str) -> DatasetPolicy | None:
    """Return the known semantic policy for a dataset identifier or alias."""
    policy_key = _ALIAS_TO_POLICY.get(_normalize_name(dataset))
    return None if policy_key is None else DATASET_POLICIES[policy_key]


def resolve_dataset_usage(
    *,
    dataset: str,
    role: str,
    split: str | None = None,
    config: str | None = None,
    consumer: str,
    n_samples: int | None = None,
    targets_included: bool = False,
) -> DatasetUsage:
    """Resolve one requested dataset use to a canonical identity and split."""
    normalized_role = _normalize_name(role)
    if normalized_role not in _DATA_USE_ROLES:
        raise ValueError(
            f"Unsupported dataset role {role!r}. Supported roles: "
            f"{sorted(_DATA_USE_ROLES)}"
        )
    if not str(dataset).strip():
        raise ValueError("Dataset identifier must not be empty.")
    if not str(consumer).strip():
        raise ValueError("Dataset consumer must not be empty.")

    policy = get_dataset_policy(dataset)
    if split is None:
        if policy is None:
            raise ValueError(
                f"Dataset {dataset!r} has no registered default split for role "
                f"{normalized_role!r}; specify the split explicitly."
            )
        resolved_split = policy.default_splits.get(normalized_role)
        if resolved_split is None:
            # Resolve the dataset's benchmark split so the central validator can
            # report the semantic role violation and honor an explicit
            # transductive override.
            resolved_split = next(iter(policy.default_splits.values()), None)
        if resolved_split is None:
            allowed = ", ".join(sorted(policy.allowed_roles))
            raise DatasetUsageError(
                f"Dataset {dataset!r} does not define a split for role "
                f"{normalized_role!r}. Allowed roles: {allowed}."
            )
    else:
        resolved_split = str(split).strip()
        if not resolved_split:
            raise ValueError("Dataset split must not be empty.")

    resolved_config = config
    if resolved_config is None and policy is not None:
        resolved_config = policy.default_config
    if resolved_config is not None:
        resolved_config = str(resolved_config).strip() or None

    return DatasetUsage(
        requested_id=str(dataset),
        canonical_id=(policy.canonical_id if policy is not None else str(dataset)),
        config=resolved_config,
        split=resolved_split,
        role=normalized_role,
        consumer=str(consumer),
        n_samples=n_samples,
        targets_included=bool(targets_included),
        policy_key=(policy.key if policy is not None else None),
    )


def _usage_policy(usage: DatasetUsage) -> DatasetPolicy | None:
    """Return the policy associated with a resolved usage descriptor."""
    if usage.policy_key is not None:
        return DATASET_POLICIES[usage.policy_key]
    return get_dataset_policy(usage.canonical_id)


def dataset_usage_risks(usage: DatasetUsage) -> list[str]:
    """Return role and split safety violations for one dataset use."""
    risks: list[str] = []
    policy = _usage_policy(usage)

    if policy is not None and usage.role not in policy.allowed_roles:
        risks.append(
            f"{usage.canonical_id} is not allowed for role {usage.role!r}; "
            f"allowed roles are {sorted(policy.allowed_roles)}"
        )

    if usage.role == CALIBRATION_ROLE and policy is not None:
        safe_splits = policy.calibration_safe_splits
        if safe_splits is not None and usage.split not in safe_splits:
            risks.append(
                f"split {usage.split!r} is not calibration-safe for "
                f"{usage.canonical_id}; safe splits are {sorted(safe_splits)}"
            )

    return risks


def _target_inclusion_risks(usage: DatasetUsage) -> list[str]:
    """Return target-leakage violations that no overlap override may waive."""
    if usage.role == CALIBRATION_ROLE and usage.targets_included:
        return [
            "gold targets are included in calibration inputs; "
            "calibration.allow_benchmark_overlap does not permit target leakage"
        ]
    return []


def _config_overlaps(left: str | None, right: str | None) -> bool:
    """Return whether two configs refer to overlapping subsets."""
    left_norm = _normalize_optional(left)
    right_norm = _normalize_optional(right)
    if left_norm == right_norm:
        return True
    return (left_norm or "") in _WILDCARD_CONFIGS or (
        right_norm or ""
    ) in _WILDCARD_CONFIGS


def dataset_usages_overlap(left: DatasetUsage, right: DatasetUsage) -> bool:
    """Return whether two descriptors resolve to the same data partition."""
    return (
        _normalize_name(left.canonical_id) == _normalize_name(right.canonical_id)
        and _normalize_name(left.split) == _normalize_name(right.split)
        and _config_overlaps(left.config, right.config)
    )


def _format_usage_error(
    calibration_usages: Sequence[DatasetUsage],
    evaluation_usages: Sequence[DatasetUsage],
    violations: Sequence[str],
) -> str:
    """Format a detailed early-validation error for unsafe dataset usage."""
    lines = ["Unsafe dataset usage detected."]
    lines.extend(f"- {violation}" for violation in violations)

    if calibration_usages:
        lines.append("")
        lines.append("Resolved calibration sources:")
        lines.extend(f"  - {usage.describe()}" for usage in calibration_usages)
    if evaluation_usages:
        lines.append("")
        lines.append("Resolved evaluation sources:")
        lines.extend(f"  - {usage.describe()}" for usage in evaluation_usages)

    lines.extend(
        [
            "",
            (
                "Choose a source allowed for the requested role. Unsafe "
                "calibration splits and calibration/evaluation overlap may be "
                "opted into with calibration.allow_benchmark_overlap=true for an "
                "explicitly transductive experiment. The override never permits "
                "gold-target leakage or an invalid evaluation role."
            ),
        ]
    )
    return "\n".join(lines)


def validate_single_dataset_usage(
    usage: DatasetUsage,
    *,
    allow_benchmark_overlap: bool = False,
) -> None:
    """Validate one loader request independently of a complete recipe."""
    target_risks = _target_inclusion_risks(usage)
    if target_risks:
        raise DatasetUsageError(
            _format_usage_error(
                [usage],
                [],
                [f"{usage.consumer}: {risk}" for risk in target_risks],
            )
        )

    risks = dataset_usage_risks(usage)
    if not risks:
        return

    message = _format_usage_error(
        [usage] if usage.role == CALIBRATION_ROLE else [],
        [usage] if usage.role != CALIBRATION_ROLE else [],
        [f"{usage.consumer}: {risk}" for risk in risks],
    )
    if usage.role != CALIBRATION_ROLE or not allow_benchmark_overlap:
        raise DatasetUsageError(message)
    warnings.warn(
        "TRANSDUCTIVE DATASET USAGE ENABLED. " + " ".join(risks),
        RuntimeWarning,
        stacklevel=2,
    )


def _coerce_optional_int(value: Any) -> int | None:
    """Convert an optional sample count to an integer."""
    if value is None:
        return None
    return int(value)


def _parse_compact_dataset_spec(
    spec: str,
    *,
    default_n_samples: int,
) -> tuple[str, dict[str, Any]]:
    """Parse dataset, dataset:count, or dataset:split:count syntax."""
    parts = [part.strip() for part in spec.split(":")]
    if not parts or not parts[0]:
        raise ValueError(f"Invalid calibration dataset spec: {spec!r}.")
    if len(parts) == 1:
        return parts[0], {"n_samples": default_n_samples}
    if len(parts) == 2:
        return parts[0], {"n_samples": int(parts[1])}
    if len(parts) == 3:
        if not parts[1]:
            raise ValueError(f"Calibration split must not be empty in {spec!r}.")
        return parts[0], {"split": parts[1], "n_samples": int(parts[2])}
    raise ValueError(
        f"Invalid calibration dataset spec {spec!r}. Expected dataset, "
        "dataset:count, or dataset:split:count."
    )


def _iter_mixed_calibration_entries(
    raw: Any,
    *,
    default_n_samples: int,
) -> list[tuple[str, dict[str, Any], str]]:
    """Normalize all supported mixed calibration configuration forms."""
    entries: list[tuple[str, dict[str, Any], str]] = []

    if isinstance(raw, str):
        items: Sequence[Any] = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, Mapping):
        for dataset, value in raw.items():
            if isinstance(value, Mapping):
                config = dict(value)
                config.setdefault("n_samples", default_n_samples)
            else:
                config = {"n_samples": int(value)}
            entries.append((str(dataset), config, f"calibration.datasets.{dataset}"))
        return entries
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        items = raw
    else:
        raise TypeError("calibration.datasets must be a string, mapping, or sequence.")

    for index, item in enumerate(items):
        consumer = f"calibration.datasets[{index}]"
        if isinstance(item, str):
            dataset, config = _parse_compact_dataset_spec(
                item,
                default_n_samples=default_n_samples,
            )
        elif isinstance(item, Mapping):
            dataset = item.get("dataset", item.get("name"))
            if not dataset:
                raise ValueError(f"{consumer} must define 'dataset' or 'name'.")
            config = dict(item)
            config.setdefault("n_samples", default_n_samples)
        else:
            raise TypeError(
                f"{consumer} must be a string or mapping, got {type(item).__name__}."
            )
        entries.append((str(dataset), config, consumer))
    return entries


def _collect_calibration_usages(cfg: Mapping[str, Any]) -> list[DatasetUsage]:
    """Resolve all calibration sources selected by a recipe."""
    model_cfg = cfg.get("model", {})
    family = str(model_cfg.get("family", "")).strip().casefold()
    calibration = cfg.get("calibration", {})
    if not isinstance(calibration, Mapping):
        raise TypeError("calibration must be a mapping.")

    default_n_samples = int(calibration.get("n_samples", 128))
    usages: list[DatasetUsage] = []

    if family == "llama":
        usages.append(
            resolve_dataset_usage(
                dataset=str(calibration.get("dataset", "Salesforce/wikitext")),
                role=CALIBRATION_ROLE,
                split=calibration.get("split", "train"),
                config=calibration.get(
                    "dataset_config",
                    "wikitext-2-raw-v1",
                ),
                consumer="calibration.dataset",
                n_samples=default_n_samples,
            )
        )
        return usages

    raw_mixed = calibration.get("datasets")
    if raw_mixed is not None:
        entries = _iter_mixed_calibration_entries(
            raw_mixed,
            default_n_samples=default_n_samples,
        )
    else:
        raw_single = calibration.get("dataset")
        if isinstance(raw_single, str) and "," in raw_single:
            entries = _iter_mixed_calibration_entries(
                raw_single,
                default_n_samples=default_n_samples,
            )
        else:
            dataset = raw_single or (
                "vqav2" if family in {"qwen3_vl", "gemma4"} else None
            )
            if dataset is None:
                return []
            entries = [
                (
                    str(dataset),
                    {
                        "split": calibration.get("split"),
                        "config": calibration.get("dataset_config"),
                        "n_samples": default_n_samples,
                        "include_targets": calibration.get(
                            "include_targets",
                            False,
                        ),
                    },
                    "calibration.dataset",
                )
            ]

    for dataset, entry, consumer in entries:
        usages.append(
            resolve_dataset_usage(
                dataset=dataset,
                role=CALIBRATION_ROLE,
                split=entry.get("split"),
                config=entry.get("config", entry.get("dataset_config")),
                consumer=consumer,
                n_samples=_coerce_optional_int(entry.get("n_samples")),
                targets_included=bool(entry.get("include_targets", False)),
            )
        )
    return usages


def _selected_evaluation_targets(
    evaluation: Mapping[str, Any],
) -> tuple[str, ...] | None:
    """Resolve evaluation.selected_tasks with the existing exclusive semantics."""
    if "selected_tasks" not in evaluation or evaluation.get("selected_tasks") is None:
        return None
    raw = evaluation.get("selected_tasks")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(
            "evaluation.selected_tasks must be a sequence of target names or null."
        )
    targets: list[str] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, str):
            raise TypeError(
                f"evaluation.selected_tasks[{index}] must be a string, got "
                f"{type(value).__name__}."
            )
        target = value.strip()
        if not target:
            raise ValueError("evaluation.selected_tasks must not contain empty names.")
        if target in seen:
            raise ValueError(
                f"evaluation.selected_tasks contains duplicate target {target!r}."
            )
        seen.add(target)
        targets.append(target)
    return tuple(targets)


def _target_enabled(
    evaluation: Mapping[str, Any],
    selected: tuple[str, ...] | None,
    target: str,
    *,
    default_enabled: bool,
) -> bool:
    """Apply selected_tasks as an exclusive allow-list when it is configured."""
    if selected is not None:
        return target in selected
    return default_enabled


def _mapping_target_config(
    evaluation: Mapping[str, Any],
    selected: tuple[str, ...] | None,
    target: str,
) -> Mapping[str, Any] | None:
    """Return an active mapping-backed evaluation target configuration."""
    raw = evaluation.get(target)
    enabled = _target_enabled(
        evaluation,
        selected,
        target,
        default_enabled=(isinstance(raw, Mapping) and bool(raw.get("enabled", False))),
    )
    if not enabled:
        return None
    if raw is None or raw is False:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"evaluation.{target} must be a mapping.")
    return raw


def _collect_lm_eval_usages(tasks: str) -> list[DatasetUsage]:
    """Resolve known lm-eval tasks to their canonical benchmark sources."""
    usages: list[DatasetUsage] = []
    for raw_task in tasks.split(","):
        task = raw_task.strip()
        if not task:
            continue
        normalized = task.casefold()
        if normalized == "mmlu" or normalized.startswith("mmlu_"):
            usages.append(
                resolve_dataset_usage(
                    dataset="mmlu",
                    role=EVALUATION_ROLE,
                    consumer=f"evaluation.lm_eval_tasks:{task}",
                    config="all",
                )
            )
        elif normalized == "hellaswag":
            usages.append(
                resolve_dataset_usage(
                    dataset="hellaswag",
                    role=EVALUATION_ROLE,
                    consumer="evaluation.lm_eval_tasks:hellaswag",
                )
            )
        else:
            usages.append(
                resolve_dataset_usage(
                    dataset=f"lm_eval:{task}",
                    role=EVALUATION_ROLE,
                    split="benchmark",
                    consumer=f"evaluation.lm_eval_tasks:{task}",
                )
            )
    return usages


def _collect_evaluation_usages(cfg: Mapping[str, Any]) -> list[DatasetUsage]:
    """Resolve all active evaluation and evaluation-context data sources."""
    evaluation = cfg.get("evaluation", {})
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation must be a mapping.")
    if not evaluation.get("enabled", False):
        return []

    selected = _selected_evaluation_targets(evaluation)
    usages: list[DatasetUsage] = []

    raw_vqa_tasks = evaluation.get("vlm_tasks") or []
    if isinstance(raw_vqa_tasks, str):
        vqa_tasks = [task.strip() for task in raw_vqa_tasks.split(",") if task.strip()]
    elif isinstance(raw_vqa_tasks, Sequence):
        vqa_tasks = [str(task).strip() for task in raw_vqa_tasks if str(task).strip()]
    else:
        raise TypeError("evaluation.vlm_tasks must be a sequence or string.")
    if _target_enabled(
        evaluation,
        selected,
        "vqa",
        default_enabled=bool(vqa_tasks),
    ):
        usages.extend(
            resolve_dataset_usage(
                dataset=task,
                role=EVALUATION_ROLE,
                consumer=f"evaluation.vlm_tasks:{task}",
                n_samples=_coerce_optional_int(evaluation.get("n_samples")),
            )
            for task in vqa_tasks
        )

    if _target_enabled(
        evaluation,
        selected,
        "coco",
        default_enabled=bool(evaluation.get("coco", False)),
    ):
        usages.append(
            resolve_dataset_usage(
                dataset="coco",
                role=EVALUATION_ROLE,
                consumer="evaluation.coco",
                n_samples=_coerce_optional_int(evaluation.get("n_samples")),
            )
        )

    raw_llava = evaluation.get("llava_bench")
    llava_default = (
        bool(raw_llava.get("enabled", False))
        if isinstance(raw_llava, Mapping)
        else bool(raw_llava)
    )
    if _target_enabled(
        evaluation,
        selected,
        "llava_bench",
        default_enabled=llava_default,
    ):
        if isinstance(raw_llava, Mapping):
            llava_cfg = raw_llava
        elif raw_llava is None or isinstance(raw_llava, bool):
            llava_cfg = {}
        else:
            raise TypeError("evaluation.llava_bench must be a mapping or boolean.")
        usages.append(
            resolve_dataset_usage(
                dataset=str(
                    llava_cfg.get(
                        "dataset",
                        "lmms-lab/llava-bench-in-the-wild",
                    )
                ),
                role=EVALUATION_ROLE,
                split=llava_cfg.get("split"),
                consumer="evaluation.llava_bench",
                n_samples=_coerce_optional_int(
                    llava_cfg.get("n_samples", evaluation.get("n_samples"))
                ),
            )
        )

    videomme = _mapping_target_config(evaluation, selected, "videomme")
    if videomme is not None:
        usages.append(
            resolve_dataset_usage(
                dataset=str(videomme.get("dataset", "lmms-eval/Video-MME")),
                role=EVALUATION_ROLE,
                split=videomme.get("split"),
                consumer="evaluation.videomme",
                n_samples=_coerce_optional_int(videomme.get("n_samples")),
            )
        )

    mmlu = _mapping_target_config(evaluation, selected, "mmlu")
    if mmlu is not None:
        usages.append(
            resolve_dataset_usage(
                dataset="mmlu",
                role=EVALUATION_ROLE,
                split=mmlu.get("split"),
                config="all",
                consumer="evaluation.mmlu",
                n_samples=_coerce_optional_int(mmlu.get("n_samples")),
            )
        )
        mmlu_n_shots = int(mmlu.get("n_shots", 5))
        if mmlu_n_shots > 0:
            usages.append(
                resolve_dataset_usage(
                    dataset="mmlu",
                    role=FEW_SHOT_ROLE,
                    split=mmlu.get("few_shot_split"),
                    config="all",
                    consumer="evaluation.mmlu.few_shot_context",
                    n_samples=mmlu_n_shots,
                )
            )

    hellaswag = _mapping_target_config(evaluation, selected, "hellaswag")
    if hellaswag is not None:
        usages.append(
            resolve_dataset_usage(
                dataset="hellaswag",
                role=EVALUATION_ROLE,
                split=hellaswag.get("split"),
                consumer="evaluation.hellaswag",
                n_samples=_coerce_optional_int(hellaswag.get("n_samples")),
            )
        )

    mmmu = _mapping_target_config(evaluation, selected, "mmmu")
    if mmmu is not None:
        dataset = str(mmmu.get("dataset") or "MMMU/MMMU")
        policy = get_dataset_policy(dataset)
        is_pro = policy is not None and policy.key == "mmmu_pro_vision"
        subjects = mmmu.get("subjects")
        if isinstance(subjects, str):
            subject_configs: list[str | None] = [subjects]
        elif isinstance(subjects, Sequence):
            subject_configs = [str(subject) for subject in subjects]
        else:
            subject_configs = ["*"]
        for subject in subject_configs:
            usages.append(
                resolve_dataset_usage(
                    dataset=dataset,
                    role=EVALUATION_ROLE,
                    split=mmmu.get("split", "test" if is_pro else "validation"),
                    config=subject,
                    consumer="evaluation.mmmu",
                    n_samples=_coerce_optional_int(mmmu.get("n_samples")),
                )
            )
        if not is_pro and int(mmmu.get("n_shots", 5)) > 0:
            usages.append(
                resolve_dataset_usage(
                    dataset=dataset,
                    role=EVALUATION_CONTEXT_ROLE,
                    split="test",
                    config="*",
                    consumer="evaluation.mmmu.few_shot_context",
                    n_samples=int(mmmu.get("n_shots", 5)),
                )
            )

    model_family = str(cfg.get("model", {}).get("family", "")).casefold()
    if model_family == "llama":
        raw_ppl = evaluation.get("perplexity")
        ppl_enabled = _target_enabled(
            evaluation,
            selected,
            "ppl",
            default_enabled=bool(raw_ppl),
        )
        if ppl_enabled:
            if isinstance(raw_ppl, Mapping):
                ppl_cfg = raw_ppl
            elif raw_ppl is None or isinstance(raw_ppl, bool):
                ppl_cfg = {}
            else:
                raise TypeError(
                    "evaluation.perplexity must be a mapping, boolean, or null."
                )
            usages.append(
                resolve_dataset_usage(
                    dataset=str(ppl_cfg.get("dataset", "Salesforce/wikitext")),
                    role=EVALUATION_ROLE,
                    split=ppl_cfg.get("split", "test"),
                    config=ppl_cfg.get("dataset_config", "wikitext-2-raw-v1"),
                    consumer="evaluation.perplexity",
                )
            )

        tasks = evaluation.get("lm_eval_tasks")
        if _target_enabled(
            evaluation,
            selected,
            "lm_eval",
            default_enabled=bool(tasks),
        ):
            if not isinstance(tasks, str) or not tasks.strip():
                raise ValueError(
                    "evaluation.lm_eval_tasks must be a non-empty string when active."
                )
            usages.extend(_collect_lm_eval_usages(tasks))
    else:
        ppl = _mapping_target_config(evaluation, selected, "ppl")
        if ppl is not None:
            usages.append(
                resolve_dataset_usage(
                    dataset=str(ppl.get("dataset", "wikitext2")),
                    role=EVALUATION_ROLE,
                    split=ppl.get("split"),
                    config=ppl.get("dataset_config"),
                    consumer="evaluation.ppl",
                )
            )

    return usages


def _store_provenance(
    cfg: dict[str, Any],
    calibration_usages: Sequence[DatasetUsage],
    evaluation_usages: Sequence[DatasetUsage],
    *,
    transductive: bool,
    violations: Sequence[str],
) -> None:
    """Store resolved data sources in the configuration saved by the runner."""
    calibration = cfg.setdefault("calibration", {})
    if not isinstance(calibration, dict):
        raise TypeError("calibration must be a mutable mapping.")
    calibration["resolved_sources"] = [
        usage.to_config() for usage in calibration_usages
    ]
    calibration["transductive"] = transductive
    if violations:
        calibration["dataset_usage_warnings"] = list(violations)
    else:
        calibration.pop("dataset_usage_warnings", None)

    evaluation = cfg.get("evaluation")
    if isinstance(evaluation, dict):
        evaluation["resolved_sources"] = [
            usage.to_config() for usage in evaluation_usages
        ]


def _print_usage_summary(
    calibration_usages: Sequence[DatasetUsage],
    evaluation_usages: Sequence[DatasetUsage],
) -> None:
    """Print resolved calibration and evaluation provenance."""
    if calibration_usages:
        print("=== Resolved calibration data ===")
        for usage in calibration_usages:
            print(f"  - {usage.describe()}")
    if evaluation_usages:
        print("=== Resolved evaluation data ===")
        for usage in evaluation_usages:
            print(f"  - {usage.describe()}")
    if calibration_usages or evaluation_usages:
        print()


def validate_recipe_dataset_usage(
    cfg: dict[str, Any],
    *,
    include_calibration: bool,
    emit_summary: bool = True,
) -> tuple[list[DatasetUsage], list[DatasetUsage]]:
    """Validate recipe data roles and overlap before loading models or datasets."""
    calibration_usages = _collect_calibration_usages(cfg) if include_calibration else []
    evaluation_usages = _collect_evaluation_usages(cfg)

    target_violations: list[str] = []
    evaluation_violations: list[str] = []
    transductive_violations: list[str] = []
    for usage in calibration_usages:
        for risk in _target_inclusion_risks(usage):
            target_violations.append(f"{usage.consumer}: {risk}")
        for risk in dataset_usage_risks(usage):
            transductive_violations.append(f"{usage.consumer}: {risk}")

    for usage in evaluation_usages:
        for risk in dataset_usage_risks(usage):
            evaluation_violations.append(f"{usage.consumer}: {risk}")

    unwaivable_violations = target_violations + evaluation_violations
    if unwaivable_violations:
        raise DatasetUsageError(
            _format_usage_error(
                calibration_usages,
                evaluation_usages,
                unwaivable_violations,
            )
        )

    for calibration in calibration_usages:
        for evaluation in evaluation_usages:
            if dataset_usages_overlap(calibration, evaluation):
                transductive_violations.append(
                    "calibration/evaluation overlap: "
                    f"calibration=({calibration.describe()}); "
                    f"evaluation=({evaluation.describe()})"
                )

    calibration_cfg = cfg.get("calibration", {})
    allow_overlap = bool(
        isinstance(calibration_cfg, Mapping)
        and calibration_cfg.get("allow_benchmark_overlap", False)
    )
    transductive = bool(transductive_violations)

    if transductive_violations and not allow_overlap:
        raise DatasetUsageError(
            _format_usage_error(
                calibration_usages,
                evaluation_usages,
                transductive_violations,
            )
        )

    _store_provenance(
        cfg,
        calibration_usages,
        evaluation_usages,
        transductive=transductive,
        violations=transductive_violations,
    )

    if transductive_violations:
        message = (
            "TRANSDUCTIVE DATASET USAGE ENABLED: benchmark data is being used "
            "to fit quantization parameters. Results from this checkpoint are "
            "not strictly held out. " + " | ".join(transductive_violations)
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        if emit_summary:
            border = "!" * 80
            print(f"\n{border}\nWARNING: {message}\n{border}\n")

    if emit_summary:
        _print_usage_summary(calibration_usages, evaluation_usages)
    return calibration_usages, evaluation_usages
