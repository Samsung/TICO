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

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping

from tico.quantization.analysis.weight_sparsity import (
    format_layer_weight_sparsity_table,
    format_weight_sparsity_table,
    measure_weight_sparsity,
    measure_weight_sparsity_report,
    write_layer_weight_sparsity_csv,
    write_layer_weight_sparsity_markdown,
    write_weight_sparsity_csv,
    write_weight_sparsity_markdown,
)
from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.base import Stage


_DEFAULT_FORMATS = ("csv", "markdown")


def _normalize_formats(value: Any) -> tuple[str, ...]:
    """Normalize configured output formats to canonical names."""

    if value is None:
        return _DEFAULT_FORMATS
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise TypeError("weight_sparsity.formats must be a string or sequence.")

    normalized: list[str] = []
    for item in values:
        name = str(item).strip().lower()
        if name == "md":
            name = "markdown"
        if name not in {"csv", "markdown"}:
            raise ValueError(
                "Unsupported weight sparsity format "
                f"{item!r}. Supported formats: csv, markdown."
            )
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def _resolve_output_dir(
    ctx: RecipeContext, stage_cfg: Mapping[str, Any]
) -> Path | None:
    """Resolve the report directory from the stage or export configuration."""

    configured = stage_cfg.get("output_dir")
    if configured is None:
        export_cfg = ctx.cfg.get("export", {})
        if isinstance(export_cfg, Mapping):
            configured = export_cfg.get("output_dir")
    if configured is None:
        return None
    return Path(str(configured))


class WeightSparsityStage(Stage):
    """Measure summary and layer-level weight sparsity after PTQ conversion."""

    name = "weight_sparsity"

    def run(self, ctx: RecipeContext, stage_cfg: Mapping[str, Any]) -> RecipeContext:
        """Measure, print, and optionally save post-convert sparsity reports."""

        precision = int(stage_cfg.get("precision", 6))
        max_chunk_elements = int(stage_cfg.get("max_chunk_elements", 4 * 1024 * 1024))
        deduplicate_shared_weights = bool(
            stage_cfg.get("deduplicate_shared_weights", True)
        )
        include_empty_scopes = bool(stage_cfg.get("include_empty_scopes", False))
        include_layer_report = bool(stage_cfg.get("include_layer_report", True))
        include_layer_totals = bool(stage_cfg.get("include_layer_totals", True))
        output_dir = _resolve_output_dir(ctx, stage_cfg)
        print_layer_report = bool(
            stage_cfg.get("print_layer_report", output_dir is None)
        )

        if include_layer_report:
            report = measure_weight_sparsity_report(
                ctx.require_model(),
                family=ctx.adapter.family,
                max_chunk_elements=max_chunk_elements,
                deduplicate_shared_weights=deduplicate_shared_weights,
                include_empty_scopes=include_empty_scopes,
                include_layer_totals=include_layer_totals,
            )
            rows = report.summary_rows
            layer_rows = report.layer_rows
        else:
            rows = tuple(
                measure_weight_sparsity(
                    ctx.require_model(),
                    family=ctx.adapter.family,
                    max_chunk_elements=max_chunk_elements,
                    deduplicate_shared_weights=deduplicate_shared_weights,
                    include_empty_scopes=include_empty_scopes,
                )
            )
            layer_rows = ()

        print("=== Post-convert weight sparsity ===")
        print(format_weight_sparsity_table(rows, precision=precision))
        print()

        if include_layer_report and print_layer_report:
            print("=== Post-convert weight sparsity by layer ===")
            print(format_layer_weight_sparsity_table(layer_rows, precision=precision))
            print()

        if output_dir is None:
            return ctx

        filename_stem = str(stage_cfg.get("filename_stem", "weight_sparsity"))
        layer_filename_stem = str(
            stage_cfg.get("layer_filename_stem", f"{filename_stem}_by_layer")
        )
        formats = _normalize_formats(stage_cfg.get("formats"))
        output_dir.mkdir(parents=True, exist_ok=True)

        if "csv" in formats:
            csv_path = write_weight_sparsity_csv(
                rows,
                output_dir / f"{filename_stem}.csv",
                precision=precision,
            )
            ctx.artifacts["weight_sparsity_csv"] = csv_path
            print(f"Saved weight sparsity CSV to {csv_path.resolve()}")

            if include_layer_report:
                layer_csv_path = write_layer_weight_sparsity_csv(
                    layer_rows,
                    output_dir / f"{layer_filename_stem}.csv",
                    precision=precision,
                )
                ctx.artifacts["weight_sparsity_by_layer_csv"] = layer_csv_path
                print(
                    "Saved layer-level weight sparsity CSV to "
                    f"{layer_csv_path.resolve()}"
                )

        if "markdown" in formats:
            markdown_path = write_weight_sparsity_markdown(
                rows,
                output_dir / f"{filename_stem}.md",
                precision=precision,
            )
            ctx.artifacts["weight_sparsity_markdown"] = markdown_path
            print(f"Saved weight sparsity Markdown to {markdown_path.resolve()}")

            if include_layer_report:
                layer_markdown_path = write_layer_weight_sparsity_markdown(
                    layer_rows,
                    output_dir / f"{layer_filename_stem}.md",
                    precision=precision,
                )
                ctx.artifacts["weight_sparsity_by_layer_markdown"] = layer_markdown_path
                print(
                    "Saved layer-level weight sparsity Markdown to "
                    f"{layer_markdown_path.resolve()}"
                )

        return ctx
