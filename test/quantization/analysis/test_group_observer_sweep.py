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

"""Tests for group-specific activation-observer override sweeps."""

from __future__ import annotations

import unittest

from types import SimpleNamespace
from unittest.mock import patch

from examples.hand_detector._support.group_observer_sweep import (
    activation_group_override_paths,
    build_group_observer_policies,
    build_group_observer_sweep_result,
    EvaluatedGroupObserverPolicy,
    make_group_observer_overrides,
    validate_group_observer_overrides,
)
from examples.hand_detector._support.quantization import make_ptq_config
from examples.hand_detector._support.sensitivity import ActivationSensitivityGroup
from tico.quantization.analysis import QuantizationGroup, SiteSelector
from tico.quantization.wrapq.control import QuantizationSite, SiteRole
from tico.quantization.wrapq.dtypes import DType
from tico.quantization.wrapq.observers.minmax import MinMaxObserver
from tico.quantization.wrapq.observers.percentile import PercentileObserver
from tico.quantization.wrapq.qscheme import QScheme


class GroupObserverSweepTest(unittest.TestCase):
    """Verify override paths, candidate policies, and group-local rankings."""

    def test_override_paths_use_original_float_module_names(self) -> None:
        """Remove PTQWrapper implementation details from override paths."""
        group = _group()
        sites = (
            _site(
                "detector.layers.0.conv.wrapped.act_out",
                "detector.layers.0.conv.wrapped",
                "detector.layers.0.conv",
                "act_out",
                SiteRole.ACTIVATION_OUTPUT,
            ),
            _site(
                "detector.layers.1.wrapped.act_in",
                "detector.layers.1.wrapped",
                "detector.layers.1",
                "act_in",
                SiteRole.ACTIVATION_INPUT,
            ),
        )
        with patch(
            "examples.hand_detector._support.group_observer_sweep."
            "iter_quantization_sites",
            return_value=iter(sites),
        ):
            paths = activation_group_override_paths(SimpleNamespace(), group)

        self.assertEqual(
            paths,
            (
                "detector.layers.0.conv.act_out",
                "detector.layers.1.act_in",
            ),
        )

    def test_candidate_builder_skips_the_global_percentile_control(self) -> None:
        """Avoid recalibrating a group with the unchanged global policy."""
        policies = build_group_observer_policies(
            percentiles=(99.9, 99.99, 99.995, 99.9, 100.0),
            global_percentile=99.99,
            max_samples=1024,
            samples_per_batch=128,
            seed=7,
        )

        self.assertEqual(
            tuple(policy.name for policy in policies),
            ("minmax", "percentile_99_9", "percentile_99_995"),
        )
        self.assertIs(policies[0].observer, MinMaxObserver)

    def test_override_quant_spec_replaces_the_global_observer_role(self) -> None:
        """Use QuantSpec leaves so percentile-only kwargs do not leak to MinMax."""
        policy = build_group_observer_policies(
            percentiles=(99.99,),
            global_percentile=99.99,
            max_samples=1024,
            samples_per_batch=128,
            seed=7,
        )[0]
        overrides = make_group_observer_overrides(
            policy,
            bit_width=8,
            override_paths=("detector.layers.0.conv.act_out",),
        )
        kwargs = overrides["detector.layers.0.conv.act_out"].to_kwargs(
            obs_name="act_out",
            context="test",
            mark_replace=True,
        )

        self.assertIs(kwargs["observer"], MinMaxObserver)
        self.assertTrue(kwargs["__quant_spec_replace_role__"])
        self.assertNotIn("percentile", kwargs)

    def test_nested_ptq_config_routes_override_to_the_exact_observer(self) -> None:
        """Pass dot-path QuantSpec overrides through the example PTQ helper."""
        policy = build_group_observer_policies(
            percentiles=(99.9,),
            global_percentile=99.99,
            max_samples=1024,
            samples_per_batch=128,
            seed=7,
            include_minmax=False,
        )[0]
        path = "detector.layers.0.conv.act_out"
        config = make_ptq_config(
            8,
            activation_observer=PercentileObserver,
            activation_observer_kwargs={
                "percentile": 99.99,
                "max_samples": 1024,
                "samples_per_batch": 128,
                "seed": 7,
            },
            overrides=make_group_observer_overrides(
                policy,
                bit_width=8,
                override_paths=(path,),
            ),
        )
        scoped = config.child("detector").child("layers").child("0").child("conv")
        kwargs = scoped.get_kwargs("act_out")

        self.assertIs(kwargs["observer"], PercentileObserver)
        self.assertEqual(kwargs["percentile"], 99.9)
        self.assertTrue(kwargs["__quant_spec_replace_role__"])

    def test_override_validation_checks_class_and_percentile(self) -> None:
        """Fail fast when a PTQConfig override did not reach the prepared site."""
        path = "detector.layers.0.conv.act_out"
        policy = build_group_observer_policies(
            percentiles=(99.9,),
            global_percentile=99.99,
            max_samples=1024,
            samples_per_batch=128,
            seed=7,
            include_minmax=False,
        )[0]
        observer = PercentileObserver(
            name="act_out",
            dtype=DType.uint(8),
            qscheme=QScheme.PER_TENSOR_ASYMM,
            percentile=99.9,
            max_samples=1024,
            samples_per_batch=128,
            seed=7,
        )
        site = _site(
            "detector.layers.0.conv.wrapped.act_out",
            "detector.layers.0.conv.wrapped",
            "detector.layers.0.conv",
            "act_out",
            SiteRole.ACTIVATION_OUTPUT,
            observer=observer,
        )
        with patch(
            "examples.hand_detector._support.group_observer_sweep."
            "iter_quantization_sites",
            return_value=iter((site,)),
        ):
            validate_group_observer_overrides(
                SimpleNamespace(),
                policy,
                (path,),
            )

        observer.percentile = 99.95
        with patch(
            "examples.hand_detector._support.group_observer_sweep."
            "iter_quantization_sites",
            return_value=iter((site,)),
        ):
            with self.assertRaisesRegex(RuntimeError, "mismatched percentile"):
                validate_group_observer_overrides(
                    SimpleNamespace(),
                    policy,
                    (path,),
                )

    def test_global_minmax_does_not_duplicate_the_control_policy(self) -> None:
        """Avoid evaluating MinMax twice when the global percentile is 100."""
        policies = build_group_observer_policies(
            percentiles=(99.9,),
            global_percentile=100.0,
            max_samples=1024,
            samples_per_batch=128,
            seed=7,
        )
        self.assertEqual(
            tuple(policy.name for policy in policies),
            ("percentile_99_9",),
        )

    def test_result_can_keep_global_policy_when_overrides_are_worse(self) -> None:
        """Treat no group override as a real candidate rather than forcing change."""
        policies = build_group_observer_policies(
            percentiles=(99.9,),
            global_percentile=99.99,
            max_samples=1024,
            samples_per_batch=128,
            seed=7,
        )
        evaluations = (
            EvaluatedGroupObserverPolicy(
                policy=policies[0],
                outputs=_outputs(1.2, 0.6),
                enabled_site_count=11,
            ),
            EvaluatedGroupObserverPolicy(
                policy=policies[1],
                outputs=_outputs(1.1, 0.55),
                enabled_site_count=11,
            ),
        )
        result = build_group_observer_sweep_result(
            group=_group(),
            override_paths=(
                "detector.layers.0.conv.act_out",
                "detector.layers.1.act_in",
            ),
            global_percentile=99.99,
            baseline_outputs=_outputs(1.0, 0.5),
            baseline_site_count=11,
            evaluations=evaluations,
            score_output="regressors",
        )

        self.assertEqual(result["best_candidate"], "global_percentile_99_99")
        self.assertEqual(float(result["best_score_improvement"]), 0.0)
        candidates = result["candidates"]
        self.assertTrue(candidates[0]["is_global_baseline"])
        self.assertEqual(candidates[0]["rank"], 1)
        self.assertLess(float(candidates[1]["score_improvement"]), 0.0)


def _group() -> ActivationSensitivityGroup:
    paths = (
        "detector.layers.0.conv.wrapped.act_out",
        "detector.layers.1.wrapped.act_in",
    )
    return ActivationSensitivityGroup(
        group=QuantizationGroup("stem", SiteSelector.paths(*paths)),
        kind="stem",
        operation_positions=(0, 1),
        operation_indices=(2, 4),
        operation_names=("CONV_2D", "PRELU"),
        tensor_ids=(141, 142),
        site_paths=paths,
    )


def _site(
    path: str,
    module_path: str,
    fp_name: str,
    observer_name: str,
    role: SiteRole,
    *,
    observer: object | None = None,
) -> QuantizationSite:
    return QuantizationSite(
        path=path,
        module_path=module_path,
        observer_name=observer_name,
        role=role,
        module=SimpleNamespace(fp_name=fp_name),  # type: ignore[arg-type]
        observer=(
            observer if observer is not None else SimpleNamespace()
        ),  # type: ignore[arg-type]
    )


def _outputs(
    regressor_mae: float,
    classifier_mae: float,
) -> dict[str, dict[str, float]]:
    return {
        "regressors": {
            "mae": regressor_mae,
            "cosine_similarity": 0.99,
        },
        "classifiers": {
            "mae": classifier_mae,
            "cosine_similarity": 0.999,
        },
    }


if __name__ == "__main__":
    unittest.main()
