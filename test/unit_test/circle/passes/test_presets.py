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

from __future__ import annotations

import unittest

from tico.circle.passes import (
    CircleOptimizationPreset,
    CirclePassContext,
    CirclePassStrategy,
    create_o1_pipeline,
    create_optimization_preset,
)
from tico.circle.passes.cleanup import CompactIndicesPass

from test.unit_test.circle.infrastructure_fixture import make_empty_document


class CircleOptimizationPresetTest(unittest.TestCase):
    """Check O1 ordering, scheduling, fresh instances, and idempotence."""

    def test_o1_uses_round_fixed_point_then_one_compaction(self) -> None:
        """Run complete optimization rounds before one final compaction."""

        pipeline = create_o1_pipeline(maximum_steps=123)

        self.assertEqual(
            [phase.name for phase in pipeline.phases],
            ["optimize", "compact"],
        )
        optimization = pipeline.phases[0].manager
        finalization = pipeline.phases[1].manager
        self.assertIs(
            optimization.strategy,
            CirclePassStrategy.UNTIL_NO_CHANGE,
        )
        self.assertEqual(optimization.maximum_steps, 123)
        self.assertEqual(
            [circle_pass.__class__.__name__ for circle_pass in optimization.passes],
            [
                "CanonicalizeEquivalentOpsPass",
                "SimplifyViewOpsPass",
                "EliminateTransposeBoundedLayoutRegionPass",
                "SimplifyReductionOpsPass",
                "EliminateIdentityOpsPass",
                "SimplifyArithmeticPass",
                "FuseCompositeOpsPass",
                "FuseLinearOpsPass",
                "FoldConstantsPass",
                "CommonSubexpressionEliminationPass",
                "DeadCodeEliminationPass",
            ],
        )
        self.assertIs(finalization.strategy, CirclePassStrategy.ONCE)
        self.assertEqual(len(finalization.passes), 1)
        self.assertIsInstance(finalization.passes[0], CompactIndicesPass)

    def test_preset_factory_returns_fresh_pass_instances(self) -> None:
        """Avoid sharing mutable pass state between independent preset requests."""

        first = create_optimization_preset(CircleOptimizationPreset.O1)
        second = create_optimization_preset("o1")

        for first_phase, second_phase in zip(first.phases, second.phases):
            self.assertIsNot(first_phase.manager, second_phase.manager)
            for first_pass, second_pass in zip(
                first_phase.manager.passes,
                second_phase.manager.passes,
            ):
                self.assertIsNot(first_pass, second_pass)

    def test_o1_is_idempotent_on_an_empty_valid_fixture(self) -> None:
        """Report no change when O1 is repeated on an already stable fixture."""

        document = make_empty_document()
        context = CirclePassContext(verify_after_each_pass=False)

        first = create_o1_pipeline().run(document, context)
        second = create_o1_pipeline().run(document, context)

        self.assertFalse(first.modified)
        self.assertFalse(second.modified)


if __name__ == "__main__":
    unittest.main()
