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

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tico.quantization.recipes.context import RecipeContext
from tico.quantization.recipes.stages.weight_sparsity import WeightSparsityStage


class _FakeDType:
    qmin = 0
    qmax = 15

    def __str__(self) -> str:
        return "uint4"


class _FakeObserver(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dtype = _FakeDType()
        self.channel_axis = 0
        self.register_buffer("_cached_scale", torch.tensor([1.0]))
        self.register_buffer("_cached_zp", torch.tensor([7], dtype=torch.int32))


class _Leaf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fp_name = "model.layers.0.self_attn.q_proj"
        self._mode = SimpleNamespace(name="QUANT")
        self.module = torch.nn.Linear(2, 1, bias=False)
        self.module.weight.data.zero_()
        self.observer = _FakeObserver()

    def get_observer(self, name: str, *, recurse: bool = True):
        if name == "weight" and not recurse:
            return self.observer
        return None


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.leaf = _Leaf()


class TestWeightSparsityStage(unittest.TestCase):
    def test_stage_writes_csv_and_markdown_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = RecipeContext(
                cfg={},
                adapter=SimpleNamespace(family="llama"),
                model=_Model(),
            )
            stage = WeightSparsityStage()
            returned = stage.run(
                ctx,
                {
                    "output_dir": tmpdir,
                    "formats": ["csv", "markdown"],
                    "precision": 3,
                },
            )

            self.assertIs(returned, ctx)
            csv_path = Path(ctx.artifacts["weight_sparsity_csv"])
            markdown_path = Path(ctx.artifacts["weight_sparsity_markdown"])
            layer_csv_path = Path(ctx.artifacts["weight_sparsity_by_layer_csv"])
            layer_markdown_path = Path(
                ctx.artifacts["weight_sparsity_by_layer_markdown"]
            )
            self.assertTrue(csv_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertTrue(layer_csv_path.is_file())
            self.assertTrue(layer_markdown_path.is_file())
            self.assertEqual(
                csv_path.read_text(encoding="utf-8").splitlines()[0],
                "scope,qdtype,sparsity_percent",
            )
            self.assertEqual(
                layer_csv_path.read_text(encoding="utf-8").splitlines()[0],
                "layer,scope,qdtype,sparsity_percent",
            )
            self.assertIn(
                "| Scope | Qdtype | Sparsity (%) |",
                markdown_path.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "| Layer | Scope | Qdtype | Sparsity (%) |",
                layer_markdown_path.read_text(encoding="utf-8"),
            )

    def test_stage_can_disable_layer_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ctx = RecipeContext(
                cfg={},
                adapter=SimpleNamespace(family="llama"),
                model=_Model(),
            )
            stage = WeightSparsityStage()
            stage.run(
                ctx,
                {
                    "output_dir": tmpdir,
                    "formats": ["csv"],
                    "include_layer_report": False,
                },
            )

            self.assertIn("weight_sparsity_csv", ctx.artifacts)
            self.assertNotIn("weight_sparsity_by_layer_csv", ctx.artifacts)
            self.assertFalse((Path(tmpdir) / "weight_sparsity_by_layer.csv").exists())


if __name__ == "__main__":
    unittest.main()
