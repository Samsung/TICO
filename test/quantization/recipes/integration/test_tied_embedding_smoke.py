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

import contextlib
import importlib.util
import io
import tempfile
import unittest
from unittest.mock import patch

HAS_CIRCLE_SCHEMA = importlib.util.find_spec("circle_schema") is not None


class FakeCircleModel:
    """Fake Circle model returned by the patched converter."""

    circle_binary = b"fake-circle"

    def __init__(self):
        self.saved_path = None

    def save(self, path):
        """Record the path used by the smoke helper."""
        self.saved_path = path


@unittest.skipUnless(
    HAS_CIRCLE_SCHEMA, "circle_schema is required for tied embedding smoke"
)
class TestTiedEmbeddingSmoke(unittest.TestCase):
    def test_tied_embedding_smoke_runs_with_patched_circle_export(self):
        """The tied embedding smoke helper should exercise real WrapQ prepare/convert logic."""
        import tico.quantization.recipes.debug.tied_embedding as tied_mod
        from tico.quantization.recipes.debug.tied_embedding import (
            run_tied_embedding_smoke,
            TiedEmbeddingSmokeConfig,
        )

        fake_circle = FakeCircleModel()

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            tied_mod.tico, "convert", lambda model, args: fake_circle
        ), patch.object(
            tied_mod,
            "_circle_data_tensors_with_shape",
            lambda circle_binary, shape: [(0, "shared_weight", 1, 32, 0)],
        ):
            from pathlib import Path

            save_path = Path(tmpdir) / "tied_embedding.q.circle"
            with contextlib.redirect_stdout(io.StringIO()):
                run_tied_embedding_smoke(
                    TiedEmbeddingSmokeConfig(
                        vocab_size=8,
                        hidden_size=4,
                        batch_size=1,
                        seq_len=3,
                        calib_iters=2,
                        save_path=str(save_path),
                        skip_circle_sharing_check=False,
                    )
                )

        self.assertEqual(fake_circle.saved_path, save_path)
