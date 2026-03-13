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

"""Tests for tico/utils/pytree_utils.py.

Each test class covers one cache type.  All tests are skipped when
transformers is not installed so the suite stays green in minimal
environments.
"""

import unittest

import torch
import torch.utils._pytree as pytree

from tico.utils.installed_packages import is_transformers_installed

_SKIP = not is_transformers_installed()
_SKIP_REASON = "transformers is not installed"


def _make_tensor(*shape):
    return torch.randn(*shape)


# ---------------------------------------------------------------------------
# Helper: round-trip a registered pytree node through flatten → unflatten
# ---------------------------------------------------------------------------


def _roundtrip(obj):
    """Flatten obj with torch pytree, then unflatten and return."""
    leaves, treespec = pytree.tree_flatten(obj)
    return pytree.tree_unflatten(leaves, treespec)


# ---------------------------------------------------------------------------
# DynamicCache
# ---------------------------------------------------------------------------


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestRegisterDynamicCache(unittest.TestCase):
    def setUp(self):
        from tico.utils.pytree_utils import (
            register_dynamic_cache,
            register_dynamic_layer,
        )

        register_dynamic_cache()
        register_dynamic_layer()

    def _make_cache(self):
        import transformers
        from packaging.version import Version
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache()
        if Version(transformers.__version__) < Version("4.54.0"):
            # Legacy attribute-based structure
            cache.key_cache = [_make_tensor(1, 4, 8, 16)]
            cache.value_cache = [_make_tensor(1, 4, 8, 16)]
        else:
            # Layer-based structure — populate via standard update call
            # so that cache.layers is initialised correctly.
            k = _make_tensor(1, 4, 8, 16)
            v = _make_tensor(1, 4, 8, 16)
            cache.update(k, v, layer_idx=0)
        return cache

    def test_roundtrip_leaves_preserved(self):
        """Flatten → unflatten keeps all tensor data intact."""
        cache = self._make_cache()
        restored = _roundtrip(cache)

        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.assertEqual(len(restored.key_cache), len(cache.key_cache))
            torch.testing.assert_close(restored.key_cache[0], cache.key_cache[0])
            torch.testing.assert_close(restored.value_cache[0], cache.value_cache[0])
        else:
            self.assertEqual(len(restored.layers), len(cache.layers))

    def test_idempotent_registration(self):
        """Calling register_dynamic_cache a second time must not raise."""
        from tico.utils.pytree_utils import register_dynamic_cache

        register_dynamic_cache()  # second call — should be a silent no-op

    def test_flatten_returns_tensors(self):
        """Flattened leaves must all be tensors."""
        cache = self._make_cache()
        leaves, _ = pytree.tree_flatten(cache)
        for leaf in leaves:
            self.assertIsInstance(leaf, torch.Tensor)


# ---------------------------------------------------------------------------
# StaticCache
# ---------------------------------------------------------------------------


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestRegisterStaticCache(unittest.TestCase):
    def setUp(self):
        from tico.utils.pytree_utils import register_static_cache

        register_static_cache()

    def _make_cache(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.skipTest("StaticCache with layers API requires transformers >= 4.54.0")

        from transformers import AutoConfig
        from transformers.cache_utils import StaticCache

        # Build a minimal config
        cfg = AutoConfig.for_model("llama")
        cfg.num_hidden_layers = 2
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 4
        cfg.head_dim = 8
        cfg.max_position_embeddings = 32
        return StaticCache(config=cfg, max_batch_size=1, max_cache_len=16)

    def test_roundtrip_layers_preserved(self):
        cache = self._make_cache()
        n_layers_before = len(cache.layers)
        restored = _roundtrip(cache)
        self.assertEqual(len(restored.layers), n_layers_before)

    def test_idempotent_registration(self):
        from tico.utils.pytree_utils import register_static_cache

        register_static_cache()


# ---------------------------------------------------------------------------
# StaticLayer
# ---------------------------------------------------------------------------

import inspect


def _create_static_layer(**potential_kwargs):
    # torch ver |
    # ----------|------
    # 2.6.0     | requires max_cache_len
    # ...
    # 2.10.0    | requires max_cache_len, batch_size, num_heads, head_dim
    # 2.12.0.dev| requires max_cache_len
    from transformers.cache_utils import StaticLayer

    sig = inspect.signature(StaticLayer.__init__)
    valid_params = set(sig.parameters.keys())

    init_kwargs = {k: v for k, v in potential_kwargs.items() if k in valid_params}

    obj = StaticLayer(**init_kwargs)
    return obj


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestRegisterStaticLayer(unittest.TestCase):
    def setUp(self):
        from tico.utils.pytree_utils import register_static_layer

        register_static_layer()

    def _make_layer(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.skipTest("StaticLayer requires transformers >= 4.54.0")

        from transformers.cache_utils import StaticLayer

        layer = _create_static_layer(
            max_cache_len=16, batch_size=1, num_heads=4, head_dim=4
        )
        layer.is_initialized = True
        layer.keys = _make_tensor(1, 4, 16, 8)
        layer.values = _make_tensor(1, 4, 16, 8)
        layer.dtype = layer.keys.dtype
        layer.device = layer.keys.device
        layer.max_batch_size = 1
        layer.num_heads = 4
        layer.head_dim = 8
        return layer

    def test_roundtrip_tensors_preserved(self):
        layer = self._make_layer()
        restored = _roundtrip(layer)
        torch.testing.assert_close(restored.keys, layer.keys)
        torch.testing.assert_close(restored.values, layer.values)

    def test_roundtrip_metadata_preserved(self):
        layer = self._make_layer()
        restored = _roundtrip(layer)
        self.assertEqual(restored.max_cache_len, layer.max_cache_len)
        self.assertEqual(restored.num_heads, layer.num_heads)
        self.assertEqual(restored.head_dim, layer.head_dim)

    def test_uninitialised_layer_raises(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.skipTest("StaticLayer requires transformers >= 4.54.0")

        from tico.utils.pytree_utils import _flatten_static_layer
        from transformers.cache_utils import StaticLayer

        layer = _create_static_layer(
            max_cache_len=16, batch_size=1, num_heads=2, head_dim=4
        )
        layer.is_initialized = False
        with self.assertRaises(ValueError):
            _flatten_static_layer(layer)

    def test_idempotent_registration(self):
        from tico.utils.pytree_utils import register_static_layer

        register_static_layer()


# ---------------------------------------------------------------------------
# DynamicLayer
# ---------------------------------------------------------------------------


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestRegisterDynamicLayer(unittest.TestCase):
    def setUp(self):
        from tico.utils.pytree_utils import register_dynamic_layer

        register_dynamic_layer()

    def _make_layer(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.skipTest("DynamicLayer requires transformers >= 4.54.0")

        from transformers.cache_utils import DynamicLayer

        layer = DynamicLayer()
        layer.is_initialized = True
        layer.keys = _make_tensor(1, 4, 8, 16)
        layer.values = _make_tensor(1, 4, 8, 16)
        layer.dtype = layer.keys.dtype
        layer.device = layer.keys.device
        return layer

    def test_roundtrip_tensors_preserved(self):
        layer = self._make_layer()
        restored = _roundtrip(layer)
        torch.testing.assert_close(restored.keys, layer.keys)
        torch.testing.assert_close(restored.values, layer.values)

    def test_roundtrip_metadata_preserved(self):
        layer = self._make_layer()
        restored = _roundtrip(layer)
        self.assertEqual(restored.is_initialized, layer.is_initialized)
        self.assertEqual(restored.dtype, layer.dtype)
        self.assertEqual(restored.device, layer.device)

    def test_uninitialised_layer_raises(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.skipTest("DynamicLayer requires transformers >= 4.54.0")

        from tico.utils.pytree_utils import _flatten_dynamic_layer
        from transformers.cache_utils import DynamicLayer

        layer = DynamicLayer()
        layer.is_initialized = False
        with self.assertRaises(ValueError):
            _flatten_dynamic_layer(layer)

    def test_idempotent_registration(self):
        from tico.utils.pytree_utils import register_dynamic_layer

        register_dynamic_layer()


# ---------------------------------------------------------------------------
# EncoderDecoderCache
# ---------------------------------------------------------------------------


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestRegisterEncoderDecoderCache(unittest.TestCase):
    def setUp(self):
        from tico.utils.pytree_utils import (
            register_dynamic_cache,
            register_encoder_decoder_cache,
        )

        register_dynamic_cache()
        register_encoder_decoder_cache()

    def _make_cache(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.skipTest(
                "EncoderDecoderCache with Layer-based internals requires transformers >= 4.54.0"
            )

        from transformers.cache_utils import DynamicCache, EncoderDecoderCache

        self_cache = DynamicCache()
        cross_cache = DynamicCache()
        k = _make_tensor(1, 4, 8, 16)
        v = _make_tensor(1, 4, 8, 16)
        self_cache.update(k, v, layer_idx=0)
        cross_cache.update(k.clone(), v.clone(), layer_idx=0)
        return EncoderDecoderCache(self_cache, cross_cache)

    def test_roundtrip_self_and_cross_caches_preserved(self):
        cache = self._make_cache()
        n_self = len(cache.self_attention_cache.layers)
        n_cross = len(cache.cross_attention_cache.layers)
        restored = _roundtrip(cache)
        self.assertEqual(len(restored.self_attention_cache.layers), n_self)
        self.assertEqual(len(restored.cross_attention_cache.layers), n_cross)

    def test_idempotent_registration(self):
        from tico.utils.pytree_utils import register_encoder_decoder_cache

        register_encoder_decoder_cache()


# ---------------------------------------------------------------------------
# Consistent flatten key paths
# ---------------------------------------------------------------------------


@unittest.skipIf(_SKIP, _SKIP_REASON)
class TestFlattenKeyPaths(unittest.TestCase):
    """The _flatten_with_keys_* helpers must return keys that match the
    children produced by the main _flatten_* function."""

    def test_dynamic_layer_keys_match_children(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.skipTest("DynamicLayer requires transformers >= 4.54.0")

        from tico.utils.pytree_utils import (
            _flatten_dynamic_layer,
            _flatten_with_keys_dynamic_layer,
        )
        from transformers.cache_utils import DynamicLayer

        layer = DynamicLayer()
        layer.is_initialized = True
        layer.keys = _make_tensor(1, 2, 4, 8)
        layer.values = _make_tensor(1, 2, 4, 8)
        layer.dtype = layer.keys.dtype
        layer.device = layer.keys.device

        children, _ = _flatten_dynamic_layer(layer)
        keyed, _ = _flatten_with_keys_dynamic_layer(layer)

        self.assertEqual(len(keyed), len(children))
        for (_, tensor_keyed), tensor_plain in zip(keyed, children):
            torch.testing.assert_close(tensor_keyed, tensor_plain)

    def test_static_layer_keys_match_children(self):
        import transformers
        from packaging.version import Version

        if Version(transformers.__version__) < Version("4.54.0"):
            self.skipTest("StaticLayer requires transformers >= 4.54.0")

        from tico.utils.pytree_utils import (
            _flatten_static_layer,
            _flatten_with_keys_static_layer,
        )
        from transformers.cache_utils import StaticLayer

        layer = _create_static_layer(
            max_cache_len=8, batch_size=1, num_heads=2, head_dim=4
        )

        layer.is_initialized = True
        layer.keys = _make_tensor(1, 2, 8, 4)
        layer.values = _make_tensor(1, 2, 8, 4)
        layer.dtype = layer.keys.dtype
        layer.device = layer.keys.device

        children, _ = _flatten_static_layer(layer)
        keyed, _ = _flatten_with_keys_static_layer(layer)

        self.assertEqual(len(keyed), len(children))
        for (_, tensor_keyed), tensor_plain in zip(keyed, children):
            torch.testing.assert_close(tensor_keyed, tensor_plain)


if __name__ == "__main__":
    unittest.main()
