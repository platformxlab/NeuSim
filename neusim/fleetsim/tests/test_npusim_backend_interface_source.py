"""Tests for FrozenLLMConfig and FrozenDeepSeekConfig hash/equality behavior.

Verifies that the frozen config classes used as lru_cache keys include ALL fields
(especially input_seqlen, output_seqlen, global_batch_size) in hash/equality,
not just chip_name + parallelism degrees.
"""

import unittest
from functools import cache

from pydantic import ValidationError

from neusim.fleetsim.npusim_backend_interface import (
    FrozenDeepSeekConfig,
    FrozenLLMConfig,
)


class TestFrozenLLMConfigHash(unittest.TestCase):
    """Ensure FrozenLLMConfig hashing includes seqlen and batch size fields."""

    def _make_config(self, **overrides) -> FrozenLLMConfig:
        defaults = dict(input_seqlen=1024, output_seqlen=128, global_batch_size=1)
        defaults.update(overrides)
        return FrozenLLMConfig(**defaults)

    def test_different_input_seqlen_different_hash(self):
        c1 = self._make_config(input_seqlen=1024)
        c2 = self._make_config(input_seqlen=4096)
        self.assertNotEqual(hash(c1), hash(c2))
        self.assertNotEqual(c1, c2)

    def test_different_output_seqlen_different_hash(self):
        c1 = self._make_config(output_seqlen=128)
        c2 = self._make_config(output_seqlen=512)
        self.assertNotEqual(hash(c1), hash(c2))
        self.assertNotEqual(c1, c2)

    def test_different_batch_size_different_hash(self):
        c1 = self._make_config(global_batch_size=1, microbatch_size_ici=1)
        c2 = self._make_config(global_batch_size=8, microbatch_size_ici=8)
        self.assertNotEqual(hash(c1), hash(c2))
        self.assertNotEqual(c1, c2)

    def test_same_config_same_hash(self):
        c1 = self._make_config(input_seqlen=2048, output_seqlen=256)
        c2 = self._make_config(input_seqlen=2048, output_seqlen=256)
        self.assertEqual(hash(c1), hash(c2))
        self.assertEqual(c1, c2)

    def test_usable_as_dict_key(self):
        c1 = self._make_config(input_seqlen=1024)
        c2 = self._make_config(input_seqlen=4096)
        d = {c1: "short", c2: "long"}
        self.assertEqual(len(d), 2)
        self.assertEqual(d[c1], "short")
        self.assertEqual(d[c2], "long")

    def test_lru_cache_distinguishes_seqlens(self):
        """Simulates the actual cache usage pattern: different seqlens must get different cache entries."""
        call_count = 0

        @cache
        def cached_fn(config: FrozenLLMConfig) -> int:
            nonlocal call_count
            call_count += 1
            return config.input_seqlen

        c1 = self._make_config(input_seqlen=1024)
        c2 = self._make_config(input_seqlen=4096)
        c3 = self._make_config(input_seqlen=1024)  # same as c1

        r1 = cached_fn(c1)
        r2 = cached_fn(c2)
        r3 = cached_fn(c3)  # should hit cache for c1

        self.assertEqual(r1, 1024)
        self.assertEqual(r2, 4096)
        self.assertEqual(r3, 1024)
        self.assertEqual(call_count, 2, "c3 should reuse c1's cache entry, not trigger a new call")

    def test_frozen_is_immutable(self):
        c = self._make_config()
        with self.assertRaises(ValidationError):
            c.input_seqlen = 9999


class TestFrozenDeepSeekConfigHash(unittest.TestCase):
    """Ensure FrozenDeepSeekConfig has the same correct hash/equality behavior."""

    # DeepSeekConfig has required MLA fields with no defaults
    _DEEPSEEK_REQUIRED = dict(
        d_ff=7168,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
    )

    def _make_config(self, **overrides) -> FrozenDeepSeekConfig:
        defaults = dict(input_seqlen=1024, output_seqlen=128, global_batch_size=1)
        defaults.update(self._DEEPSEEK_REQUIRED)
        defaults.update(overrides)
        return FrozenDeepSeekConfig(**defaults)

    def test_different_input_seqlen_different_hash(self):
        c1 = self._make_config(input_seqlen=1024)
        c2 = self._make_config(input_seqlen=4096)
        self.assertNotEqual(hash(c1), hash(c2))
        self.assertNotEqual(c1, c2)

    def test_same_config_same_hash(self):
        c1 = self._make_config(input_seqlen=2048)
        c2 = self._make_config(input_seqlen=2048)
        self.assertEqual(hash(c1), hash(c2))
        self.assertEqual(c1, c2)

    def test_lru_cache_distinguishes_seqlens(self):
        call_count = 0

        @cache
        def cached_fn(config: FrozenDeepSeekConfig) -> int:
            nonlocal call_count
            call_count += 1
            return config.input_seqlen

        c1 = self._make_config(input_seqlen=1024)
        c2 = self._make_config(input_seqlen=4096)

        r1 = cached_fn(c1)
        r2 = cached_fn(c2)

        self.assertEqual(r1, 1024)
        self.assertEqual(r2, 4096)
        self.assertEqual(call_count, 2)


if __name__ == "__main__":
    unittest.main()
