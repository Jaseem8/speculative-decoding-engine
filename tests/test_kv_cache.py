"""
Unit tests for Key-Value (KV) Cache operations and rollback.
"""

import numpy as np
import pytest
from speculative_engine.kv_cache import KVCache, LayerKVCache


def test_layer_kv_cache_append_and_truncate():
    layer = LayerKVCache()
    assert layer.seq_len == 0

    # Append first chunk
    k1 = np.ones((1, 2, 4, 16))  # seq_len = 4
    v1 = np.ones((1, 2, 4, 16))
    layer.append(k1, v1)
    assert layer.seq_len == 4

    # Append second chunk
    k2 = np.ones((1, 2, 3, 16))  # seq_len = 3
    v2 = np.ones((1, 2, 3, 16))
    layer.append(k2, v2)
    assert layer.seq_len == 7

    # Truncate back to 4
    layer.truncate(4)
    assert layer.seq_len == 4
    assert layer.key.shape == (1, 2, 4, 16)

    # Truncate to 0 clears
    layer.truncate(0)
    assert layer.seq_len == 0
    assert layer.key is None


def test_multi_layer_kv_cache_coordination():
    cache = KVCache(num_layers=4)
    assert cache.num_layers == 4
    assert cache.seq_len == 0

    k = np.zeros((1, 1, 5, 8))
    v = np.zeros((1, 1, 5, 8))

    for i in range(4):
        cache.update(i, k, v)

    assert cache.seq_len == 5

    # Test cloning
    cloned = cache.clone()
    assert cloned.seq_len == 5

    # Truncate original
    cache.truncate(3)
    assert cache.seq_len == 3
    # Cloned copy remains unchanged
    assert cloned.seq_len == 5


def test_kv_cache_invalid_layer_index():
    cache = KVCache(num_layers=2)
    with pytest.raises(IndexError):
        cache.update(3, np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1, 1)))

