"""
Key-Value (KV) Cache Manager for Speculative Decoding.
Supports arbitrary batch/layer dimensions, incremental appending,
and fast rollback/truncation upon draft token rejection.
"""

from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class LayerKVCache:
    """Stores key and value state for a single transformer layer."""

    def __init__(self, key: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
                 value: Optional[Union[np.ndarray, "torch.Tensor"]] = None):
        self.key = key
        self.value = value

    @property
    def seq_len(self) -> int:
        if self.key is None:
            return 0
        # Assume shape (batch_size, num_heads, seq_len, head_dim) or (batch_size, seq_len, dim)
        # Sequence dimension is -2
        return self.key.shape[-2]

    def append(self, k: Union[np.ndarray, "torch.Tensor"], v: Union[np.ndarray, "torch.Tensor"]) -> None:
        """Concatenate new key and value tensors along the sequence dimension (-2)."""
        if self.key is None or self.value is None:
            self.key = k
            self.value = v
            return

        if HAS_TORCH and isinstance(k, torch.Tensor) and isinstance(self.key, torch.Tensor):
            self.key = torch.cat([self.key, k], dim=-2)
            self.value = torch.cat([self.value, v], dim=-2)
        else:
            k_np = np.asarray(k)
            v_np = np.asarray(v)
            self.key = np.concatenate([np.asarray(self.key), k_np], axis=-2)
            self.value = np.concatenate([np.asarray(self.value), v_np], axis=-2)

    def truncate(self, target_len: int) -> None:
        """Roll back key/value caches to target_len along sequence dimension (-2)."""
        if self.key is None or self.seq_len <= target_len:
            return

        if target_len <= 0:
            self.key = None
            self.value = None
            return

        if HAS_TORCH and isinstance(self.key, torch.Tensor):
            self.key = self.key[..., :target_len, :]
            self.value = self.value[..., :target_len, :]
        else:
            self.key = self.key[..., :target_len, :]
            self.value = self.value[..., :target_len, :]

    def clone(self) -> "LayerKVCache":
        """Create a deep copy of the layer cache."""
        if self.key is None:
            return LayerKVCache(None, None)

        if HAS_TORCH and isinstance(self.key, torch.Tensor):
            return LayerKVCache(self.key.clone(), self.value.clone())
        return LayerKVCache(np.copy(self.key), np.copy(self.value))


class KVCache:
    """
    Multi-layer transformer Key-Value Cache manager.
    Coordinates layer-wise cache allocation, inspection, and rollback.
    """

    def __init__(self, num_layers: int):
        if num_layers <= 0:
            raise ValueError("num_layers must be greater than 0")
        self.num_layers = num_layers
        self.layers: List[LayerKVCache] = [LayerKVCache() for _ in range(num_layers)]

    @property
    def seq_len(self) -> int:
        """Returns sequence length of the first initialized layer, or 0."""
        for layer in self.layers:
            if layer.key is not None:
                return layer.seq_len
        return 0

    def update(self, layer_idx: int, key: Union[np.ndarray, "torch.Tensor"],
               value: Union[np.ndarray, "torch.Tensor"]) -> None:
        """Append key/value states to a specific layer."""
        if not (0 <= layer_idx < self.num_layers):
            raise IndexError(f"Layer index {layer_idx} out of bounds [0, {self.num_layers})")
        self.layers[layer_idx].append(key, value)

    def truncate(self, target_len: int) -> None:
        """Roll back all layer caches to target_len (essential for speculative rejection)."""
        for layer in self.layers:
            layer.truncate(target_len)

    def clone(self) -> "KVCache":
        """Deep copy entire multi-layer cache."""
        new_cache = KVCache(self.num_layers)
        new_cache.layers = [l.clone() for l in self.layers]
        return new_cache

    def clear(self) -> None:
        """Reset all layer states."""
        self.layers = [LayerKVCache() for _ in range(self.num_layers)]

    def __repr__(self) -> str:
        return f"KVCache(num_layers={self.num_layers}, seq_len={self.seq_len})"

