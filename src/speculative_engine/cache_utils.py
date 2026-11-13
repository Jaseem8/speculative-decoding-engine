"""
Vector Normalization and Fast Dot-Product Utilities.
"""

import numpy as np
from typing import Dict, Tuple

class NormalizedEmbeddingCache:
    """Caches unit-normalized embeddings to bypass repeated norm computations."""

    def __init__(self):
        self._cache: Dict[int, np.ndarray] = {}

    def get_or_normalize(self, token_id: int, vector: np.ndarray) -> np.ndarray:
        if token_id in self._cache:
            return self._cache[token_id]

        norm = np.linalg.norm(vector)
        normalized = vector / norm if norm > 1e-9 else vector
        self._cache[token_id] = normalized
        return normalized

    def clear(self) -> None:
        self._cache.clear()
