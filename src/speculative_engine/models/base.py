"""
Abstract Base Inference Model Interface.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
from ..kv_cache import KVCache


class BaseInferenceModel(ABC):
    """Abstract interface for autoregressive causal language models."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns vocabulary size."""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Number of transformer layers."""
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: Union[np.ndarray, list],
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, Optional[KVCache]]:
        """
        Run forward pass over input_ids.

        Args:
            input_ids: Token IDs of shape (seq_len,) or (1, seq_len).
            kv_cache: Existing KVCache or None.
            use_cache: Whether to use and update the KV cache.

        Returns:
            Tuple of:
                logits: Array of shape (seq_len, vocab_size)
                kv_cache: Updated KVCache
        """
        pass

