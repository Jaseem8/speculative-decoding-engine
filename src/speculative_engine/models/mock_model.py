"""
Synthetic Inference Model for Deterministic Testing and Benchmarking.
Allows testing speculative decoding with controllable alignment rates and latencies.
"""

import time
from typing import Optional, Tuple, Union
import numpy as np
from .base import BaseInferenceModel
from ..kv_cache import KVCache


class MockInferenceModel(BaseInferenceModel):
    """
    Mock autoregressive model for testing speculative decoding dynamics.
    Produces deterministic token probability distributions based on a seeded
    Markov / embedding transition matrix.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        num_layers: int = 4,
        hidden_dim: int = 64,
        seed: int = 42,
        latency_per_step_ms: float = 0.0,
        divergence_noise: float = 0.0,
    ):
        self._vocab_size = vocab_size
        self._num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.latency_per_step_ms = latency_per_step_ms
        self.divergence_noise = divergence_noise

        rng = np.random.default_rng(seed)
        # Random embedding table
        self.embeddings = rng.standard_normal((vocab_size, hidden_dim))
        # Projection weight matrix
        self.proj = rng.standard_normal((hidden_dim, vocab_size)) / np.sqrt(hidden_dim)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def forward(
        self,
        input_ids: Union[np.ndarray, list],
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = True,
    ) -> Tuple[np.ndarray, Optional[KVCache]]:
        """
        Simulate forward pass:
        If KV cache is provided and seq_len > 0, evaluate only new tokens.
        """
        if self.latency_per_step_ms > 0:
            time.sleep(self.latency_per_step_ms / 1000.0)

        input_arr = np.atleast_1d(np.asarray(input_ids, dtype=np.int64))

        if kv_cache is not None and use_cache and kv_cache.seq_len > 0:
            # Incremental pass on newly added tokens
            cached_len = kv_cache.seq_len
            if len(input_arr) > cached_len:
                tokens_to_process = input_arr[cached_len:]
            else:
                tokens_to_process = input_arr[-1:]
        else:
            tokens_to_process = input_arr

        if kv_cache is None and use_cache:
            kv_cache = KVCache(self._num_layers)

        # Compute synthetic logits via embedding lookup & projection
        seq_len = len(tokens_to_process)
        embeds = self.embeddings[tokens_to_process % self._vocab_size]  # (seq_len, hidden_dim)
        raw_logits = embeds @ self.proj  # (seq_len, vocab_size)

        if self.divergence_noise > 0.0:
            noise = np.random.normal(0.0, self.divergence_noise, size=raw_logits.shape)
            raw_logits += noise

        # Update synthetic KV cache states
        if use_cache and kv_cache is not None:
            for l_idx in range(self._num_layers):
                # Shape (1, 1, seq_len, hidden_dim)
                fake_k = embeds[np.newaxis, np.newaxis, :, :]
                fake_v = embeds[np.newaxis, np.newaxis, :, :]
                kv_cache.update(l_idx, fake_k, fake_v)

        return raw_logits, kv_cache

