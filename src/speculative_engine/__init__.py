"""
Speculative Decoding Inference Engine
=====================================
High-performance speculative decoding implementation with provable
target distribution preservation (Leviathan et al., 2023; Chen et al., 2023).
"""

from .sampler import RejectionSampler, sample_from_logits
from .kv_cache import KVCache, LayerKVCache
from .engine import SpeculativeDecoder, GenerationResult
from .benchmark import BenchmarkHarness, BenchmarkResult

__version__ = "0.1.0"
__all__ = [
    "RejectionSampler",
    "sample_from_logits",
    "KVCache",
    "LayerKVCache",
    "SpeculativeDecoder",
    "GenerationResult",
    "BenchmarkHarness",
    "BenchmarkResult",
]

