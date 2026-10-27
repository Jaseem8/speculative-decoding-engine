"""
Mathematical Sampling and Modified Rejection Sampling Engine
=============================================================
Implements the exact rejection sampling algorithm from:
Leviathan et al. (2023) "Fast Inference from Transformers via Speculative Decoding"
Chen et al. (2023) "Accelerating Large Language Model Decoding with Speculative Sampling"
"""

from typing import Optional, Tuple
import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Numerically stable Softmax with temperature scaling.
    logits: 1D array of shape (vocab_size,)
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")

    scaled = logits / temperature
    # Shift by max for numerical stability
    shifted = scaled - np.max(scaled)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals)
    return probs


def sample_from_logits(logits: np.ndarray, temperature: float = 1.0,
                       top_k: Optional[int] = None, top_p: Optional[float] = None,
                       rng: Optional[np.random.Generator] = None) -> int:
    """
    Sample token from logits with temperature, top-k, and nucleus (top-p) filtering.
    """
    if rng is None:
        rng = np.random.default_rng()

    probs = softmax(logits, temperature=temperature)

    if top_k is not None and top_k > 0 and top_k < len(probs):
        indices_to_remove = np.argsort(probs)[:-top_k]
        probs[indices_to_remove] = 0.0
        probs = probs / np.sum(probs)

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to keep also the first token above the threshold
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        probs = probs / np.sum(probs)

    token_id = rng.choice(len(probs), p=probs)
    return int(token_id)


class RejectionSampler:
    """
    Modified Rejection Sampler preserving exact target model output distribution.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def sample_token(self, probs: np.ndarray) -> int:
        """Sample from a normalized probability vector."""
        probs = np.asarray(probs, dtype=np.float64)
        sum_p = np.sum(probs)
        if sum_p <= 0 or not np.isfinite(sum_p):
            # Uniform fallback if probabilities are corrupted
            return int(self.rng.choice(len(probs)))
        probs = probs / sum_p
        return int(self.rng.choice(len(probs), p=probs))

    def evaluate_draft_token(
        self,
        draft_token: int,
        target_probs: np.ndarray,
        draft_probs: np.ndarray,
    ) -> Tuple[bool, int, float]:
        """
        Evaluate candidate draft token against target distribution.

        Returns:
            (is_accepted, token_to_emit, acceptance_probability)

        Algorithm:
            alpha = min(1, p(x) / q(x))
            Draw r ~ Uniform(0, 1)
            If r < alpha:
                emit draft_token
            Else:
                emit replacement ~ max(0, p(x) - q(x)) / sum(...)
        """
        target_probs = np.asarray(target_probs, dtype=np.float64)
        draft_probs = np.asarray(draft_probs, dtype=np.float64)

        p_val = float(target_probs[draft_token])
        q_val = float(draft_probs[draft_token])

        if p_val <= 0.0:
            acceptance_prob = 0.0
        elif q_val <= 0.0:
            acceptance_prob = 1.0
        else:
            acceptance_prob = min(1.0, p_val / q_val)

        r = float(self.rng.uniform(0.0, 1.0))

        if r < acceptance_prob:
            # Accepted! Target matches draft
            return True, draft_token, acceptance_prob

        # Rejected. Compute normalized residual distribution p'(x) = max(0, p(x) - q(x))
        residual = np.maximum(0.0, target_probs - draft_probs)
        norm_factor = np.sum(residual)

        if norm_factor > 1e-9:
            norm_residual = residual / norm_factor
            replacement_token = self.sample_token(norm_residual)
        else:
            # Fallback to direct target model sampling if residual mass is vanishingly small
            replacement_token = self.sample_token(target_probs)

        return False, replacement_token, acceptance_prob
