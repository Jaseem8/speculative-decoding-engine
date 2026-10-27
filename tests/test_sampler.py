"""
Unit tests for RejectionSampler and probability mathematics.
"""

import numpy as np
import pytest
from speculative_engine.sampler import RejectionSampler, softmax, sample_from_logits


def test_softmax_numerical_stability():
    # Extremely large logits that would overflow naive exp
    large_logits = np.array([1000.0, 1005.0, 995.0])
    probs = softmax(large_logits, temperature=1.0)
    assert np.all(np.isfinite(probs))
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[1] > probs[0] > probs[2]


def test_softmax_invalid_temperature():
    with pytest.raises(ValueError):
        softmax(np.array([1.0, 2.0]), temperature=0.0)
    with pytest.raises(ValueError):
        softmax(np.array([1.0, 2.0]), temperature=-1.0)


def test_rejection_sampling_deterministic_acceptance():
    sampler = RejectionSampler(rng=np.random.default_rng(42))

    # Case 1: Target probability is equal or higher than draft -> alpha = 1.0 (always accept)
    target_p = np.array([0.7, 0.3])
    draft_p = np.array([0.4, 0.6])

    # Draft token 0: p(0)=0.7, q(0)=0.4 -> alpha = min(1, 0.7/0.4) = 1.0
    is_accepted, token, alpha = sampler.evaluate_draft_token(
        draft_token=0, target_probs=target_p, draft_probs=draft_p
    )
    assert is_accepted is True
    assert token == 0
    assert alpha == 1.0


def test_rejection_sampling_rejection_and_residual_sampling():
    sampler = RejectionSampler(rng=np.random.default_rng(123))

    # Case 2: Draft token 1 has high draft probability but zero target probability
    # target_p has all mass on 0; draft has mass on 1
    target_p = np.array([1.0, 0.0])
    draft_p = np.array([0.0, 1.0])

    # Token 1: p(1)=0.0, q(1)=1.0 -> alpha = 0.0 (always reject)
    is_accepted, token, alpha = sampler.evaluate_draft_token(
        draft_token=1, target_probs=target_p, draft_probs=draft_p
    )
    assert is_accepted is False
    # Recovery token must be sampled from residual mass (which is token 0)
    assert token == 0
    assert alpha == 0.0


def test_sample_from_logits_top_k():
    logits = np.array([10.0, 8.0, 5.0, 1.0, 0.0])
    rng = np.random.default_rng(42)
    # top_k=2 limits sampling strictly to indices 0 and 1
    tokens = [sample_from_logits(logits, top_k=2, rng=rng) for _ in range(50)]
    assert all(t in (0, 1) for t in tokens)

