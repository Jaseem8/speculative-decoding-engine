"""
Unit and integration tests for SpeculativeDecoder.
"""

import numpy as np
import pytest
from speculative_engine.engine import SpeculativeDecoder, standard_generate
from speculative_engine.models.mock_model import MockInferenceModel


@pytest.fixture
def test_models():
    target = MockInferenceModel(vocab_size=200, num_layers=4, hidden_dim=32, seed=10)
    draft = MockInferenceModel(vocab_size=200, num_layers=2, hidden_dim=32, seed=10, divergence_noise=0.1)
    return target, draft


def test_speculative_decoder_generates_correct_token_count(test_models):
    target, draft = test_models
    decoder = SpeculativeDecoder(target, draft, gamma=3, seed=42)

    prompt = [5, 12, 19]
    max_tokens = 25
    res = decoder.generate(prompt, max_new_tokens=max_tokens)

    assert res.total_tokens_generated == max_tokens
    assert len(res.token_ids) == len(prompt) + max_tokens
    assert res.draft_tokens_proposed > 0
    assert 0.0 <= res.acceptance_rate <= 1.0


def test_speculative_decoder_stop_token(test_models):
    target, draft = test_models
    decoder = SpeculativeDecoder(target, draft, gamma=2, seed=42)

    prompt = [1, 2]
    stop_id = 99
    res = decoder.generate(prompt, max_new_tokens=10, stop_token_id=stop_id)
    # The generation should stop gracefully
    assert len(res.token_ids) >= len(prompt)


def test_standard_generate_baseline(test_models):
    target, _ = test_models
    prompt = [1, 2, 3]
    res = standard_generate(target, prompt, max_new_tokens=15, seed=42)

    assert res.total_tokens_generated == 15
    assert len(res.token_ids) == len(prompt) + 15
    assert res.target_forward_passes > 0
