"""
Unit tests for BenchmarkHarness.
"""

from speculative_engine.benchmark import BenchmarkHarness
from speculative_engine.models.mock_model import MockInferenceModel


def test_benchmark_harness_run_sweep():
    target = MockInferenceModel(vocab_size=100, num_layers=3, hidden_dim=32, seed=1)
    draft = MockInferenceModel(vocab_size=100, num_layers=2, hidden_dim=32, seed=1)

    harness = BenchmarkHarness(target, draft)
    results = harness.run_sweep(prompt_ids=[1, 2], max_new_tokens=10, gamma_values=[2, 3])

    assert len(results) == 2
    assert results[0].gamma == 2
    assert results[1].gamma == 3
    assert results[0].speedup_ratio > 0.0

    table_str = harness.format_table(results)
    assert "Gamma (γ)" in table_str
    assert "Speedup Factor" in table_str

