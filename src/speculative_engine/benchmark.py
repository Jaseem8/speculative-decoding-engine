"""
Benchmarking Harness for Speculative vs Autoregressive Decoding.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import numpy as np

from .engine import SpeculativeDecoder, standard_generate, GenerationResult
from .models.base import BaseInferenceModel


@dataclass
class BenchmarkResult:
    gamma: int
    baseline_tokens_per_sec: float
    speculative_tokens_per_sec: float
    speedup_ratio: float
    acceptance_rate: float
    baseline_latency_sec: float
    speculative_latency_sec: float
    total_tokens: int


class BenchmarkHarness:
    """
    Evaluates speculative decoding acceleration profiles.
    """

    def __init__(
        self,
        target_model: BaseInferenceModel,
        draft_model: BaseInferenceModel,
        temperature: float = 1.0,
    ):
        self.target_model = target_model
        self.draft_model = draft_model
        self.temperature = temperature

    def run_sweep(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 50,
        gamma_values: Optional[List[int]] = None,
        warmup_runs: int = 1,
    ) -> List[BenchmarkResult]:
        """
        Run comparative benchmark across multiple lookahead gamma values.
        """
        if gamma_values is None:
            gamma_values = [2, 3, 4, 5, 6]

        # Warmup
        for _ in range(warmup_runs):
            _ = standard_generate(
                self.target_model, prompt_ids, max_new_tokens=5, temperature=self.temperature
            )

        # Baseline Autoregressive Run
        baseline_res = standard_generate(
            self.target_model, prompt_ids, max_new_tokens=max_new_tokens, temperature=self.temperature
        )

        results: List[BenchmarkResult] = []

        for gamma in gamma_values:
            decoder = SpeculativeDecoder(
                target_model=self.target_model,
                draft_model=self.draft_model,
                gamma=gamma,
                temperature=self.temperature,
            )

            # Warmup
            for _ in range(warmup_runs):
                _ = decoder.generate(prompt_ids, max_new_tokens=5)

            spec_res = decoder.generate(prompt_ids, max_new_tokens=max_new_tokens)

            speedup = (
                baseline_res.elapsed_time_sec / spec_res.elapsed_time_sec
                if spec_res.elapsed_time_sec > 0
                else 1.0
            )

            results.append(
                BenchmarkResult(
                    gamma=gamma,
                    baseline_tokens_per_sec=baseline_res.tokens_per_second,
                    speculative_tokens_per_sec=spec_res.tokens_per_second,
                    speedup_ratio=speedup,
                    acceptance_rate=spec_res.acceptance_rate,
                    baseline_latency_sec=baseline_res.elapsed_time_sec,
                    speculative_latency_sec=spec_res.elapsed_time_sec,
                    total_tokens=spec_res.total_tokens_generated,
                )
            )

        return results

    @staticmethod
    def format_table(results: List[BenchmarkResult]) -> str:
        """Format benchmark outputs into an ANSI/Markdown comparison table."""
        header = "| Gamma (γ) | Acceptance Rate | Baseline (tok/s) | Speculative (tok/s) | Speedup Factor |"
        sep = "| :---: | :---: | :---: | :---: | :---: |"
        rows = [header, sep]
        for r in results:
            rows.append(
                f"| γ = {r.gamma} | {r.acceptance_rate * 100:5.1f}% | "
                f"{r.baseline_tokens_per_sec:6.1f} | {r.speculative_tokens_per_sec:6.1f} | "
                f"**{r.speedup_ratio:5.2f}x** |"
            )
        return "\n".join(rows)

