"""
CLI Interface for Speculative Decoding Engine.
"""

import argparse
import sys
from typing import List
from .models.mock_model import MockInferenceModel
from .engine import SpeculativeDecoder, standard_generate
from .benchmark import BenchmarkHarness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speculative Decoding Engine & Inference Acceleration Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gamma", type=int, default=4, help="Speculative lookahead depth (gamma)")
    parser.add_argument("--tokens", type=int, default=60, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--vocab-size", type=int, default=1000, help="Vocabulary size for synthetic models")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-gamma comparative benchmark sweep")
    parser.add_argument("--target-latency-ms", type=float, default=15.0, help="Simulated target model latency per step in ms")
    parser.add_argument("--draft-latency-ms", type=float, default=2.5, help="Simulated draft model latency per step in ms")

    args = parser.parse_args()

    print("\n🚀 SPECULATIVE DECODING ENGINE")
    print("=" * 65)
    print(f"Vocab Size:         {args.vocab_size}")
    print(f"Target Step Delay:  {args.target_latency_ms} ms")
    print(f"Draft Step Delay:   {args.draft_latency_ms} ms")
    print(f"Sampling Temp:      {args.temperature}")
    print(f"Lookahead Gamma:    {args.gamma}")
    print("=" * 65)

    # Initialize models
    target_model = MockInferenceModel(
        vocab_size=args.vocab_size,
        num_layers=12,
        hidden_dim=128,
        seed=100,
        latency_per_step_ms=args.target_latency_ms,
        divergence_noise=0.0,
    )
    # Draft model has slight divergence noise simulating draft model variance
    draft_model = MockInferenceModel(
        vocab_size=args.vocab_size,
        num_layers=4,
        hidden_dim=128,
        seed=100,
        latency_per_step_ms=args.draft_latency_ms,
        divergence_noise=0.15,
    )

    prompt = [12, 45, 99, 102]

    if args.benchmark:
        print("\n⏳ Running benchmark sweep across γ in [2, 3, 4, 5, 6]...\n")
        harness = BenchmarkHarness(target_model, draft_model, temperature=args.temperature)
        results = harness.run_sweep(prompt, max_new_tokens=args.tokens)
        print(harness.format_table(results))
        print("\n✔ Benchmark complete.")
        return

    # Single comparison run
    print("\n1. Running Baseline Autoregressive (Target Only)...")
    base_res = standard_generate(target_model, prompt, max_new_tokens=args.tokens, temperature=args.temperature)
    print(f"   • Generated: {base_res.total_tokens_generated} tokens")
    print(f"   • Time:      {base_res.elapsed_time_sec:.3f} s")
    print(f"   • Throughput:{base_res.tokens_per_second:.1f} tokens/sec")
    print(f"   • Forward:   {base_res.target_forward_passes} target calls")

    print(f"\n2. Running Speculative Decoding (γ={args.gamma})...")
    decoder = SpeculativeDecoder(target_model, draft_model, gamma=args.gamma, temperature=args.temperature)
    spec_res = decoder.generate(prompt, max_new_tokens=args.tokens)
    print(f"   • Generated:       {spec_res.total_tokens_generated} tokens")
    print(f"   • Time:            {spec_res.elapsed_time_sec:.3f} s")
    print(f"   • Throughput:      {spec_res.tokens_per_second:.1f} tokens/sec")
    print(f"   • Acceptance Rate: {spec_res.acceptance_rate * 100:.1f}% ({spec_res.draft_tokens_accepted}/{spec_res.draft_tokens_proposed})")
    print(f"   • Forward Calls:   {spec_res.target_forward_passes} target calls (vs {base_res.target_forward_passes} baseline)")

    speedup = base_res.elapsed_time_sec / spec_res.elapsed_time_sec if spec_res.elapsed_time_sec > 0 else 1.0
    print(f"\n✨ Effective Acceleration Speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()

