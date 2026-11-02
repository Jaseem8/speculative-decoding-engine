"""
Speculative Decoding Orchestrator Engine.
Coordinates the iterative Draft-Verify-Rollback generation cycle.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np

from .kv_cache import KVCache
from .models.base import BaseInferenceModel
from .sampler import RejectionSampler, softmax, sample_from_logits


@dataclass
class GenerationResult:
    """Output metrics and token sequence from generation."""
    token_ids: List[int]
    total_tokens_generated: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    acceptance_rate: float
    target_forward_passes: int
    draft_forward_passes: int
    elapsed_time_sec: float
    tokens_per_second: float


class SpeculativeDecoder:
    """
    Speculative Decoding Engine.
    Executes accelerated inference using a fast draft model and a high-capacity target model.
    """

    def __init__(
        self,
        target_model: BaseInferenceModel,
        draft_model: BaseInferenceModel,
        gamma: int = 4,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ):
        if gamma < 1:
            raise ValueError(f"Lookahead gamma must be >= 1, got {gamma}")
        if target_model.vocab_size != draft_model.vocab_size:
            raise ValueError(
                f"Vocab size mismatch: target={target_model.vocab_size}, draft={draft_model.vocab_size}"
            )

        self.target_model = target_model
        self.draft_model = draft_model
        self.gamma = gamma
        self.temperature = temperature
        self.sampler = RejectionSampler(rng=np.random.default_rng(seed))

    def generate(
        self,
        prompt_ids: Union[List[int], np.ndarray],
        max_new_tokens: int = 50,
        stop_token_id: Optional[int] = None,
    ) -> GenerationResult:
        """
        Generate tokens using speculative decoding.
        """
        start_time = time.perf_counter()
        current_tokens = list(prompt_ids)
        total_generated = 0
        draft_proposed = 0
        draft_accepted = 0
        target_calls = 0
        draft_calls = 0

        # Initialize KV Caches with prefill pass
        target_cache = KVCache(self.target_model.num_layers)
        draft_cache = KVCache(self.draft_model.num_layers)

        # Prefill pass on prompt
        target_logits, target_cache = self.target_model.forward(current_tokens, target_cache)
        draft_logits, draft_cache = self.draft_model.forward(current_tokens, draft_cache)
        target_calls += 1
        draft_calls += 1

        while total_generated < max_new_tokens:
            # ---------------------------------------------------------------
            # 1. Draft Phase: Speculatively propose gamma tokens
            # ---------------------------------------------------------------
            draft_tokens: List[int] = []
            draft_probs_list: List[np.ndarray] = []
            draft_seq = list(current_tokens)

            # Last logits from previous step
            last_draft_logits = draft_logits[-1]

            for _ in range(self.gamma):
                draft_probs = softmax(last_draft_logits, temperature=self.temperature)
                tok = self.sampler.sample_token(draft_probs)

                draft_tokens.append(tok)
                draft_probs_list.append(draft_probs)
                draft_seq.append(tok)

                # Incremental forward pass on draft model
                draft_step_logits, draft_cache = self.draft_model.forward(draft_seq, draft_cache)
                draft_calls += 1
                last_draft_logits = draft_step_logits[-1]

            draft_proposed += len(draft_tokens)

            # ---------------------------------------------------------------
            # 2. Target Verification Phase: Parallel forward pass on candidate tokens
            # ---------------------------------------------------------------
            target_candidates = current_tokens + draft_tokens
            target_multi_logits, target_cache = self.target_model.forward(
                target_candidates, target_cache
            )
            target_calls += 1

            num_draft = len(draft_tokens)
            # In causal autoregressive models:
            # - draft_tokens[0] is evaluated against target_logits[-1] (the distribution at the prefix)
            # - draft_tokens[1..gamma-1] are evaluated against target_multi_logits[0..gamma-2]
            # - bonus token (if all accepted) is evaluated against target_multi_logits[-1]
            target_eval_logits = [target_logits[-1]] + [target_multi_logits[i] for i in range(num_draft)]

            # ---------------------------------------------------------------
            # 3. Modified Rejection Sampling & Rollback
            # ---------------------------------------------------------------
            n_accepted = 0
            all_accepted = True
            recovery_token = None

            for i in range(num_draft):
                cand_tok = draft_tokens[i]
                target_p = softmax(target_eval_logits[i], temperature=self.temperature)
                draft_p = draft_probs_list[i]

                is_accepted, emitted_tok, _ = self.sampler.evaluate_draft_token(
                    draft_token=cand_tok,
                    target_probs=target_p,
                    draft_probs=draft_p,
                )

                if is_accepted:
                    current_tokens.append(cand_tok)
                    total_generated += 1
                    n_accepted += 1
                    draft_accepted += 1
                    if stop_token_id is not None and cand_tok == stop_token_id:
                        all_accepted = False
                        break
                    if total_generated >= max_new_tokens:
                        all_accepted = False
                        break
                else:
                    # Token rejected. Accept recovery token and truncate caches
                    current_tokens.append(emitted_tok)
                    total_generated += 1
                    recovery_token = emitted_tok
                    all_accepted = False
                    break

            # ---------------------------------------------------------------
            # 4. Cache Rollback and Bonus Token
            # ---------------------------------------------------------------
            if all_accepted and total_generated < max_new_tokens:
                # Bonus token from target model's final position
                bonus_logits = target_multi_logits[-1]
                bonus_probs = softmax(bonus_logits, temperature=self.temperature)
                bonus_tok = self.sampler.sample_token(bonus_probs)
                current_tokens.append(bonus_tok)
                total_generated += 1

                # Update caches on the single bonus token
                target_logits, target_cache = self.target_model.forward(current_tokens, target_cache)
                draft_logits, draft_cache = self.draft_model.forward(current_tokens, draft_cache)
                target_calls += 1
                draft_calls += 1
            else:
                # Roll back caches to prefix before recovery token, then process recovery token
                valid_prefix_len = len(current_tokens) - 1
                target_cache.truncate(valid_prefix_len)
                draft_cache.truncate(valid_prefix_len)

                # Process recovery token so caches reach len(current_tokens)
                target_logits, target_cache = self.target_model.forward(current_tokens, target_cache)
                draft_logits, draft_cache = self.draft_model.forward(current_tokens, draft_cache)
                target_calls += 1
                draft_calls += 1

            if stop_token_id is not None and current_tokens[-1] == stop_token_id:
                break

        elapsed = time.perf_counter() - start_time
        acceptance_rate = (draft_accepted / draft_proposed) if draft_proposed > 0 else 0.0
        tok_per_sec = total_generated / elapsed if elapsed > 0 else 0.0

        return GenerationResult(
            token_ids=current_tokens,
            total_tokens_generated=total_generated,
            draft_tokens_proposed=draft_proposed,
            draft_tokens_accepted=draft_accepted,
            acceptance_rate=acceptance_rate,
            target_forward_passes=target_calls,
            draft_forward_passes=draft_calls,
            elapsed_time_sec=elapsed,
            tokens_per_second=tok_per_sec,
        )


def standard_generate(
    model: BaseInferenceModel,
    prompt_ids: Union[List[int], np.ndarray],
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    stop_token_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> GenerationResult:
    """
    Standard autoregressive baseline decoding on a single model.
    """
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    tokens = list(prompt_ids)
    cache = KVCache(model.num_layers)
    forward_calls = 0

    logits, cache = model.forward(tokens, cache)
    forward_calls += 1

    for _ in range(max_new_tokens):
        next_logits = logits[-1]
        probs = softmax(next_logits, temperature=temperature)
        next_token = int(rng.choice(len(probs), p=probs))
        tokens.append(next_token)

        if stop_token_id is not None and next_token == stop_token_id:
            break

        logits, cache = model.forward(tokens, cache)
        forward_calls += 1

    elapsed = time.perf_counter() - start_time
    total_gen = len(tokens) - len(prompt_ids)
    tok_per_sec = total_gen / elapsed if elapsed > 0 else 0.0

    return GenerationResult(
        token_ids=tokens,
        total_tokens_generated=total_gen,
        draft_tokens_proposed=0,
        draft_tokens_accepted=0,
        acceptance_rate=0.0,
        target_forward_passes=forward_calls,
        draft_forward_passes=0,
        elapsed_time_sec=elapsed,
        tokens_per_second=tok_per_sec,
    )
