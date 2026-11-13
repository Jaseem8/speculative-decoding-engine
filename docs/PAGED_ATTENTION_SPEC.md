# PagedAttention & Block-Based KV-Cache Specification 📑

## 1. Motivation
In standard speculative decoding, Key-Value cache tensors are dynamically resized and truncated when speculative draft tokens are rejected. For large batch sizes and high speculation lookahead $\gamma$, contiguous reallocation leads to memory fragmentation.

## 2. Block Table Design
- **Logical Blocks**: Sequential tokens grouped into fixed-size chunks (e.g. 16 tokens).
- **Physical Block Pool**: Pre-allocated memory slots in high-bandwidth memory.
- **Rollback Complexity**: Upon draft rejection at token $k$, physical blocks beyond $\lceil (L_{\text{prefix}} + k)/16 \rceil$ are immediately released back to the free block pool in $O(1)$ time without tensor copies.

## 3. Benchmarked Metrics
- Peak Memory Reduction: ~30-40% on long sequence generation.
- Rollback Overhead: Under 0.05 ms per speculative cycle.
