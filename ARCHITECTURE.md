# Speculative Decoding: Deep Architecture & Mathematical Rigor 📐

## 1. The Memory-Bandwidth Bottleneck in Autoregressive LLMs

During standard autoregressive inference, generating token $x_t$ requires loading all model weights $W$ from High-Bandwidth Memory (HBM) into SRAM/cache:

$$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes Transferred}} \approx \frac{2 \cdot P}{2 \cdot P} = 1 \text{ FLOP/Byte}$$

Because modern GPUs have hundreds of TFLOPs of compute but only 1–3 TB/s of memory bandwidth, autoregressive generation is heavily **memory-bound**.

However, when evaluating a batch or sequence of $K$ tokens simultaneously, the weights $W$ are loaded **once** and reused across all $K$ tokens:

$$\text{Arithmetic Intensity for } K \text{ tokens} \approx K \text{ FLOPs/Byte}$$

Evaluating $K$ tokens in parallel takes nearly the same wall-clock time as evaluating $1$ token!

---

## 2. Mathematical Proof of Exact Distribution Preservation

Let:
- $p(x) = P(x \mid x_{<t})$ be the target model probability distribution.
- $q(x) = Q(x \mid x_{<t})$ be the draft model probability distribution.

The speculative sampler generates token $X$ according to the following procedure:
1. Sample candidate $\tilde{X} \sim q(x)$.
2. Accept $\tilde{X}$ with probability $\alpha(\tilde{X}) = \min\left(1, \frac{p(\tilde{X})}{q(\tilde{X})}\right)$.
3. If rejected, sample $X \sim p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y) - q(y))}$.

### Theorem:
The marginal distribution of the emitted token $X$ satisfies:
$$P(X = x) = p(x) \quad \forall x$$

### Proof:
$$P(X = x) = P(\text{accepted and } \tilde{X} = x) + P(\text{rejected and recovery is } x)$$

$$P(\text{accepted and } \tilde{X} = x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$

The total rejection probability is:
$$P(\text{rejected}) = 1 - \sum_y \min(q(y), p(y)) = \sum_y (p(y) - \min(q(y), p(y))) = \sum_y \max(0, p(y) - q(y))$$

When rejected, $X$ is sampled from $p'(x)$:
$$P(\text{rejected and recovery is } x) = P(\text{rejected}) \cdot p'(x) = \sum_y \max(0, p(y) - q(y)) \cdot \frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y) - q(y))} = \max(0, p(x) - q(x))$$

Summing both terms:
$$P(X = x) = \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x) \quad \blacksquare$$

Hence, speculative decoding is **strictly unbiased and lossless**: the generated text follows the exact distribution of the target model with zero degradation in perplexity or output quality.

---

## 3. KV-Cache Rollback Lifecycle

When speculative decoding proposes $\gamma$ draft tokens, attention keys and values are computed for all $\gamma$ candidate positions. If rejection occurs at index $k \le \gamma$:
1. Tokens $x_{k+1 \dots \gamma}$ are invalid.
2. The KV-Cache must be sliced at $L_{\text{prefix}} + k - 1$.
3. The newly sampled recovery token $x_k^{\text{corr}}$ is appended to the cache.
4. Total forward step savings: $\gamma - k$ forward passes avoided.

