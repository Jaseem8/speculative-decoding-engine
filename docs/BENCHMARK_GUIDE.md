# Benchmark Interpretation Guide & FAQ 📊

## 1. Key Metrics Explained

| Metric | Symbol | Description | Ideal Range |
| :--- | :---: | :--- | :--- |
| **Acceptance Rate** | $\alpha$ | Fraction of draft tokens accepted by the target model | $70\% - 90\%$ |
| **Lookahead Depth** | $\gamma$ | Number of speculative tokens proposed per cycle | $3 - 6$ |
| **Speedup Factor** | $S$ | Ratio of baseline time to speculative time ($T_{\text{base}} / T_{\text{spec}}$) | $1.5\times - 2.8\times$ |

## 2. Frequently Asked Questions

### Why does speedup plateau at high $\gamma$?
Beyond $\gamma = 6$, draft model sequential latency begins to exceed the parallel verification benefit, especially if the acceptance probability $\alpha$ drops.

### Does speculative decoding alter model output distribution?
No. Leviathan et al. mathematically proved that the output distribution is **identically distributed** to sampling directly from the target model with zero degradation in perplexity.
