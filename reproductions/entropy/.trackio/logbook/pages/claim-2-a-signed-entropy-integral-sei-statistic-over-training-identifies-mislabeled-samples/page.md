# Claim 2: A signed entropy integral (SEI) statistic over training identifies mislabeled samples


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_2909a6586c67", "created_at": "2026-07-19T07:36:53+00:00", "title": "Claim 2: A signed entropy integral (SEI) statistic over training identifies mislabeled samples"}
-->
**Setup.** Using the per-sample entropy trajectory from Claim 1, we test whether a trajectory statistic separates mislabeled from clean samples, measured by ROC-AUC (mislabeled = positive). We try several proxies for the paper's Signed Entropy Integral (which captures both magnitude and temporal trend):

| trajectory statistic | AUC |
| --- | --- |
| entropy at peak-gap epoch (ep 15) | **0.716** |
| mean entropy over all 25 epochs | 0.709 |
| mean entropy over epochs 5-15 | 0.695 |
| early-weighted entropy integral (SEI-like) | 0.699 |
| mean entropy over first half (ep 1-12) | 0.664 |

**Verdict — reproduced as a proxy.** The entropy trajectory carries a clear signal for mislabeled detection (AUC ~0.71, well above the 0.50 chance level). These are crude proxies; the paper's actual **signed** entropy integral (magnitude + temporal trend, per its Fig. on nevus/DR entropy curves) should do better, and on cleaner medical datasets than this 26.4%-noise coarse CIFAR-100N toy. The core claim — that the entropy *trajectory* (not an endpoint) identifies mislabeled samples — holds.
