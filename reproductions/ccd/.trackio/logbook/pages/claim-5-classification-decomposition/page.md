# Claim 5: Classification decomposition


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_ccd_v2_c5", "created_at": "2026-07-29T09:00:00+00:00", "title": "Claim 5: Classification decomposition"}
-->
**Anchored claim.** Table 4 reports divergent directional KL-ERT components
across conformal prediction methods on image-classification datasets.

**Verdict: VERIFIED BY RELEASED-OUTPUT REPLAY.**

This audit reads all 80 rows of the official released classification CSV:
MNIST, FashionMNIST, CIFAR10, and CIFAR100; cumulative and likelihood
conformal methods; 10 repetitions per dataset/method. It does not claim to
retrain the image classifiers.

The row-level additive identity between total KL-ERT and its two directional
components holds with maximum residual `8.94e-8`. All `8/8` dataset/method
groups have an absolute difference above `0.01` between the directional
means. Examples:

| Dataset / method | Direction 1 | Direction 2 | Absolute gap |
| --- | ---: | ---: | ---: |
| CIFAR100 / cumulative | `0.0160` | `0.1745` | `0.1585` |
| CIFAR100 / likelihood | `0.0852` | `0.3234` | `0.2382` |
| MNIST / cumulative | `-0.1594` | `-0.0569` | `0.1025` |

The signed values are preserved exactly as released. The claim concerns the
different directional components, which the complete output panel confirms.

```powershell
python audit_released_results.py
```
