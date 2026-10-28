# Frozen forecast

Frozen before installing dependencies or executing released code.

- Expected score: **5 / 10**
- Plausible range: **3–7 / 10**

## Rationale

The official repository contains a pinned requirements file, complete Python
training and expert-inference paths, causal-distance utilities, and a
four-megabyte synthetic example dataset. This should permit independent
method, proposition, and toy-data checks without GPU hardware.

The UK Biobank cohort is not released and cannot be reconstructed from the
repository. No checkpoints, metric outputs, or tests are cached. Static
inspection also shows that the released adaptive prior normalizes each
non-zero `W z` update to `adaptation_scale * ||theta||`, whereas the paper's
theory assumes an unnormalized linear mean `theta + W z`. That mismatch may
reduce the method and theorem scores.

## Expected claim scores

| Claim | Forecast |
|---|---:|
| Embedding-conditional prior | 1 |
| Prior-risk proposition | 2 |
| Negative-transfer theorem | 1 |
| Synthetic evidence | 1 |
| Clinical evidence | 0 |

