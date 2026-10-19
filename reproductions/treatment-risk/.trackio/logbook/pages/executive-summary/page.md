# Executive summary


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_804366d281c9", "created_at": "2026-07-28T16:28:29+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-28T16:28:29+00:00"}
-->
This independent CPU reproduction supports two claims and finds a repairable
defect in the proof of a third. Lemma 4.1 passed 268,800 exact finite-state
risk checks, including equality at `Gamma=1`, while controls with underestimated
`Gamma` violated the bounds. The main-text observational Section 5.1 protocol
was rerun at paper scale (`1000` datasets per arm): every valid-`Gamma` arm
controlled all eight tolerances, while the invalid-`Gamma` arm controlled only
five. Theorem
4.3's Eq. (18) is invalid for a learned policy, but an ordered finite-grid
first-unsafe argument repairs the certificate under explicit extra assumptions.

## Scope & cost

| Item | Value |
| --- | --- |
| GPU / compute | None; Windows CPU only |
| Main run | `4000` learned policies; canonical run `237.3 s`, fresh preflight rerun `530.3 s` |
| Theory audits | Exact finite arithmetic plus exhaustive Boolean/binomial enumeration |
| Prepared verdicts | C1 `VERIFIED`; C2 `PARTIAL`; C3 `VERIFIED`; C4/C5 `NOT ATTEMPTED` |
| Data access | Synthetic data generated from Eqs. (21)-(24); no patient records used |
| Source | [arXiv 2505.08378](https://arxiv.org/abs/2505.08378), [OpenReview 9kWVQtpvJi](https://openreview.net/forum?id=9kWVQtpvJi), [HF paper page](https://huggingface.co/papers/2505.08378) |
| Draft Space | [kondratevakate/repro-learning-treatment-allocations-with-risk-control-under-partial-identifiability](https://huggingface.co/spaces/kondratevakate/repro-learning-treatment-allocations-with-risk-control-under-partial-identifiability) |


---
<!-- trackio-cell
{"type": "figure", "id": "cell_b82ac27354f0", "created_at": "2026-07-28T16:28:29+00:00", "title": "Reproduction poster (poster_embed.html)", "pinned": true, "pinned_at": "2026-07-28T16:28:29+00:00"}
-->
[Open the full-resolution synthetic coverage figure](./synthetic_coverage.png)

Valid Gamma assumptions retain at least 90% coverage; the underestimated-Gamma
control fails at three middle tolerances. PNG SHA-256:
`62986A815A0505B343718C2390E806648AC35EDB7983007F1CE0677FC3CD7B16`.
