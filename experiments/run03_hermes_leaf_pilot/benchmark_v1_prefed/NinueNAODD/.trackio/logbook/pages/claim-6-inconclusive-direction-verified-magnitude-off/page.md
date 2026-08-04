## Claim 6 — — inconclusive (direction verified, magnitude off)

**Verdict:** sec. 4.3, fig. 2, eq. 2

*Source:* Sec. 4.3, Fig. 2, Eqs. (2) and (12). *Script:* `verify_claim6.py` (5 s). Real IHQ data.

| split | our r | n (ours) | paper r | n (paper) |
|---|---|---|---|---|
| unscreened | **0.855** (p = 6.8e-18) | 59 | 0.724 | 62 |
| screened | **0.755** (p = 5.5e-10) | 48 | 0.551 | 50 |

- Strong, highly significant positive correlation between the fitted rater-quality parameter q_r and each rater's agreement with the final BBQ ranking, on both splits, with the same pattern as Fig. 2 (**unscreened > screened**). Fitted line on unscreened: `q = 0.519·agreement + 0.500` (paper: `0.348·x + 0.631`).
- Magnitudes are ~0.13–0.20 higher than the paper's. Plausible causes: the smaller public sample, and the paper not specifying its "agreement with the final ranking" statistic (we use the fraction of a rater's comparisons whose winner outranks the loser under the BBQ ranking on the same split).
- **Mutation:** permuting q_r across raters (1,000 permutations) gives mean |r| = 0.106, max 0.493, permutation p = 0.000. ✔ the correlation is not an artefact of the marginals.
- The *practical* part of the claim — that q_r flags unreliable raters without a separate screening step — is supported: on the unscreened split q_r spans 0.60–0.95 and tracks agreement (0.31–0.97) monotonically.

---

---
<!-- trackio-cell
{"type": "markdown", "id": "cell_<built-in function hash>", "created_at": "2026-01-01T00:00:00+00:00", "title": "Claim 6 \u2014 \u2014 inconclusive (direction verified, magnitude off)"}\n-->
