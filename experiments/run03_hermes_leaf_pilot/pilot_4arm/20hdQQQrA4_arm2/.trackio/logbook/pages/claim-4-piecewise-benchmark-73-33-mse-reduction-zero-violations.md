## Claim 4 — piecewise benchmark (73.33% MSE reduction, zero violations)

Route: split the claim (skill §2g).

- **Structural half — verified.** Over 1200 test points (3 seeds × 400) the CAffine model
  records **0** violations (max 4.4e-16) while the soft-penalty baseline, trained
  identically, records **413**. This is the mutation: removing the layer makes violations
  appear.
- **Magnitude half — toy.** Measured MSE 7.3826e-04 (soft) vs 6.7017e-04 (CAffine), a
  reduction of **9.22%**, far from the paper's 73.33%. `scale_note`: PAPER = CAffNet-TF
  transformer on the (unreleased) piecewise benchmark, GPU; THIS RUN = 1-32-32-2 numpy MLP,
  1500 Adam steps, 3 seeds, CPU. The reduced-scale number is **not** presented as
  reproducing the table.
- **Paper-internal arithmetic:** 1 − 0.0012/0.0045 = **73.333%**, so the quoted percentage
  is at least internally consistent with a 4:15 MSE ratio.

Verdict: **verified (structural) / toy (magnitude)**.
