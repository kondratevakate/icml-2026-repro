# TASK: reproduce the anchored claims of one ICML 2026 paper (CPU only)

You are an autonomous reproduction agent. Work in the current directory.

## Input
`input_bundle.json` contains: paper title, OpenReview URL, arXiv id, and the list of
**anchored claims**. These claims are the task specification — do not invent your own.

## What to produce
1. `plan.md` — for each claim: is it reproducible on CPU within budget? theory (analytic/symbolic),
   simulation, or requires data/GPU (then say so and STOP for that claim — a correct refusal is
   worth more than a fake toy substitute).
2. Verification scripts (`verify_claim<N>.py`) that actually run. Python 3.12, CPU only,
   stdlib + numpy/scipy/sympy only (pip install is allowed for these).
3. `results/claim<N>.json` — raw numeric outputs of each run, with the exact command used.
4. `logbook.md` — per claim: `verdict` ∈ {verified, falsified, toy, inconclusive}, the number(s)
   that justify it, the exact source location in the paper (theorem/section/equation number),
   and an **"Evidence boundary"** section at the end listing what your evidence does NOT cover.

## Hard rules
- **Mutation test**: for every claim you mark `verified`, also break the mechanism deliberately
  (invert a condition, replace a component with a naive alternative) and show the result changes
  as predicted. A number without a mutation test proves correlation, not mechanism.
- **Exhaustive enumeration over seeds** when the claim space is finite. Seeds are a fallback.
- Never mark a claim `verified` on a single seed.
- If data is unavailable, the verdict is `inconclusive` with the reason — NOT a synthetic toy
  standing in for real data.
- Do NOT search for or read other people's reproduction logbooks (HuggingFace Spaces tagged
  `icml2026-repro`). Reading the paper itself and its official code is allowed.
- Budget: ~90 minutes wall time. Prefer finishing 2 claims properly over 6 claims shallowly.

## Finish
Print a final summary table: claim number | verdict | one-line evidence.
