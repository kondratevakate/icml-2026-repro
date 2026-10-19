# Consumer subscription baseline prompt v1

Version frozen: 2026-07-29

Use the text below unchanged as the initial prompt for both subscription
systems. Replace only the bracketed paper-specific paths and identifiers.

---

You are independently reproducing one life-science or medical ML paper.

Inputs:

- paper ID: `[PAPER_ID]`
- paper PDF: `[PAPER_PATH]`
- anchored claims: `[ANCHORED_CLAIMS_PATH]`
- supplement, if available: `[SUPPLEMENT_PATH_OR_NONE]`
- official repository, if available: `[OFFICIAL_REPOSITORY_PATH_OR_NONE]`
- output directory: `[OUTPUT_DIRECTORY]`

Produce the strongest scientifically defensible CPU reproduction possible
within 2 hours of active work and 4 CPU-hours. Do not use paid APIs, paid GPUs,
leaderboard logbooks, challenge verdicts, third-party reproductions, or work
from another agent.

Work autonomously:

1. Read the paper and exact anchored claims.
2. Audit the official artifacts and data availability.
3. Select the claims that can be tested faithfully within the budget.
4. Implement and run the experiments.
5. Use exact enumeration or analytic checks when stronger than sampling.
6. Include baselines, repeated seeds, negative controls, or mutation tests when
   needed to distinguish the claimed mechanism from a proxy.
7. Separate full reproduction, toy evidence, and unresolved claims. Never
   convert partial evidence into a full-paper claim.
8. Produce a runnable artifact, environment specification, machine-readable
   results, and a concise logbook mapped exactly to the anchored claims.
9. Run local validation and use one repair pass to correct failures.

Before stopping, write `frozen_prediction.json` containing:

- a predicted challenge verdict for every anchored claim;
- predicted total points;
- evidence path for every non-inconclusive prediction;
- whether each experiment is full or toy;
- unresolved limitations;
- wall time, CPU time, and all requested human interventions.

Do not inspect external reproductions or challenge verdicts after beginning the
task. Ask the operator only for missing files, environment approval, or a
choice between alternatives you have already analyzed. The operator will not
write or debug code for you.

---

Any follow-up after the first response must be limited to environment answers,
approvals, and one explicit repair request:

> Run the baseline protocol's single repair pass. Audit every anchored claim,
> execute the local preflight, repair only defects found in this artifact, and
> update `frozen_prediction.json`. Do not use external reproductions or
> challenge verdicts.

