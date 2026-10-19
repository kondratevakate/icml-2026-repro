# PyHealth 2.0 Claim Decomposition

Paper ID: `gMLVFN9hl8`. Frozen forecast before execution: `4/10`, with a
`3-6/10` plausible range.

## C1: toolkit coverage

Prepared verdict: `VERIFIED` if both the paper inventory and the pinned public
release independently clear all four numeric thresholds and expose witnesses
for every named modality and interpretation method.

Required evidence:

- Parse the dataset, task, model, and interpretability overview tables.
- Inventory public exports at official release `v2.0.1`.
- Require explicit witnesses for EHR, imaging, physiological signals, genomics,
  Attention-Grad, GIM, DeepLift, and SHAP.

Expected score: `2/2`.

## C2: mortality throughput and memory

Prepared verdict: `NOT ATTEMPTED`.

A full test requires MIMIC-IV v2.2, the paper's task definition, worker counts
1/4/8/12/16, peak-memory instrumentation, and all Pandas, PyHealth 1.16, MEDS,
and PyHealth 2.0 baselines. A MIMIC demo or single-worker smoke test is not a
valid substitute.

Expected score before data/GCP: `0/2`.

## C3: drug-recommendation and length-of-stay throughput

Prepared verdict: `NOT ATTEMPTED`.

The same full benchmark contract as C2 applies to both tasks. No extrapolation
from mortality or from a subset of tables is allowed.

Expected score before data/GCP: `0/2`.

## C4: seven lines and Table 2

Prepared verdict: `FALSIFIED` if the challenge's attributed Table 2 values do
not equal the machine-parsed paper table.

The challenge claim is conjunctive and specifically anchors `7/24/51` to Table
2. The audit must preserve the distinction between:

- the abstract/methodology statement that a training path can take seven lines;
- the Table 2 mortality-task implementation counts.

Expected score: `2/2`.

## C5: community, tutorials, and RHealth

Prepared verdict: `INCONCLUSIVE` unless all three components have independently
auditable, time-matched evidence.

Repository files can test the examples/tutorials component, and the cited
RHealth repository can test the R interface component. A paper assertion alone
is not independent evidence for the historical community membership count.

Expected score: `0/2`; judge upside `1/2` for the two auditable components is
not included in the forecast.
