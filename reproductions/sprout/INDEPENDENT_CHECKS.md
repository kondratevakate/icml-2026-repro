# Independent checks

Prepared score: **5 / 12**, versus the frozen forecast of **4 / 12**.

| Claim | Score | Independent result |
|---|---:|---|
| Fully training-free automatic prompting | 1/2 | The release exposes the intended image-to-prompt-to-SAM stages, but the canonical pipeline is not executable as released: POT raises on a missing attribute, SAM2 is external, the documented dataset link is empty, and the default backbone repository does not resolve. |
| Self-reference feature calibration | 2/2 | On a deterministic 128x128 H&E-like image, the official high-confidence reference-mask branch executes with precision 1.000 and recall 0.607 against nine synthetic nuclei. |
| POT-Scan formulation and guarantee | 1/2 | The canonical solver raises `AttributeError` because `numItermax` is unset. Supplying only that runtime field lets the solver run, but it produces unit row sums and total real mass 2.4 for `N=4, rho=0.6`, rather than the paper-normalized mass 0.6. A zero-cost 2x2 counterexample also shows that convexity alone does not imply equality of particular optimizers. |
| Containment-aware refinement | 1/2 | The official soft-NMS penalizes a large mask containing two smaller masks from 0.9 to 0.211 while retaining the smaller masks at 0.8. The same refinement path crashes with `IndexError` on an empty proposal set. |
| Benchmark performance | 0/2 | No released predictions, processed splits, checkpoints, metric tables, or run outputs independently recover MoNuSeg, CPM17, TNBC, or PanNuke results. |
| Robustness and efficiency | 0/2 | No three-repeat outputs, per-backbone/SAM/hyperparameter artifacts, or runtime traces are released. |

## Deterministic evidence

- official PDF SHA-256: `bf31eea0e1312268f3dbaee43f821cc4d90ac5e6bba57d149e6899989339d1b0`;
- official source SHA-256: `d7abb8b085d2ea5183e69a32509d332967af9f45ecff33f1e5cf7fda41e6927f`;
- official code commit: `ee23c5fb41c9fe302a28883a08935b674639d7cb`;
- eight independent unit tests pass;
- all 17 released Python files compile;
- perfect identity masks yield Dice 1.0 and numerically unit AJI/PQ;
- audit results are serialized in `evidence/audit_results.json`.

## Evidence boundary

The synthetic checks test the released mechanisms, not the paper's pathology
benchmark accuracy. No leaderboard, third-party verdict, prior logbook, or
unreleased author output is used.
