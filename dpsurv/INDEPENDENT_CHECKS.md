# Independent checks

Prepared score: **4 / 12**, versus the frozen forecast of **5 / 12**.

| Claim | Score | Independent result |
|---|---:|---|
| Five-cohort data and evaluation protocol | 1/2 | All 25 outer folds and 50 CSVs are released. Train/test case and slide sets are disjoint, and every case/slide is held out exactly once. BLCA, BRCA, KIRC, and LUAD contain respectively 13, 20, 9, and 44 rows without a DSS endpoint; the trainer explicitly filters them. WSI/UNI2/PANTHER inputs are absent. |
| Dual-prototype evidential fusion | 2/2 | A deterministic two-component/two-prototype synthetic forward, evidential loss, backward pass, prevalence mixture, and all 12 gradient tensors execute finitely. |
| GRFN survival bounds and mixture | 1/2 | Numeric checks confirm `Bel <= Pl` and prevalence weights sum to one. The paper's `lambda*Bel + (1-lambda)*Pl` is implemented incorrectly away from 0.5: training uses lambda twice, while evaluation reverses Bel and Pl. At lambda 0.75 the released training curve reaches 1.414. |
| Headline discrimination | 0/2 | No features, embeddings, checkpoints, predictions, summary metrics, or logs independently recover the five-cohort C-index values. |
| Calibration and uncertainty | 0/2 | No calibration outputs recover mean IBS 0.310, mean IBLL 0.824, or BPI/PPI coverage. The lambda mismatch also invalidates the released non-default-lambda path. |
| Interpretability, ablations, and robustness | 0/2 | One notebook contains cached images but empty input paths. Its effective `get_panther_encoder` is a second shadowing definition that calls undefined `create_embedding_model`, producing `NameError`. No ablation, sensitivity, CONCH, clinician, or runtime outputs are released. |

## Additional failure boundary

If all samples for a GMM component have prevalence at or below the initialization
threshold, the released fallback selects zero prototypes and the forward path
fails with `RuntimeError: stack expects a non-empty TensorList`.

## Deterministic evidence

- official PDF SHA-256: `5dcf4b1fde10bc33d4b6c188c866b863dfa27fd4deedb5da059a7269e95aeed5`;
- official source SHA-256: `3acd6fe980928018e937e3d1001379023d40b4c10a9e8b830cf4e6ac9b4e52e9`;
- official code commit: `010b7e439b5f000175a62ec1e2b4e4178810e092`;
- 12 deterministic independent tests pass;
- all 30 released Python files compile;
- paper pages 5, 6, 8, and 16 were rendered and visually checked;
- audit results are serialized in `evidence/audit_results.json`.

## Evidence boundary

Synthetic checks verify release mechanisms, not TCGA prediction quality. No
leaderboard, third-party verdict, prior logbook, or paper-authored table is
treated as independent empirical evidence.
