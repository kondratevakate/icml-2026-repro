# Independent checks

## Outcome

Prepared score: **2/12**, below the frozen forecast of **4/12**.

## C1 - Partial (1/2)

An independent NumPy implementation of the printed ridge lift and fixed
projection recovers random full-rank latent trajectories with maximum absolute
error `6.90e-06` and visible-lead reprojection error `3.74e-06`.

The official geometry module cannot be executed: importing
`lvcg.models.vcg` first imports `lvcg.models.lvcg`, which requires
`lvcg.data.angle` and `lvcg.data.beat_segmentation`. Neither module, nor the
containing `lvcg/data` package, exists anywhere in the public Git history.

## C2 - Unsupported (0/2)

The default training entrypoint fails before argument parsing with
`ModuleNotFoundError: No module named 'lvcg.data'`. The latest official commit
also states that the prior GRU beat-loss shape mismatch caused the default
training path to crash.

Equation 7 contains ECG, VCG-beat, and temporal losses. The released default
configuration adds a fourth unit-weight `base_beat_loss` term that is not in
the paper objective. Appendix Table 8 also lists only VCG and temporal loss
weights, confirming visually that no base-beat weight is disclosed. Therefore
neither the executable path nor exact paper/code conformance is established.

## C3 - Partial (1/2)

All six released task groups have disjoint record splits; where `patient_id`
is supplied, patient sets are also disjoint. The Table-1 LVCG averages
recompute to 67.3017, 74.5417, and 80.4650, consistent with the displayed
67.30, 74.54, and 80.47 at printed precision.

However:

- no pretrained checkpoint or result CSV is released;
- `--checkpoint` in `scripts/evaluate.py` only mutates an in-memory YAML
  object, then launches the subprocess with the unchanged YAML file, so the
  override has no effect;
- PTB-XL Sub-Class label columns have different order in train versus
  validation/test. The dataset loader converts positional columns to arrays
  independently, so some validation/test targets are assigned to the wrong
  class positions;
- `chapman_test.csv` has a leading blank line, tolerated by pandas but not by
  every CSV reader.

These checks support the split inventory, not the reported AUCs.

## C4 - Unsupported (0/2)

No checkpoint, reconstruction predictions, per-lead outputs, or metric files
are released. Table 2 cannot be regenerated.

## C5 - Unsupported (0/2)

The geometry and component ablations appear only as paper tables. There are no
ablation configurations, checkpoints, seeds, or outputs in the repository.

## C6 - Unsupported (0/2)

The paper reports non-cardiac tasks on MIMIC-IV-ECG-Ext-ICD. The repository
contains no corresponding loader, code-to-label mapping, folds, or config.
Instead it configures AI-READI, a different dataset. The reported CKD,
diabetes, and sepsis AUCs therefore have no executable official path.
