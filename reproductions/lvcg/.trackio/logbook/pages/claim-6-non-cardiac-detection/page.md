# Claim 6: non-cardiac detection


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_b680b10a7aef", "created_at": "2026-07-30T07:29:41+00:00", "title": "Claim 6: non-cardiac detection"}
-->
**UNSUPPORTED - 0/2.** The paper evaluates MIMIC-IV-ECG-Ext-ICD,
but the repository supplies an AI-READI provider instead and contains no
paper-dataset loader, folds, labels, checkpoint, or outputs.


---
<!-- trackio-cell
{"type": "code", "id": "cell_ba4fe1ce0f20", "created_at": "2026-07-30T07:29:41+00:00", "title": "Claim 6: non-cardiac detection evidence", "language": "python"}
-->
````output
{
  "paper_objective_has_three_terms": true,
  "released_training_adds_base_loss": true,
  "post_paper_gru_shape_fix": {
    "commit": "0fcacbf34784cd876b4c197599253a9130707f93",
    "message": "Fix beat loss shape mismatch in GRU path; add project links to README.",
    "changed_training_file": true
  },
  "checkpoint_override_is_not_serialized": true,
  "paper_noncardiac_dataset": "MIMIC-IV-ECG-Ext-ICD",
  "released_noncardiac_dataset": "AI-READI",
  "release_config_mentions_aireadi": true,
  "release_config_mentions_mimic_ext_icd": false,
  "paper_source_mentions_mimic_ext_icd": true
}
````
