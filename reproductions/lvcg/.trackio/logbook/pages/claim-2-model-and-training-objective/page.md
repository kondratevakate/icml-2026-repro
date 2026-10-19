# Claim 2: model and training objective


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_7214ca6ddb79", "created_at": "2026-07-30T07:29:41+00:00", "title": "Claim 2: model and training objective"}
-->
**UNSUPPORTED - 0/2.** Training cannot import. The released default
GRU path required a post-paper beat-loss shape fix, and the implementation
adds a unit-weight base-beat loss absent from Equation 7.


---
<!-- trackio-cell
{"type": "code", "id": "cell_6eafb314e114", "created_at": "2026-07-30T07:29:41+00:00", "title": "Claim 2: model and training objective evidence", "language": "python"}
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
