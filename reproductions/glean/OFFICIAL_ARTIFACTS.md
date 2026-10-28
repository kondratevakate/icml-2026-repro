# Official artifacts

## Paper

- OpenReview: `FP23eFYhAy`
- arXiv: `2603.02798v1`
- PDF SHA-256:
  `1EE9BF721494DE9EF78CB06EF396583DE90CEF8BB8093B18E6D8B9C71E291FF6`
- source archive SHA-256:
  `CEB733675F7DA5F1C3991F47D5B77506009E02D584AED2B3541074F8BF2D4016`

The arXiv source contains the tables, figures, method, prompts, and proof. No
official GLEAN repository, executable implementation, generated trajectory
release, prediction matrix, checkpoint, or calibration split is linked.

## Public guideline corpus

The paper links `epfl-llm/guidelines`. The Hugging Face Dataset Viewer reports:

- one `default/train` split;
- 37,970 rows and seven columns;
- 877,724,423 bytes in the original files;
- 424,634,397 bytes in Parquet form.

Only public metadata and first-row schema information were cached. The
credentialed clinical data are never copied into the publication bundle.

## Credentialed clinical dataset

The exact upstream benchmark is MIMIC-IV-Ext Clinical Decision Making v1.1,
not the full MIMIC-IV archive. It contains 2,400 abdominal-pathology cases and
is stored locally under the repository-wide ignored `data/` directory.

- local archive SHA-256:
  `C55AD01AAF9D424A582C404687730614A78A0F4155A5889B3C7108FAB5FE038B`
- all 12 payload checksums in the official `SHA256SUMS.txt` are checked by the
  audit;
- MIMIC-IV-Note v2.2 is also present locally for cross-checking table headers,
  but is not required to publish this audit.

No patient identifiers or text are emitted by the audit.

