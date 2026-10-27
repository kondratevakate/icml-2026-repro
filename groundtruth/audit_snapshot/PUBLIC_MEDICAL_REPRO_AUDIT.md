# Public Medical Reproducibility Audit

Snapshot: `2026-07-29T01:40:39+00:00`.

This audit separates independent reproduction evidence from author-artifact signals.
Leaderboard evidence comes from Hugging Face Spaces tagged `icml2026-repro` and the public verdicts dataset.
Author-artifact evidence is extracted from local ICML challenge metadata and abstracts, with live URL checks where possible.

## Headline Counts

- Medical/life-science candidates: 703
- Author GitHub repository signal: 127 (18.1%)
- Live GitHub repository OK: 9 (1.3%)
- HF URL signal: 4 (0.6%)
- Weights/checkpoint signal: 103 (14.7%)
- Open/public dataset signal: 466 (66.3%)
- Simulation/theory no-external-data signal: 184 (26.2%)
- Private/restricted data signal: 12 (1.7%)
- GPU-required/risk signal: 475 (67.6%)
- Leaderboard attempts: 224 (31.9%)
- Positive leaderboard evidence: 193 (27.5%)
- At least one verified/falsified claim event: 132 (18.8%)
- At least one falsified claim event: 34 (4.8%)

## Output Files

- `public_repro_audit.csv`: one row per candidate paper.
- `leaderboard_medical_logbooks.csv`: one row per relevant leaderboard Space/logbook.
- `artifact_links.csv`: extracted non-OpenReview artifact URLs and live status checks.
- `github_repo_audit.csv`: GitHub repository metadata; tree-level fields require `--deep-github`.
- `public_repro_audit_summary.json`: machine-readable headline counts.

## Caveats

- `repo_signal`, `dataset_signal`, `weights_signal`, and `gpu_required_signal` are audit signals, not final manual verdicts.
- Leaderboard points may come from full reproduction, falsification, or toy-scale evidence, and may not cover all claims in the paper.
- GitHub live metadata is rate-limited without authentication; use `GITHUB_TOKEN` plus `--github-metadata --deep-github` for a complete repository/weights pass.
- Dataset size and GPU-hours require a second pass over repositories, READMEs, paper PDFs, and/or published logbook evidence.
