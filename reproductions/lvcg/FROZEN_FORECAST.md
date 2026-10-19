# Frozen forecast

Locked at `2026-07-30T11:23:00+04:00`, after inventorying the paper, source,
official repository at commit `0fcacbf34784cd876b4c197599253a9130707f93`,
released split files, and dataset requirements. This forecast was fixed before
independent numerical, runtime, and split-integrity checks.

## Forecast

**4/12 points**, plausible range **3-6/12**.

| Claim | Forecast | Pre-check basis |
| --- | ---: | --- |
| C1 fixed ECG/VCG geometry | 2/2 | Equations and executable lift/project code are public and permit synthetic numerical checks. |
| C2 model and training objective | 1/2 | The full implementation and configuration are public, but the default GRU beat-loss path required a post-paper shape fix. |
| C3 six-dataset linear probing | 1/2 | Official split CSVs and probing code are public, but the referenced pretrained checkpoint and result files are absent. |
| C4 multi-lead reconstruction | 0/2 | Evaluation code is public, but checkpoints, predictions, and metric outputs are absent. |
| C5 geometry/component ablations | 0/2 | Only rendered paper tables are released; no ablation configs, checkpoints, or outputs were found. |
| C6 non-cardiac detection | 0/2 | AI-READI preparation/probing code is present, but no task split, checkpoint, predictions, or result files are released. |

The main upside is that the central fixed-geometry and model paths can be
tested independently on synthetic inputs. The main downside is that none of
the headline empirical tables can be regenerated from the repository without
large external datasets, a long pretraining run, and unreleased checkpoints.
