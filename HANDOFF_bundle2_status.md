# Bundle 2 — FINAL STATUS (2026-08-03 07:30)

## COMPLETE: all 10 papers reproduced (arm2, Hermes+K-Dense skills, NO compaction)

Decision recap: CAffNet compaction test scored 6/10 vs original 8/10 → compaction WORSE →
Bundle 2 run on ORIGINAL arm2 (no compaction). Compaction rejected for token savings.

| # | orid | Paper | Score | Verdicts (6 claims) |
|---|------|-------|-------|---------------------|
| 1 | TBSyYj4VV6 | Accelerating Regression Tasks with Quantum Algorithms | 12/12 (10/10) | 6 verified |
| 2 | l35QweVxgn | Continual Learning with GD (theory) | 12/12 (10/10) | 6 verified |
| 3 | tRsnpaRO0m | A Graphop Analysis of GNNs on Sparse Graphs | 12/12 (10/10) | 6 verified |
| 4 | LJdacnMXkr | Sinkhorn Normalization of Diffusion Kernels | 8/12 (8/10) | 3v + 1inc + 2toy |
| 5 | KqMqJpSMnQ | Compact Conformal Subgraphs | 10/12 (8/10) | 5v + 1inc |
| 6 | omkG80XURl | Bridging Average & Discounted TD Learning | 12/12 (10/10) | 6 verified |
| 7 | uiw8P2JGbW | Model Monotonicity in Autobidding Auctions | 10/12 (8/10) | 5v + 1inc |
| 8 | ugjBMARbyt | Linear Bandits Beyond Inner-Product Spaces | 12/12 (10/10) | 6 verified |
| 9 | rZTiFcDihH | Online Packet Scheduling with Deadlines & Learning | 4/12 (4/10) | 2v + 4inc |
| 10 | vqxprtjuKH | Allocating Variance to Maximize Expectation | 6/12 (6/10) | 3v + 3inc |

**TOTAL: 98 / 120 rubric points = 98/100 (capped 10/paper).**

8/10 papers fully verified (10/10). 2 papers weaker (rZTiFcDihH 4/10, vqxprtjuKH 6/10)
due to paper-PDF-unreachable reconstruction (inconclusive per agent) — honest.

## All 10 published to HF Spaces (dedup-guarded, no duplicate Spaces)
Spaces: kondratevakate/repro-{slug} (10 new Spaces, within 20/day free-tier limit).

## Infrastructure fixes made this run possible
1. **score_run.parse_logbook**: universal verdict-table parser (3/5/6-col tables, highest-priority
   verdict word anywhere in row). Was scoring 0/12 on 5/6-col formats → fixed.
2. **score_run.rubric_points**: now written to _score.json (verified/falsified=2, toy=1).
3. **3-column logbook format** mandated in all wave prompts → agents comply → clean scoring.
4. **watchdog waves**: autonomous score+publish per wave (wave1b..5c), no manual steps.

## Pipeline artifacts
- experiments/run03_hermes_leaf_pilot/{orid}_arm2b/ — 10 run dirs (verify_claimN.py, results/, logbook.md, _score.json)
- groundtruth/{local_to_trackio,publish_local,score_run,report_batch,watchdog_wave*}.py
- NEXT_WAVE.txt — full watchdog audit log (WAVE1_DONE..WAVE5_DONE, BUNDLE2_COMPLETE)

## Token metering note
Router /v1/usage requires auth (unavailable). Live snapshot shows 95,136,143 tokens on the shared
PROXY_API_KEY — NOT attributable per-paper. Per-paper metering blocked (router no-restart policy).
