# Model quality baseline: Claude workflow vs Codex

Date fixed: 2026-07-28

Purpose: keep a stable baseline for the current Claude-generated ICML reproduction workflow, then compare future Codex/OpenAI-model runs against the same scoring rubric.

> Status: historical development/pilot evidence. The prospective
> `Best-effort consumer subscription baseline` was frozen on 2026-07-29 in
> [`CONSUMER_SUBSCRIPTION_BASELINE.md`](CONSUMER_SUBSCRIPTION_BASELINE.md).
> Its numerical comparator is the 15 reported Claude-only logbooks:
> `46/15 = 3.07` points per logbook and `46/130 = 35.4%` of available points.
> Future engine runs must be recorded in `consumer_baseline_runs.csv` before
> challenge verdicts are inspected.

## Source of truth

Leaderboard visible score is the competition source of truth:

- User: `kondratevakate`
- Visible rank: `#66`
- Visible logbooks: `15`
- Visible points: `46`
- Score rate: `46 / 15 = 3.07 points/logbook`

Local diagnostic snapshot from downloaded judge verdicts on 2026-07-28:

- Judged entries in snapshot: `4980`
- Latest `judged_at`: `2026-07-28T12:51:33+00:00`
- Local `kondratevakate` diagnostic total: `16 logbooks / 48 points`

Use the visible leaderboard score for ranking decisions. Use local verdicts only to diagnose claim-level failures and lag/deduplication.

## Scoring rule

Per leaderboard:

- `verified` or `falsified`: `2 points`
- `toy`: `1 point`
- `inconclusive`: `0 points`
- Only one logbook per paper counts for each username.

## Claude-workflow baseline

Current average quality is low-to-medium throughput:

- Mean score: `3.07 points/logbook`
- Typical failure mode: many claims left `inconclusive`
- Strongest successful pattern: narrow theorem/simulation/survival claims with exact code and clear numerical evidence
- Weakest pattern: broad medical empirical claims where only synthetic proxies or partial datasets were run

Current per-logbook diagnostic snapshot:

| Paper/work | Local points | Max | Main judge pattern |
|---|---:|---:|---|
| SurvFD | 6 | 12 | mixed: verified algebra plus toy/inconclusive real-data/scalability claims |
| Solvable AE | 6 | 10 | strong on implemented claims, missing AMP/gradient-flow theory |
| Conditional Coverage Diagnostics | 4 | 4 | clean full score |
| Survival evaluation under censoring | 4 | 4 | clean full score |
| Online Cox / COLSA | 4 | 4 | clean full score |
| KM-ARL | 4 | 12 | missing KM-ADD, WISDM, Poisson/full panel |
| LERD | 3 | 12 | partial; theorem audit and updates need republish/rejudge |
| CalPro canonical | 3 | 12 | mostly synthetic proxy, real protein/docking claims untested |
| Entropy | 2 | 6 | toy CIFAR proxy, medical imaging SOTA untested |
| MDRCP | 2 | 12 | synthetic deviations, theory and FMoW untested |
| DC-PnPDP | 2 | 12 | toy manifold proxy, CT/MRI claims untested |
| uqct | 2 | 12 | toy phantom, real CT/deep reconstructors untested |
| SALaD | 2 | 4 | one claim falsified, broad 10-dataset/factor claim not fully addressed |
| FunCQNet | 1 | 4 | surrogate simulation only; private clinical claim unresolved |
| ConfSleepNet | 1 | 10 | old judged version lacked implemented pipeline; local repair exists |

## Error taxonomy to beat

Future Codex runs should reduce these failure rates:

1. Claim not attempted: automatic `0`.
2. Synthetic proxy for real medical claim: usually `toy`, rarely full score.
3. One dataset/layer tested for a many-dataset claim: usually `inconclusive`.
4. Mechanism tested but not the actual claim object: `toy` or `inconclusive`.
5. Missing baseline panel: empirical superiority claims fail.
6. Published duplicate Space for same paper: may not improve leaderboard because only one logbook per paper counts.
7. Hedged verdict sentence: judge often treats it as weak evidence even when the experiment is useful.

## Codex comparison targets

Codex/OpenAI workflow should be judged on:

| Metric | Claude baseline | Codex target |
|---|---:|---:|
| Points/logbook | 3.07 | 6.0 minimum, 8.0 target |
| Full-score small papers | 3 clean examples | repeatably identify before implementation |
| Inconclusive claims/logbook | high | below 25% of claims |
| Toy claims/logbook | moderate | only when intentionally accepted |
| Expected points/hour | not tracked | tracked before each paper |
| Preflight pass before publish | informal | mandatory |

## Go/no-go rule for new medical targets

Take a paper only if preflight estimates:

- `expected_points >= 6`, and
- `expected_points_per_hour >= 1.5`, and
- at least one claim has a full `verified` or full `falsified` path that does not require private data.

Exceptions require an explicit strategic reason, such as unusually high medical relevance or easy repair of an already published Space.

## Comparison protocol

For every future Codex paper, record:

- model and thinking setting;
- time spent;
- number of claims;
- predicted score before publish;
- actual judge verdicts;
- delta between predicted and actual score;
- postmortem for every `toy` or `inconclusive`.

Recommended Codex model allocation:

- `GPT-5.6 Terra` or `GPT-5 mini`, medium: mass triage.
- `GPT-5.6`, high: target selection and claim decomposition.
- `GPT-5.3-Codex`, high: implementation, scripts, packaging.
- `GPT-5.6 Pro`, highest: final judge-preflight and repair decisions.

## GPT-5.6 Sol, high: prediction baseline

Latest verdict-derived snapshot on 2026-07-29: `17 logbooks / 62 points`.
The visible leaderboard rank has not been refreshed in this note. This section
does not rewrite the fixed Claude baseline above.

Model and setting: `GPT-5.6 Sol`, `high` thinking, in Codex.

| Work | First rough forecast | Final forecast after claim audit | Actual |
|---|---:|---:|---:|
| Treatment Allocations | 5-7/10 | 4-5/10 | 4/10 |
| KM-ARL repair | 8-10/12 | 8-10/12 | 10/12 |
| ROCP repair | 2-3/8 | 2-3/8 | 2/8 |
| MDRCP repair | 5-6/12 | 5-6/12 | 6/12 |

Strict first-forecast calibration:

- Per-paper interval hit rate: `3/4 = 75%`.
- Mean distance outside the forecast interval: `(1 + 0 + 0 + 0) / 4 = 0.25` points.
- Mean absolute error to forecast midpoint:
  `(2 + 1 + 0.5 + 0.5) / 4 = 1.0` point.
- Treatment was overestimated; KM-ARL and MDRCP hit their upper bounds; ROCP
  hit its lower bound.

Final post-audit calibration:

- Per-paper interval hit rate: `4/4 = 100%`.
- Forecast total: `19-24`; actual total: `22`.
- Mean absolute error to forecast midpoint:
  `(0.5 + 1.0 + 0.5 + 0.5) / 4 = 0.625` points.
- Net account gain from the four judged actions: `+16` points, from `46` to
  `62`.

The sample is too small to claim general calibration. The earlier,
pre-decomposition Treatment estimate of `5-7` was optimistic and missed by one
point below its lower bound; the final `4-5` estimate was accurate.

Claim-level calibration is weaker than the total-score calibration:

- MDRCP was predicted correctly in both total and composition: C1-C3 were all
  judged `verified`, and C4-C6 remained `inconclusive`, for `6/12`.
- ROCP reached the predicted total (`2/8`) for a different reason than
  expected: C1 and C3 were both judged `toy`, while C2 and C4 were
  `inconclusive`. The interval hit is real, but the claim-level forecast was
  not well calibrated and should not be counted as a clean mechanistic hit.

### Pending forecasts locked before verdict

| Work | Model / thinking | Forecast | Point estimate | Expected gain | State |
|---|---|---:|---:|---:|---|
| Conditional Coverage six-claim repair | GPT-5.6 Sol / high | 11-12/12 | 12/12 | +7 to +8 | published, pending |
| SurvFD three-scale + scalability repair | GPT-5.6 Sol / high | 7-8/12 | 8/12 | +1 to +2 | published, pending |
| Cox subgroup discovery C1/C2/C4 | GPT-5.6 Sol / high | 5-6/12 | 6/12 | +5 to +6 | published, pending |
| ROCP full cached C3/C4 repair | GPT-5.6 Sol / high | 6/8 | 6/8 | +4 | published, pending |

The MDRCP forecast was locked after independent claim audits but before
updating the canonical Space or reading a new judge verdict. It includes the
previously verified C1, predicts full credit for C2, and allows either toy or
full credit for C3. C4-C6 are forecast at zero until their exact official
protocols are completed.

The Conditional Coverage forecast was locked after the six-claim bundle
passed locally but before updating the canonical Space. C2 and C5 are exact
replays of hash-pinned official released outputs, not fresh training; the
one-point range allows one replay claim to be scored as toy.

The SurvFD repair forecast was locked after the independent scripts passed but
before the canonical Space was updated. C1 now exercises exact functional ANOVA
on log-hazard, hazard, and survival scales with machine-precision
reconstruction. C6 is intentionally forecast as `toy`: it covers four
synthetic 10-16-feature games and two estimator families, not the paper's four
real datasets and full approximation panel.

The Cox subgroup discovery forecast was locked after the independent C1
proper-scoring-rule audit, the full C2 theorem dependency audit, and a fresh
10-seed C4 run at official commit
`96725eec197ce3d08d24560d4bd2e697b3bfdbdb`, but before creating a canonical
Space. The Table 2 core run reproduced the paper's DDGroup values to rounding:
box F1 `0.969 (0.008)`, EPE `0.378 (0.016)`, and C-index `0.867 (0.006)`.
C3, C5, and C6 remain explicitly not attempted; peer leaderboard artifacts
were used only to locate the official repository and were not used as
reproduction evidence.

The ROCP repair forecast was locked after 20 COVID seeds under each of two
loss matrices, 20 BDD splits, the exact finite Algorithm 1 audit, and remote
hash verification, but before a new judge verdict. C1/C3/C4 are forecast at
two points each and C2 remains not attempted. The logbook explicitly reports
small BDD RAC reversals against the paper's stronger all-alpha statement.
