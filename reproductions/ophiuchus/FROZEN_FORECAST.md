# Ophiuchus frozen forecast

Frozen: 2026-07-29. OpenReview `coJqVkqb03`; arXiv `2512.14157v2`.

Written after primary paper/repository inventory and before independent code,
schema, runtime, or result checks. Leaderboards and third-party verdicts are
excluded.

| Claim | Expected |
| --- | ---: |
| C1. The release implements an interleaved medical image agent with SAM2, BiomedParse, and zoom-in tools. | 2/2 possible from source tracing plus dependency-light parser/state/tool tests. |
| C2. The release implements cold-start SFT, self-reflection fine-tuning, and ATRL with decomposed rewards. | 1-2/2 possible from script/config/reward-path inspection; full training is likely blocked by model/tool services and accelerators. |
| C3. Training uses a curated 64k tool-integrated trajectory dataset. | 0/2 expected: the README labels the dataset as coming soon and exposes only minimal examples. |
| C4. Ophiuchus outperforms open and closed models across eight medical benchmarks. | 0/2 expected: checkpoint and complete evaluation instructions are marked coming soon. |
| C5. Tool use improves grounded segmentation, reflective correction, and tool transfer. | 0/2 expected without the trained checkpoint, evaluation datasets, and canonical ablation outputs. |

Frozen expectation: **3/10**; plausible range **2-4/10**. Released examples
may support schema and state-machine checks only; they cannot substitute for
the 64k corpus or paper-level performance evidence.
