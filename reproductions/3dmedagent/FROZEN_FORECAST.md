# 3DMedAgent frozen forecast

Frozen: 2026-07-29. Paper `TH6pLxCOQ3`, arXiv `2602.18064v2`.

Written after official artifact inventory and before independent counts or
runtime checks. Leaderboards and third-party verdicts are excluded.

| Claim | Expected |
|---|---:|
| C1. OAMI → CFLT → T1S-Loop (`T_max=5`) is implemented as the declared three-stage agent. | 2/2 possible via source/code routing audit and CLI/static smoke checks. |
| C2. DeepChestVQA contains 892 CT scans and 1,020 QA pairs across 17 subtypes: Recognition 3/180, Visual Reasoning 8/480, Medical Reasoning 6/360. | 2/2 expected from an independent CSV schema/count/uniqueness audit. |
| C3. 3DMedAgent yields about 20% average accuracy gain across 40+ tasks. | 0/2 expected: full CT volumes, masks, intermediate caches, model credentials and canonical predictions are not locally frozen. |
| C4. 3DMedAgent consistently outperforms general, medical and 3D-specific MLLMs on DeepChestVQA. | 0/2 expected for the same reason. |
| C5. OAMI, CFLT and T1S each provide consistent ablation gains. | 0/2 expected without canonical cached evidence/predictions. |
| C6. CFLT slice selection approaches inter-radiologist agreement and T1S improves matched cases at every turn. | 0/2 expected: radiologist annotations and matched per-turn outputs are not available as a compact runnable bundle. |

Frozen expectation: **4/12**; plausible range **2–4/12**. Stop empirical
claims if inputs, credentials, or original predictions are missing. Paper
tables and cached author scores count only as artifact concordance.
