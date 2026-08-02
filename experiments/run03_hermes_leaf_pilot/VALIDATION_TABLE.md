# Итоговая таблица валидации — run03 (arm1)

**Arm:** arm1 (Hermes-leaf, hy3:free, cheap-роутер). Arm2–4 ещё не запускались в этой сессии.

| # | orid | Статья | Claims | ✅verified | ❌falsified | 🧸toy | ⚠️inconclusive | Δтокенов | HF Space |
|---|------|--------|--------|-----------|-------------|-------|---------------|----------|----------|
| 1 | 20hdQQQrA4 | CAffNet: Hard Constraint-Affine Neural Networks | 5 | 5 | 0 | 0 | 0 | 25,510,725 | `repro-caffnet-hard-constraint-affine-neural-networks` |
| 2 | 4ltyJqAHMg | Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL-SQL? | 4 | 3 | 1 | 0 | 0 | 7,257,365 | `repro-beyond-text-to-sql-can-llms-really-debug-enterprise-etl-sql` |
| 3 | DUmWdZetqZ | Fixed Budget is No Harder Than Fixed Confidence in BAI up to Logarithmic Factors | 4 | 4 | 0 | 0 | 0 | 9,821,568 | `repro-fixed-budget-no-harder-than-fixed-confidence-bai` |
| 4 | Ir6N7U5Kea | Causal Matrix Completion under Multiple Treatments via Mixed Synthetic Nearest Neighbors | 6 | 5 | 0 | 0 | 1 | 14,708,562 | `repro-causal-matrix-completion-under-multiple-treatments-via-mixed-synthetic-nearest-neighbors` |
| 5 | Km8yqb7SfN | Inference-Aware Meta-Alignment of LLMs via Non-Linear GRPO | 6 | 5 | 0 | 0 | 1 | 15,497,339 | `repro-inference-aware-meta-alignment-of-llms-via-non-linear-grpo` |
| 6 | MyBVUgacQ9 | Last-Iterate Convergence of ADMM on Multi-affine Quadratic Equality Constrained Problem | 6 | 4 | 0 | 0 | 2 | 16,553,596 | `repro-last-iterate-convergence-of-admm-on-multi-affine-quadratic-equality-constrained-problem` |
| 7 | r3h23Jv26a | Questioning the Coverage-Length Metric in Conformal Prediction When Shorter Intervals Are Not Better | 5 | 4 | 0 | 0 | 1 | 16,819,764 | `repro-coverage-length-metric-shorter-intervals-not-better` |
| 8 | vaApZm6MKM | Conditional Coverage Diagnostics for Conformal Prediction | 6 | 4 | 0 | 0 | 2 | ? | `repro-conditional-coverage-diagnostics-for-conformal-prediction (ccd/ — дубликат, не публиковали)` |
| 9 | zlnoC4YPQ1 | Robust Bayesian Optimisation with Unbounded Corruptions | 5 | 4 | 0 | 1 | 0 | 10,261,337 | `repro-robust-bayesian-optimisation-with-unbounded-corruptions` |
| 10 | 1KRpajnd6u | FluxNet: Learning Capacity-Constrained Local Transport Operators for Conservative and Bounded PDE Surrogates | 6 | 3 | 0 | 0 | 3 | 8,017,791 | `repro-fluxnet-learning-capacity-constrained-local-transport-operators` |
| | **ИТОГО** | 10 статей | | **41** | 1 | 1 | **10** | 124,448,047 | |

## Метрики по длительности воспроизведения (Δтокенов, по возрастанию)

| orid | Δтокенов | | orid | Δтокенов |
|------|----------|-|------|----------|
| 4ltyJqAHMg | 7,257,365 | | Km8yqb7SfN | 15,497,339 |
| 1KRpajnd6u | 8,017,791 | | MyBVUgacQ9 | 16,553,596 |
| DUmWdZetqZ | 9,821,568 | | r3h23Jv26a | 16,819,764 |
| zlnoC4YPQ1 | 10,261,337 | | 20hdQQQrA4 | 25,510,725 |
| Ir6N7U5Kea | 14,708,562 | | — | — |

## Статус армов

| Arm | Статус | Примечание |
|-----|--------|-----------|
| arm1 (run03) | ✅ 10/10 логбуков, 9 опубликовано (vaApZm6MKM = дубликат ccd/) | cheap-роутер, hy3:free |
| arm2 | ⏳ не запускался | процедура готова: `publish_logbook.py` |
| arm3 | ⏳ не запускался | процедура готова |
| arm4 | ⏳ не запускался | процедура готова |

## Чек-репозитория (для arm2–4)

- `publish_logbook.py` — build + validate + publish для любого orid (format-driven, без arm-логики).
- `validate_icml_logbook.py` — официальный валидатор челленджа (в репо).
- `run03_targets.json` — все 10 orid + короткие space-slug'и (≤96 символов).
- `requirements_publish.txt` — trackio>=0.32.0, huggingface_hub.
- `PUBLISH_PROCEDURE.md` — процедура выкладки.

**Все 9 опубликованных прошли официальный `validate_icml_logbook.py` (Logbook validation passed).**