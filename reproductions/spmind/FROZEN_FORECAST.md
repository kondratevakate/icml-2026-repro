# SP-Mind frozen reproducibility forecast

Frozen on: 2026-07-29  
Paper: *SP-Mind: An Autonomous Reasoning Agent for Spatial Proteomics Analysis*  
Paper ID: `UJcB3XrffF`

This forecast was written after identifying and inventorying the official paper,
source, repository, and dataset, but before running the independent checks in
this package. Leaderboards and third-party verdicts were not consulted.

## Claim-level forecast

| Claim | Forecast | Reason fixed before execution |
|---|---:|---|
| C1. SP-Mind implements an autonomous end-to-end spatial-proteomics agent with a ReAct-style loop, expert skills, and more than ten tools spanning eight analysis stages. | 2/2 possible for the implementation/architecture portion. The global novelty word “first” is out of scope. | The source and official repository expose the agent, skill documents, and tool registry, so their internal consistency can be audited without model calls. |
| C2. SP-Mind obtains 68.9% average SP-Bench success, including 61.9% Advanced and 33.3% Challenging. | 0/2 expected. | The paper used `claude-sonnet-4-20250514` for three repetitions; the model is retired, the evaluation requires API-backed agent executions, and the repository does not provide the 306 original SP-Mind run traces/predictions or an automatic judge. |
| C3. SP-Bench contains 102 queries across 18 categories and eight stages, with tier counts 40 Basic, 28 Intermediate, 21 Advanced, and 13 Challenging. | 2/2 expected. | These are deterministic manifest properties and can be recomputed independently, including schema, uniqueness, stage/tier consistency, and placeholder checks. |
| C4. On CRC-CODEX quantification, SP-Mind reports Pearson 0.953, Spearman 0.902, correlation-matrix similarity 0.727, MMD 0.368, and spatial Spearman 0.513. | 0/2 expected. | The official dataset does not contain the CRC-CODEX evaluation inputs, original agent outputs, or complete ground-truth bundle needed to rerun the five metrics. |
| C5. On cell annotation, SP-Mind averages 0.681 CyteOnto GHK similarity and reaches 0.765 on `cHL_2_MIBI_5`. | 0/2 expected. | Original predictions are absent. A faithful rerun additionally requires repeated Claude-backed annotation plus Qwen3-Embedding-8B evaluation and the paper’s exact retired model. |

## Frozen total

- Conservative expectation: **4/10 claim points** (C1 implementation and C3
  benchmark construction).
- Plausible range: **2–4/10**.
- Stretch ceiling without author outputs or the original model: **4/10**.
- Expected local effort: **2–4 hours**, excluding the 16.1 GB full dataset and
  external API execution.

## Stop conditions

Do not claim numerical reproduction of C2, C4, or C5 if any of the following
remains true:

1. the paper-faithful model cannot be invoked;
2. original per-run predictions/traces are unavailable;
3. the corresponding inputs or ground truth are absent;
4. judging requires undocumented manual decisions;
5. only table values copied from the paper can be recovered.

Static source inspection or re-reading paper tables counts as artifact
verification, not experimental reproduction.
