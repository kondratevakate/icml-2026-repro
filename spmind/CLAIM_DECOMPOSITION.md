# Claim decomposition

## C1 — agent implementation

The paper presents SP-Mind as an autonomous LLM-driven agent that uses a
ReAct-style loop, expert-curated skills, and more than ten computational tools
to cover eight spatial-proteomics stages.

Independent target: inspect the executable agent path, model/tool interface,
skill inventory, and tool registry. The priority claim “first” is not evaluated.

## C2 — SP-Bench performance

Reported headline values:

- overall: 68.9%;
- Advanced: 61.9%;
- Challenging: 33.3%;
- strongest specialized baselines: Biomni-SP 55.4%, ToolUniverse-SP 50.5%.

Independent target: a faithful result would require three runs of all 102 tasks
under the paper configuration and success criteria. Merely reproducing the
published table from TeX/PDF is insufficient.

## C3 — SP-Bench construction

Reported composition:

- 102 queries;
- 18 categories;
- eight pipeline stages;
- tiers: 40 Basic, 28 Intermediate, 21 Advanced, 13 Challenging;
- source domains include MCMICRO, MAPS cHL, and PDAC IMC.

Independent target: recompute all manifest statistics, validate record schema,
IDs, enum values, stage counts, placeholders, and query duplication without
calling the author’s benchmark-generation script.

## C4 — CRC-CODEX quantification

Reported SP-Mind means: Pearson 0.953, Spearman 0.902, correlation-matrix
similarity 0.727, MMD 0.368, and spatial Spearman 0.513.

Independent target: recompute all five metrics from original agent outputs and
ground truth. Paper/source concordance alone is not a rerun.

## C5 — cell annotation

Reported CyteOnto GHK similarity: 0.681 average, with 0.765 on
`cHL_2_MIBI_5`.

Independent target: run the original predictions through the specified
Qwen3-Embedding-8B/GHK evaluator, or rerun the complete annotation procedure if
the exact model and inputs are available.
