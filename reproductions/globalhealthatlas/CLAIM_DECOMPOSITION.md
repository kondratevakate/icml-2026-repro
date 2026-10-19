# Claim decomposition

The paper is audited as six two-point claim groups.

## C1 — corpus scale and coverage

GlobalHealthAtlas contains 280,210 instances from more than 35,500 WHO IRIS
sources, spanning 15 public-health domains and 17 languages.

## C2 — corpus composition and quality control

The corpus has 138,267 QA and 141,943 single-choice instances, difficulty
proportions A/B/C of 26.26%/69.33%/4.41%, and splits of 247,599 training,
27,511 testing, and 5,100 evaluator-construction instances. A 14,010-example
expert audit reports a mean quality score of 4.503.

## C3 — evaluator validity and stability

The Public-Evaluator agrees strongly with independent expert judgments
(reported ICC 0.9735) and has favorable repeated-run stability across the
reported metrics.

## C4 — multilingual and multi-domain benchmark

The released benchmark results support the model-level, domain-level,
language-level, difficulty-level, and task-type comparisons in the main
benchmark tables.

## C5 — supervised fine-tuning and transfer

Domain-aligned supervised fine-tuning improves public-health reasoning across
model scales and yields useful transfer to other public-health, medical,
GPQA, and MMLU-Pro evaluations.

## C6 — robustness and leakage

The Public-Model remains comparatively stable under paraphrase, corruption,
and cross-lingual perturbations, while continuation tests report little
verbatim leakage.

