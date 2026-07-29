# SP-Mind independent artifact reproduction

This CPU-only bundle audits five claims from *SP-Mind: An Autonomous Reasoning
Agent for Spatial Proteomics Analysis* (arXiv `2606.24235`, OpenReview
`UJcB3XrffF`).

## Result

- C1 `PARTIALLY VERIFIED FOR RELEASED ARCHITECTURE`: the pinned repository exposes an
  SDK-driven autonomous agent, 19 uniquely registered tools, eight skill
  documents, and implementations for all eight SP-Bench stages. This supports
  the released wiring, but not a full autonomous end-to-end execution.
- C2 `INCONCLUSIVE`: the 68.9% execution accuracy requires 306 calls to a
  retired Claude model and unreleased original traces.
- C3 `VERIFIED`: an independent parser confirms 102 unique tasks, 18
  categories, eight stages, and the declared 40/28/21/13 tier split.
- C4 `INCONCLUSIVE`: CRC-CODEX inputs and generated outputs are absent.
- C5 `INCONCLUSIVE`: ground truth is available, but original predictions are
  absent. The released evaluator is also order-sensitive when files have equal
  lengths, which is recorded as a qualification rather than a claim verdict.

Prepared score: `3/10`.

## Run

Fetch and pin the official repository and arXiv source, then run the audit:

```bash
python prepare_official.py
python audit_artifacts.py
python -m unittest discover -s . -p "test_*.py" -v
```

The artifact audit is deterministic and makes no network or LLM calls. It
parses the benchmark, Python registry, implementation symbols, skills, and
paper anchors independently.

## Integrity

No leaderboard entries or peer reproductions were used. The publishable
bundle contains no biomedical images, cell-level records, author code, paper
source, model outputs, API keys, or private traces.
