# plan.md

## Nature of this paper
It is a **benchmark + LLM-evaluation** paper. There is no theorem, no algorithm with a closed form, and no
released artifact (no code repo, no dataset, no model outputs — verified by grep over the full paper text).

## Claim classification

| Claim | Content | Type | CPU-feasible? |
|---|---|---|---|
| 1 | Benchmark size (469/516), 1000+ seeds, 26 scenarios, >140 lines, AST width>11, depth>8.7 | data-artifact statistics | Recomputation: NO (data not released). Paper-internal audit: YES |
| 2 | Claude-4-Sonnet 36.46 / 32.17 GM, best of all evaluated models | LLM inference over 985 private tasks with a private scorer | Recomputation: NO. Audit of "best" over Table 2: YES (exhaustive) |
| 3 | DeepSeek-V3 30.28/21.32, Qwen-2.5-Coder-32B 23.45, "most LLMs <20%" | same | Recomputation: NO. Audit + exhaustive enumeration of all 24 model rows: YES |
| 4 | 420+ tokens, 17.34/21.62 functions per script | data-artifact statistics | Recomputation: NO. Audit: YES |

## Decision
No synthetic toy stand-in will be built (TASK.md hard rule). Independent reproduction of every claim requires
(a) the Squirrel benchmark SQL corpus and (b) API inference on ~30 commercial LLMs + the unreleased GM scorer.
Both are inaccessible => the *reproduction* verdict for the numbers themselves is `inconclusive`.

What IS executable and worth doing: a machine-checked audit that parses the paper's own tables out of
`paper/paper.txt` and tests every quantitative sub-assertion of each anchored claim against them, including the
*derived* assertions ("best among evaluated models", "most models fail to exceed 20%") by **exhaustive enumeration
over all 24 evaluated model rows x 2 splits** — no sampling, no seeds. These derived assertions are genuinely
falsifiable from the paper alone, and one of them fails.

Verdict vocabulary used in logbook.md:
- `verified`  = the claim's assertion is entailed by the paper's own reported table, checked programmatically,
                with a mutation test — and explicitly scoped as *paper-internal consistency*, not independent
                re-measurement (stated in Evidence boundary).
- `falsified` = the assertion contradicts the paper's own table.
- `inconclusive` = requires unavailable data/inference.

## Budget
Start 2026-08-01T22:56:40+04:00. Target: all four claims audited in < 1h. No dataset download possible.
