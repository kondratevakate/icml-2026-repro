# Independent checks and verdicts

Date: 2026-07-29

## Outcome

| Claim | Verdict | Points |
|---|---|---:|
| C1 released agent/tool architecture | PARTIALLY VERIFIED FOR RELEASED IMPLEMENTATION | 1 |
| C2 SP-Bench performance | INCONCLUSIVE — NOT RERUN | 0 |
| C3 SP-Bench composition | VERIFIED | 2 |
| C4 CRC-CODEX metrics | INCONCLUSIVE — REQUIRED ARTIFACTS ABSENT | 0 |
| C5 annotation metrics | INCONCLUSIVE — PREDICTIONS ABSENT | 0 |
| **Total** |  | **3/10** |

The frozen forecast was 4/10. Independent preflight reduced C1 by one point,
so the final prepared result is 3/10.

## C1

Two independent routes agree:

1. AST inspection derives 19 unique tool descriptions from the nine registered
   tool modules and finds a top-level implementation for every description.
2. A clean editable installation builds the MCP server and returns the same 19
   fully qualified tool names at runtime.

All eight stages named by the benchmark map to present registered modules.
Eight expert skill Markdown files are present. The agent constructs
`ClaudeAgentOptions`, supplies a stepwise data-exploration/tool-execution
prompt, and consumes the SDK `query()` stream.

Scope qualification: the released repository delegates the iterative loop to
Claude Agent SDK; it does not implement a separate local ReAct state machine.
The global novelty word “first” is not verified.

## C2

The table values are concordant between the rendered PDF and frozen TeX, but
that is not reproduction. A faithful run is stopped because:

- the paper model `claude-sonnet-4-20250514` is retired;
- the local environment has no `ANTHROPIC_API_KEY`;
- the original 306 SP-Mind task-run traces are not published;
- no automatic implementation of the stated success criteria is included;
- the container runtime is installed but its daemon is not running.

## C3

The independent parser does not call `create_benchmark.py`. It validates every
JSONL record, exact schema, IDs, normalized query uniqueness, stage lengths,
tier rules, and placeholders.

Results:

- 102 records and 102 unique normalized queries;
- 18 categories;
- eight stages and 231 total stage invocations;
- Basic 40, Intermediate 28, Advanced 21, Challenging 13;
- no schema or tier/stage errors.

One non-fatal data warning was found:
`CHALLENGING_PIPELINEs` uses a lowercase trailing `s` while all other category
labels are uppercase.

## C4

No numerical check was attempted. The required CRC-CODEX source images,
paper-run quantification outputs, and complete matching ground truth are not
available as one official reproducible bundle.

## C5

The four official ground-truth CSVs pass their required three-column schema and
contain 2,013,525 rows in total. The released evaluator passes unit checks for
GHK, cluster-mode aggregation, and inconsistent prediction labels.

An adversarial row-order check finds that equal-length files are aligned
positionally. Reversing two correct prediction rows changes the synthetic
oracle GHK from 1.0 to `exp(-8)`, approximately 0.000335. Therefore an eventual
rerun must preserve and verify row order. This is a robustness finding, not
evidence that the paper’s original predictions were misordered.
