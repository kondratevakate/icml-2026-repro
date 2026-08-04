# notes_paper.md — Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL SQL? (arXiv 2601.18119v2)

Read ONCE from paper/paper.txt (28p, extracted from arXiv PDF). This file is the only paper source afterwards.

## Artifacts / availability
- No GitHub / HuggingFace / dataset release link anywhere in the paper (grep: github|huggingface|available at|release
  hits only citations + LLaMA-Factory footnote at line 1949). Squirrel Benchmark data, model outputs and the
  GM/EM/MB scorer are NOT public. Enterprise seed SQL is proprietary (ByteDance).

## Claim -> source location
- C1 (469/516, 1000+ seeds, 26 scenarios, >140 lines, AST width >11, depth >8.7)
  - Abstract (lines 19-26); §3.1 "The final seed corpus contains 1,000+ SQL scripts spanning 26 business scenarios,
    averaging over 120 lines with AST depth > 8 and width > 12" (l.324-329) — note: SEED corpus stats, not benchmark.
  - Table 1 (§4, l.596-686) gives the benchmark-level numbers.
- C2/C3 (model GM scores): §5.1 Main Results, Table 2 (l.825-1077) + prose l.1078-1086.
- C4 (420+ tokens, 17.34 / 21.62 functions per script): §4 "Complexity of SQL Scripts" (l.577-592) + Table 1.

## Table 1 (relevant rows) — # Test Ex. | Tok/SQL | Line/SQL | Func/SQL | AST Depth | AST Width
- Squirrel-Syntax   : 469 | 496.90 | 163.69 | 21.62 | 8.93 | 11.69
- Squirrel-Semantic : 516 | 425.93 | 141.58 | 17.34 | 8.75 | 11.12
- BIRD-Critic-open  : 600 | 49.18 | 9.73 | 4.30 | 8.03 | 6.01

## Table 2 (EM/GM/MB per split), key rows
- Claude-4-Sonnet (closed): Syntax 23.88 / 36.46 / 68.02 ; Semantic 31.78 / 32.17 / 43.69   (bold/best)
- Deepseek-V3 685B: Syntax 17.91 / 30.28 / 60.34 ; Semantic 11.24 / 21.32 / 33.27
- Qwen-2.5-Coder 32B: Syntax 12.79 / 20.26 / 52.88 ; Semantic 17.44 / 23.45 / 34.69
- Doubao-Seed-1.6: Syntax 19.19 / 30.92 / 64.39 ; Semantic 16.09 / 20.93 / 32.82
- 24 evaluated LLM rows total (12 open source + 12 closed source) + 3 SFT variants of Qwen-2.5-Coder-7B
  (+SFT, +diff-SFT, +DM-SFT) listed separately (not "evaluated LLMs" of the survey).

## Metrics (§3.5, App. D.1.2)
- EM: string identity. GM: match of optimized ASTs of predicted vs reference SQL. MB: edit-distance improvement.
- Evaluation is execution-free. "Success rate" in the abstract == GM score (prose l.1080 uses "peak success rate of
  36.46% GM score").

## Internal inconsistency found while reading (relevant to C2)
- Abstract (l.30): Claude-4-Sonnet 32.17% on Squirrel-Semantic.
- Introduction (l.165): "36.46% success on Squirrel-Syntax and 33.17% on Squirrel-Semantic".
- Table 2 + §5.1 prose: 32.17. => the Intro's 33.17 contradicts the table (33.17 also appears at l.1152 as an SFT gain).

## Prose claims to test against the table
- Abstract: "most models score below 20%".
- §5.1: "Other closed-source LLMs perform even worse, with most failing to exceed 20% GM."
