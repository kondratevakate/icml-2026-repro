"""verify_claim1.py — Claim 1: benchmark composition and script/AST complexity.

Sub-assertions (input_bundle.json claim 1, sources: Abstract; Sec 3.1; Table 1):
  a) 469 Squirrel-Syntax tasks, 516 Squirrel-Semantic tasks
  b) built from over 1,000 seed SQL scripts spanning 26 business scenarios
  c) scripts average over 140 lines
  d) AST width > 11
  e) AST depth > 8.7

(a),(c),(d),(e) are checked against Table 1 as parsed from the paper text.
(b) is a prose statement in Sec 3.1 (no table) -> checked by locating the sentence, not by recomputation.

MUTATION TEST: the mechanism under test is "the Squirrel rows of Table 1 clear the thresholds stated in the
claim". Break it by substituting the BIRD-Critic-open row (the strongest prior SQL-debugging benchmark) for the
Squirrel rows; every complexity threshold must then fail. If the checks passed for BIRD-Critic too, they would be
vacuous.

Run: .venv/bin/python verify_claim1.py
"""
import json
import os
import re

from paper_tables import PAPER, table1

HERE = os.path.dirname(os.path.abspath(__file__))
CMD = ".venv/bin/python verify_claim1.py"


def thresholds(row):
    return {
        "lines_gt_140": row["line"] > 140,
        "ast_width_gt_11": row["width"] > 11,
        "ast_depth_gt_8.7": row["depth"] > 8.7,
    }


def main():
    t1 = table1()
    syn, sem = t1["Squirrel-Syntax"], t1["Squirrel-Semantic"]

    txt = open(PAPER, encoding="utf-8", errors="replace").read()
    prose = re.search(r"contains 1,000\+ SQL scripts span-?\s*ning 26 business scenarios", txt) is not None

    checks = {
        "n_syntax_is_469": syn["n_test"] == 469,
        "n_semantic_is_516": sem["n_test"] == 516,
        "seed_1000plus_26_scenarios_sentence_present": prose,
    }
    for split, row in (("syntax", syn), ("semantic", sem)):
        for k, v in thresholds(row).items():
            checks[f"{split}_{k}"] = v

    # --- mutation: swap in BIRD-Critic-open (a short-query benchmark) ---
    mut_row = t1["BIRD-Critic-open (Li et al., 2026)"]
    mutation = thresholds(mut_row)
    mutation_ok = not any(mutation.values())  # all thresholds must fail

    out = {
        "claim": 1,
        "command": CMD,
        "source": "Table 1 (Sec 4); Abstract; Sec 3.1",
        "observed": {
            "Squirrel-Syntax": syn,
            "Squirrel-Semantic": sem,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "mutation_test": {
            "description": "replace Squirrel rows with BIRD-Critic-open row",
            "mutant_row": mut_row,
            "thresholds_under_mutation": mutation,
            "predicted": "all complexity thresholds fail",
            "mutation_behaves_as_predicted": mutation_ok,
        },
        "verdict": "verified" if all(checks.values()) and mutation_ok else "falsified",
        "scope": "paper-internal consistency; benchmark corpus is not public so the statistics "
                 "cannot be independently recomputed",
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "claim1.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
