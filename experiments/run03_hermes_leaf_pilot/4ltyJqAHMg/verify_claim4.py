"""verify_claim4.py — Claim 4: token count and functions per script.

Sub-assertions (source: Sec 4 "Complexity of SQL Scripts" + Table 1):
  a) scripts contain on the order of 420+ tokens on average
  b) 17.34 (Semantic) and 21.62 (Syntax) functions per script
  c) this is far above prior Text-to-SQL / SQL-debugging benchmarks

MUTATION TEST: assert the same "420+ tokens and >=17 functions" predicate on every non-Squirrel benchmark row of
Table 1 that reports those cells. Prediction: all of them fail, i.e. the predicate discriminates Squirrel from the
prior benchmarks and is not trivially true.

Run: .venv/bin/python verify_claim4.py
"""
import json
import os

from paper_tables import table1

HERE = os.path.dirname(os.path.abspath(__file__))
CMD = ".venv/bin/python verify_claim4.py"


def predicate(row):
    if row["tok"] is None or row["func"] is None:
        return None
    return bool(row["tok"] >= 420 and row["func"] >= 17)


def main():
    t1 = table1()
    syn, sem = t1["Squirrel-Syntax"], t1["Squirrel-Semantic"]

    checks = {
        "syntax_tokens_ge_420": syn["tok"] >= 420,
        "semantic_tokens_ge_420": sem["tok"] >= 420,
        "semantic_func_is_17.34": abs(sem["func"] - 17.34) < 1e-9,
        "syntax_func_is_21.62": abs(syn["func"] - 21.62) < 1e-9,
        "func_range_17.34_to_21.62": abs(min(syn["func"], sem["func"]) - 17.34) < 1e-9
                                     and abs(max(syn["func"], sem["func"]) - 21.62) < 1e-9,
    }

    others = {k: v for k, v in t1.items() if not k.startswith("Squirrel")}
    mutation = {k: predicate(v) for k, v in others.items()}
    mutation_ok = not any(v for v in mutation.values() if v is not None)
    max_other_tok = max(v["tok"] for v in others.values() if v["tok"] is not None)

    out = {
        "claim": 4,
        "command": CMD,
        "source": "Table 1 (Sec 4); Sec 4 'Complexity of SQL Scripts'",
        "observed": {
            "Squirrel-Syntax_tokens": syn["tok"],
            "Squirrel-Semantic_tokens": sem["tok"],
            "Squirrel-Syntax_func": syn["func"],
            "Squirrel-Semantic_func": sem["func"],
            "max_tokens_among_prior_benchmarks": max_other_tok,
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "mutation_test": {
            "description": "apply the '>=420 tokens and >=17 functions' predicate to every prior benchmark row",
            "per_benchmark": mutation,
            "predicted": "no prior benchmark satisfies it",
            "mutation_behaves_as_predicted": mutation_ok,
        },
        "verdict": "verified" if all(checks.values()) and mutation_ok else "falsified",
        "scope": "paper-internal consistency; the SQL corpus is not public so token/function counts "
                 "cannot be independently recomputed",
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "claim4.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
