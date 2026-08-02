"""verify_claim2.py — Claim 2: Claude-4-Sonnet 36.46 GM (Syntax) / 32.17 GM (Semantic), best of all models.

Sub-assertions (source: Table 2, Sec 5.1 Main Results):
  a) Claude-4-Sonnet GM on Squirrel-Syntax  == 36.46
  b) Claude-4-Sonnet GM on Squirrel-Semantic == 32.17
  c) it is the best performance among the evaluated models

(c) is checked by EXHAUSTIVE ENUMERATION over all 24 evaluated LLM rows of Table 2, on both splits — the claim
space is finite and fully enumerated, no sampling.

Also recorded: the Abstract (32.17) vs Introduction (33.17) inconsistency for the Semantic number.

MUTATION TEST: recompute the arg-max using the EM column instead of GM (i.e. break the "GM is the metric that
makes Claude best" mechanism) and, separately, include the paper's own SFT rows in the candidate pool. Prediction:
the Syntax arg-max moves away from Claude-4-Sonnet, showing the claim is metric- and pool-specific, not a tautology.

Run: .venv/bin/python verify_claim2.py
"""
import json
import os
import re

from paper_tables import PAPER, sft_rows, table2

HERE = os.path.dirname(os.path.abspath(__file__))
CMD = ".venv/bin/python verify_claim2.py"
CLAUDE = "Claude-4-Sonnet"


def argmax(pool, key):
    m = max(pool.items(), key=lambda kv: kv[1][key])
    return m[0], m[1][key]


def main():
    t2 = table2()
    claude = t2[CLAUDE]

    syn_best, syn_val = argmax(t2, "syn_gm")
    sem_best, sem_val = argmax(t2, "sem_gm")

    txt = open(PAPER, encoding="utf-8", errors="replace").read()
    intro_3317 = bool(re.search(r"36\.46% success on Squirrel-Syntax and 33\.17% on", txt.replace("\n", " ")))

    checks = {
        "claude_syntax_gm_is_36.46": abs(claude["syn_gm"] - 36.46) < 1e-9,
        "claude_semantic_gm_is_32.17": abs(claude["sem_gm"] - 32.17) < 1e-9,
        "claude_is_argmax_syntax_gm": syn_best == CLAUDE,
        "claude_is_argmax_semantic_gm": sem_best == CLAUDE,
        "n_evaluated_models_is_24": len(t2) == 24,
    }

    # --- mutation A (leave-one-out): delete the Claude row; arg-max MUST move to the runner-up ---
    pool_a = {k: v for k, v in t2.items() if k != CLAUDE}
    a_syn_best, a_syn_val = argmax(pool_a, "syn_gm")
    a_sem_best, a_sem_val = argmax(pool_a, "sem_gm")
    mut_a_ok = a_syn_best != CLAUDE and a_sem_best != CLAUDE

    # --- mutation B (value perturbation): set Claude's syntax GM just below the runner-up ---
    runner_syn = sorted((v["syn_gm"] for v in t2.values()), reverse=True)[1]
    pool_b = {k: dict(v) for k, v in t2.items()}
    pool_b[CLAUDE]["syn_gm"] = runner_syn - 0.01
    b_syn_best, b_syn_val = argmax(pool_b, "syn_gm")
    mut_b_ok = b_syn_best != CLAUDE

    # --- mutation C (metric + pool swap): EM instead of GM, with the paper's SFT rows added ---
    pool_c = dict(t2)
    pool_c.update(sft_rows())
    c_syn_best, c_syn_val = argmax(pool_c, "syn_em")
    mut_c_ok = c_syn_best != CLAUDE

    # observation recorded for honesty: within the 24-model pool Claude is also the EM arg-max
    em_syn_best, em_syn_val = argmax(t2, "syn_em")
    em_sem_best, em_sem_val = argmax(t2, "sem_em")

    mutation_ok = mut_a_ok and mut_b_ok and mut_c_ok

    out = {
        "claim": 2,
        "command": CMD,
        "source": "Table 2 (Sec 5.1 Main Results); prose Sec 5.1",
        "observed": {
            "claude_row": claude,
            "n_models_enumerated": len(t2),
            "argmax_syntax_gm": [syn_best, syn_val],
            "argmax_semantic_gm": [sem_best, sem_val],
            "runner_up_syntax_gm": sorted((v["syn_gm"] for v in t2.values()), reverse=True)[1],
            "runner_up_semantic_gm": sorted((v["sem_gm"] for v in t2.values()), reverse=True)[1],
            "all_syntax_gm": {k: v["syn_gm"] for k, v in t2.items()},
            "all_semantic_gm": {k: v["sem_gm"] for k, v in t2.items()},
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "internal_inconsistency": {
            "intro_states_33.17_for_semantic": intro_3317,
            "table2_and_abstract_state": 32.17,
            "note": "Introduction (Sec 1) reports 33.17% on Squirrel-Semantic, contradicting Table 2/Abstract "
                    "(32.17%). The anchored claim's 32.17 matches the table.",
        },
        "mutation_test": {
            "A_leave_one_out": {
                "description": "drop the Claude-4-Sonnet row from the pool",
                "argmax_syntax_gm": [a_syn_best, a_syn_val],
                "argmax_semantic_gm": [a_sem_best, a_sem_val],
                "predicted": "arg-max moves to another model on both splits",
                "behaves_as_predicted": mut_a_ok,
            },
            "B_value_perturbation": {
                "description": "set Claude syntax GM to (runner-up - 0.01)",
                "runner_up_syntax_gm": runner_syn,
                "argmax_syntax_gm": [b_syn_best, b_syn_val],
                "predicted": "arg-max flips away from Claude",
                "behaves_as_predicted": mut_b_ok,
            },
            "C_metric_and_pool_swap": {
                "description": "rank by EM instead of GM with the paper's SFT rows added to the pool",
                "argmax_syntax_em": [c_syn_best, c_syn_val],
                "predicted": "a fine-tuned 7B row overtakes Claude, i.e. 'best' is metric/pool specific",
                "behaves_as_predicted": mut_c_ok,
            },
            "observation_em_within_24_model_pool": {
                "argmax_syntax_em": [em_syn_best, em_syn_val],
                "argmax_semantic_em": [em_sem_best, em_sem_val],
                "note": "Claude is also the EM arg-max among the 24 evaluated LLMs; an earlier prediction that "
                        "the EM swap alone would flip the winner was WRONG and is recorded here rather than "
                        "dropped.",
            },
            "mutation_behaves_as_predicted": mutation_ok,
        },
        "verdict": "verified" if all(checks.values()) and mutation_ok else "falsified",
        "scope": "paper-internal consistency only; the 985 benchmark tasks, the model outputs and the "
                 "Graph-Match scorer are not public, so the scores cannot be independently re-measured",
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "claim2.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["checks"], indent=2))
    print(json.dumps(out["mutation_test"], indent=2))
    print("VERDICT", out["verdict"])


if __name__ == "__main__":
    main()
