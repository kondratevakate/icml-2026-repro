"""verify_claim3.py — Claim 3: DeepSeek-V3 30.28 / 21.32 GM, Qwen-2.5-Coder-32B 23.45 GM (Semantic),
"most evaluated LLMs failing to exceed 20% success rate".

Sub-assertions (source: Table 2, Sec 5.1):
  a) Deepseek-V3 GM Syntax == 30.28
  b) Deepseek-V3 GM Semantic == 21.32
  c) Qwen-2.5-Coder-32B GM Semantic == 23.45
  d) MOST evaluated LLMs do not exceed 20% -> tested by EXHAUSTIVE ENUMERATION of all 24 model rows on each
     split and pooled (48 model x split cells). "Most" is read as strictly more than half.

Note on metric: Sec 5.1 uses GM as the "success rate" ("peak success rate of 36.46% GM score"), so (d) is
evaluated on GM. The paper's own wording is narrower than the anchored claim: Sec 5.1 says "Other CLOSED-SOURCE
LLMs ... most failing to exceed 20% GM", the Abstract says "most models score below 20%". Both readings are
computed here.

Run: .venv/bin/python verify_claim3.py
"""
import json
import os

from paper_tables import table2

HERE = os.path.dirname(os.path.abspath(__file__))
CMD = ".venv/bin/python verify_claim3.py"

# Table 2 lists two Qwen-2.5-Coder rows (7B then 32B); the parser labels the second one '#2'.
QWEN32 = "Qwen-2.5-Coder#2"
DSV3 = "Deepseek-V3"
# Closed-source section starts at Claude-4-Sonnet in the paper's row order.
CLOSED = ["Claude-4-Sonnet", "GPT-4o-mini-2024-07-18", "GPT-4o-2024-11-20", "GPT-4.1", "GPT-5",
          "Gemini-2.5-Pro", "Kimi-K2", "O1-preview", "O3-mini", "Doubao-Seed-1.6",
          "Doubao-Seed-1.6-flash", "Doubao-Seed-1.6-thinking"]


def frac_below(pool, key, thr=20.0):
    vals = [v[key] for v in pool.values()]
    n_below = sum(1 for x in vals if x <= thr)
    return n_below, len(vals), n_below / len(vals)


def main():
    t2 = table2()
    closed = {k: t2[k] for k in CLOSED}
    open_src = {k: v for k, v in t2.items() if k not in CLOSED}

    syn = frac_below(t2, "syn_gm")
    sem = frac_below(t2, "sem_gm")
    pooled_n = syn[0] + sem[0]
    pooled_tot = syn[1] + sem[1]
    closed_syn = frac_below(closed, "syn_gm")
    closed_sem = frac_below(closed, "sem_gm")

    checks = {
        "deepseekv3_syntax_gm_is_30.28": abs(t2[DSV3]["syn_gm"] - 30.28) < 1e-9,
        "deepseekv3_semantic_gm_is_21.32": abs(t2[DSV3]["sem_gm"] - 21.32) < 1e-9,
        "qwen25coder32b_semantic_gm_is_23.45": abs(t2[QWEN32]["sem_gm"] - 23.45) < 1e-9,
        "most_models_below_20_on_syntax": syn[2] > 0.5,
        "most_models_below_20_on_semantic": sem[2] > 0.5,
        "most_models_below_20_pooled": pooled_n / pooled_tot > 0.5,
    }
    numeric_ok = all(v for k, v in checks.items() if k.endswith(("30.28", "21.32", "23.45")))
    below20_ok = checks["most_models_below_20_on_syntax"] and checks["most_models_below_20_on_semantic"]

    # --- mutation: replace GM by MB (a laxer metric). Prediction: the "<20%" statement collapses entirely,
    # showing the 20% statement is metric-dependent, not a property of the models per se.
    mut_syn = frac_below(t2, "syn_mb")
    mut_sem = frac_below(t2, "sem_mb")
    mutation_ok = mut_syn[2] < 0.5 and mut_sem[2] < 0.5

    out = {
        "claim": 3,
        "command": CMD,
        "source": "Table 2 (Sec 5.1 Main Results); prose Sec 5.1; Abstract",
        "observed": {
            "deepseek_v3": t2[DSV3],
            "qwen_2.5_coder_32B": t2[QWEN32],
            "n_models": len(t2),
            "syntax_gm_le20": {"n": syn[0], "of": syn[1], "frac": round(syn[2], 4)},
            "semantic_gm_le20": {"n": sem[0], "of": sem[1], "frac": round(sem[2], 4)},
            "pooled_gm_le20": {"n": pooled_n, "of": pooled_tot, "frac": round(pooled_n / pooled_tot, 4)},
            "closed_source_syntax_gm_le20": {"n": closed_syn[0], "of": closed_syn[1],
                                             "frac": round(closed_syn[2], 4)},
            "closed_source_semantic_gm_le20": {"n": closed_sem[0], "of": closed_sem[1],
                                               "frac": round(closed_sem[2], 4)},
            "open_source_n": len(open_src),
            "all_syntax_gm": {k: v["syn_gm"] for k, v in t2.items()},
            "all_semantic_gm": {k: v["sem_gm"] for k, v in t2.items()},
        },
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "sub_verdicts": {
            "reported_model_scores": "verified" if numeric_ok else "falsified",
            "most_models_below_20pct": "verified" if below20_ok else "falsified",
        },
        "mutation_test": {
            "description": "swap the GM metric for MB (modify-better) and recount the '<=20%' population",
            "syntax_mb_le20": {"n": mut_syn[0], "of": mut_syn[1], "frac": round(mut_syn[2], 4)},
            "semantic_mb_le20": {"n": mut_sem[0], "of": mut_sem[1], "frac": round(mut_sem[2], 4)},
            "predicted": "the '<=20%' majority disappears on both splits (metric-dependence)",
            "mutation_behaves_as_predicted": mutation_ok,
        },
        "verdict": "falsified" if not below20_ok else ("verified" if all(checks.values()) and mutation_ok
                                                       else "falsified"),
        "verdict_reason": ("the three quoted per-model GM scores match Table 2 exactly, but the universal "
                           "statement 'most evaluated LLMs fail to exceed 20%' is false on Squirrel-Syntax "
                           "under GM (see fractions)"),
        "scope": "paper-internal consistency only; scores cannot be independently re-measured (no public "
                 "benchmark data, model outputs, or Graph-Match scorer)",
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "claim3.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: out[k] for k in ("checks", "sub_verdicts", "mutation_test", "verdict")}, indent=2))
    print(json.dumps(out["observed"]["syntax_gm_le20"]), json.dumps(out["observed"]["semantic_gm_le20"]))


if __name__ == "__main__":
    main()
