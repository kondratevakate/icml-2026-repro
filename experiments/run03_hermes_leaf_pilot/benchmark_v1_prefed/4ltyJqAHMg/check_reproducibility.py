"""check_reproducibility.py — reproducibility gate for logbook.md.

Re-asserts EVERY number quoted in logbook.md against results/claim*.json.
Exit code 0 = the logbook is backed by the stored results; 1 = drift.

Run: .venv/bin/python check_reproducibility.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")
LOG = os.path.join(HERE, "logbook.md")

fails = []
oks = 0


def eq(label, got, want, tol=1e-9):
    global oks
    ok = (got == want) if isinstance(want, (bool, str)) else abs(got - want) <= tol
    if ok:
        oks += 1
    else:
        fails.append(f"{label}: logbook says {want!r}, results say {got!r}")


def load(n):
    with open(os.path.join(R, f"claim{n}.json")) as fh:
        return json.load(fh)


c1, c2, c3, c4 = (load(i) for i in (1, 2, 3, 4))

# ---------------- claim 1 ----------------
s, m = c1["observed"]["Squirrel-Syntax"], c1["observed"]["Squirrel-Semantic"]
eq("c1 syntax n", s["n_test"], 469)
eq("c1 semantic n", m["n_test"], 516)
eq("c1 syntax lines", s["line"], 163.69)
eq("c1 semantic lines", m["line"], 141.58)
eq("c1 syntax depth", s["depth"], 8.93)
eq("c1 semantic depth", m["depth"], 8.75)
eq("c1 syntax width", s["width"], 11.69)
eq("c1 semantic width", m["width"], 11.12)
eq("c1 all checks pass", c1["all_checks_pass"], True)
eq("c1 verdict", c1["verdict"], "verified")
eq("c1 mutation", c1["mutation_test"]["mutation_behaves_as_predicted"], True)
mr = c1["mutation_test"]["mutant_row"]
eq("c1 mutant lines", mr["line"], 9.73)
eq("c1 mutant depth", mr["depth"], 8.03)
eq("c1 mutant width", mr["width"], 6.01)

# ---------------- claim 2 ----------------
cl = c2["observed"]["claude_row"]
eq("c2 claude syn EM", cl["syn_em"], 23.88)
eq("c2 claude syn GM", cl["syn_gm"], 36.46)
eq("c2 claude syn MB", cl["syn_mb"], 68.02)
eq("c2 claude sem EM", cl["sem_em"], 31.78)
eq("c2 claude sem GM", cl["sem_gm"], 32.17)
eq("c2 claude sem MB", cl["sem_mb"], 43.69)
eq("c2 n models", c2["observed"]["n_models_enumerated"], 24)
eq("c2 argmax syn", c2["observed"]["argmax_syntax_gm"][0], "Claude-4-Sonnet")
eq("c2 argmax sem", c2["observed"]["argmax_semantic_gm"][0], "Claude-4-Sonnet")
eq("c2 runner-up syn", c2["observed"]["runner_up_syntax_gm"], 30.92)
eq("c2 runner-up sem", c2["observed"]["runner_up_semantic_gm"], 28.68)
eq("c2 intro 33.17 inconsistency", c2["internal_inconsistency"]["intro_states_33.17_for_semantic"], True)
eq("c2 mutation A target", c2["mutation_test"]["A_leave_one_out"]["argmax_syntax_gm"][0], "Doubao-Seed-1.6")
eq("c2 mutation A sem target", c2["mutation_test"]["A_leave_one_out"]["argmax_semantic_gm"][0], "O3-mini")
eq("c2 mutation C target", c2["mutation_test"]["C_metric_and_pool_swap"]["argmax_syntax_em"][0], "+ DM-SFT")
eq("c2 mutation C value", c2["mutation_test"]["C_metric_and_pool_swap"]["argmax_syntax_em"][1], 27.27)
eq("c2 mutation overall", c2["mutation_test"]["mutation_behaves_as_predicted"], True)
eq("c2 verdict", c2["verdict"], "verified")

# ---------------- claim 3 ----------------
o = c3["observed"]
eq("c3 dsv3 syn", o["deepseek_v3"]["syn_gm"], 30.28)
eq("c3 dsv3 sem", o["deepseek_v3"]["sem_gm"], 21.32)
eq("c3 qwen32b sem", o["qwen_2.5_coder_32B"]["sem_gm"], 23.45)
eq("c3 syntax <=20 count", o["syntax_gm_le20"]["n"], 24 - 14)          # 10
eq("c3 syntax <=20 of", o["syntax_gm_le20"]["of"], 24)
eq("c3 syntax <=20 frac", o["syntax_gm_le20"]["frac"], 0.4167, tol=5e-5)
eq("c3 semantic <=20 count", o["semantic_gm_le20"]["n"], 15)
eq("c3 semantic <=20 frac", o["semantic_gm_le20"]["frac"], 0.625, tol=5e-5)
eq("c3 pooled <=20 count", o["pooled_gm_le20"]["n"], 25)
eq("c3 pooled <=20 of", o["pooled_gm_le20"]["of"], 48)
eq("c3 pooled <=20 frac", o["pooled_gm_le20"]["frac"], 0.5208, tol=5e-5)
eq("c3 closed syn <=20 count", o["closed_source_syntax_gm_le20"]["n"], 6)
eq("c3 closed syn <=20 of", o["closed_source_syntax_gm_le20"]["of"], 12)
eq("c3 mutation syn MB<=20", c3["mutation_test"]["syntax_mb_le20"]["n"], 3)
eq("c3 mutation syn MB frac", c3["mutation_test"]["syntax_mb_le20"]["frac"], 0.125, tol=5e-5)
eq("c3 mutation sem MB<=20", c3["mutation_test"]["semantic_mb_le20"]["n"], 6)
eq("c3 mutation sem MB frac", c3["mutation_test"]["semantic_mb_le20"]["frac"], 0.25, tol=5e-5)
eq("c3 mutation overall", c3["mutation_test"]["mutation_behaves_as_predicted"], True)
eq("c3 sub verdict scores", c3["sub_verdicts"]["reported_model_scores"], "verified")
eq("c3 sub verdict most<20", c3["sub_verdicts"]["most_models_below_20pct"], "falsified")
eq("c3 verdict", c3["verdict"], "falsified")

# ---------------- claim 4 ----------------
o4 = c4["observed"]
eq("c4 syntax tokens", o4["Squirrel-Syntax_tokens"], 496.90)
eq("c4 semantic tokens", o4["Squirrel-Semantic_tokens"], 425.93)
eq("c4 syntax func", o4["Squirrel-Syntax_func"], 21.62)
eq("c4 semantic func", o4["Squirrel-Semantic_func"], 17.34)
eq("c4 max prior tokens", o4["max_tokens_among_prior_benchmarks"], 154.63)
eq("c4 all checks pass", c4["all_checks_pass"], True)
eq("c4 mutation", c4["mutation_test"]["mutation_behaves_as_predicted"], True)
eq("c4 verdict", c4["verdict"], "verified")

# ---------------- logbook text cross-check ----------------
txt = open(LOG, encoding="utf-8").read()
for token in ["469", "516", "163.69", "141.58", "8.93", "8.75", "11.69", "11.12",
              "36.46", "32.17", "33.17", "30.92", "28.68", "27.27",
              "30.28", "21.32", "23.45", "41.67", "62.5", "52.08",
              "496.90", "425.93", "21.62", "17.34", "154.63"]:
    if token not in txt:
        fails.append(f"logbook.md is missing quoted number {token}")
    else:
        oks += 1

print(f"{oks} assertions passed, {len(fails)} failed")
for f in fails:
    print("  FAIL:", f)
sys.exit(1 if fails else 0)
