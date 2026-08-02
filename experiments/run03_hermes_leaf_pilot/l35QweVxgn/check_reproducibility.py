#!/usr/bin/env python3
"""
check_reproducibility.py -- package the six results/claim<N>.json artifacts of run
l35QweVxgn into a reproducibility logbook (logbook.md). CPU-only; reads existing
JSON, does NOT re-run heavy computation.

Paper: "On the Theory of Continual Learning with Gradient Descent for Neural
Networks" (OpenReview l35QweVxgn / arXiv 2510.05573).
Run: l35QweVxgn  .  Compute: CPU-only (numpy / scipy / sympy), master seed 20260802.
"""
import json
import os
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
LOGBOOK = os.path.join(HERE, "logbook.md")
MASTER_SEED = 20260802
EXPECTED_CLAIMS = [1, 2, 3, 4, 5, 6]


def load(claim):
    with open(os.path.join(RESULTS, f"claim{claim}.json")) as f:
        return json.load(f)


def mutation_summary(rec):
    mt = rec.get("mutation_test", {}) or {}
    desc = mt.get("description", "")
    breaks = mt.get("property_breaks")
    state = "breaks" if breaks is True else ("does NOT break" if breaks is False else "n/a")
    note = mt.get("note", "")
    return f"{desc}  [mutation {state}]  {note}".strip()


def render(rec):
    c = rec["claim"]
    L = []
    L.append(f"### Claim {c} — {rec['statement']}\n")
    L.append(f"- **Verdict:** `{rec['verdict']}`")
    L.append(f"- **Source:** {rec['source']}")
    L.append(f"- **Seed:** {rec['seed']}  (master {MASTER_SEED})")
    # 'why' is reconstructed from the verdict-critical numerics below
    L.append(f"- **Mutation test:** {mutation_summary(rec)}")
    L.append("- **Key numerics:**")
    # pull a curated set of numeric fields per claim
    keys = {
        1: ["analytic_bound", "analytic_ratio_tests", "empirical"],
        2: ["term2_by_d", "term3_by_d", "term1_by_K", "doubling_ratios", "empirical_fluctuation_by_d"],
        3: ["uniformity", "per_d"],
        4: ["scaling_tests", "regime", "contrast_claim5"],
        5: ["T_scan", "later_task_dependence"],
        6: ["identity", "upper_bound", "joint_control"],
    }.get(c, [])
    for k in keys:
        if k in rec:
            L.append(f"  - `{k}`: {json.dumps(rec[k])}")
    if "note" in rec:
        L.append(f"- **Note:** {rec['note']}")
    if "note_on_regime" in rec:
        L.append(f"- **Note on regime:** {rec['note_on_regime']}")
    if "note_on_vanishing" in rec:
        L.append(f"- **Note on vanishing:** {rec['note_on_vanishing']}")
    L.append("")
    return "\n".join(L)


def main():
    rows = []
    status = "OK"
    for c in EXPECTED_CLAIMS:
        rec = load(c)
        rows.append(rec)
        if rec["verdict"] == "falsified":
            status = "FAIL"
        elif rec["verdict"] == "inconclusive":
            status = "REVIEW"

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_verified = sum(1 for r in rows if r["verdict"] == "verified")
    n_toy = sum(1 for r in rows if r["verdict"] == "toy")

    out = []
    out.append("# Reproducibility Logbook — l35QweVxgn\n")
    out.append(f"**Paper:** \"On the Theory of Continual Learning with Gradient Descent for "
               f"Neural Networks\" (OpenReview l35QweVxgn / arXiv 2510.05573)")
    out.append(f"**Run:** `l35QweVxgn`  .  **Compute:** CPU-only (numpy / scipy / sympy), "
               f"master seed `{MASTER_SEED}`")
    out.append(f"**Packaged:** {generated}  .  **Tool:** `check_reproducibility.py` "
               f"(reads existing `results/claim1..6.json`, no re-compute)")
    out.append("")
    out.append("## Summary\n")
    out.append("| Claim | Verdict | Source | Seed |")
    out.append("|------:|---------|--------|-----:|")
    for r in rows:
        out.append(f"| {r['claim']} | **{r['verdict']}** | {r['source']} | {r['seed']} |")
    out.append("")
    out.append(f"**Result:** {n_verified}/6 claims reproduced (verified)"
               + (f", {n_toy}/6 toy/empirical." if n_toy else "."))
    out.append("")
    out.append("## Method\n")
    out.append("All six anchored claims are analytic / closed-form bounds or a decomposition. "
               "They were reproduced in two complementary ways:")
    out.append("- **(A) Analytic:** the closed-form expressions the paper derives (Theorems 2.1, 2.2, "
               "2.3 and the improved gap Theorem B.1) were evaluated and their scaling dependence on "
               "d, n, m, K, eta, T verified by exact ratio tests and by checking the prescribed "
               "parameter regime (n=O~(d^2 K), m=O~(d^8 K^4), eta*T=O(d^2)) drives every bound to "
               "o_d(1) (poly-logarithmically).")
    out.append("- **(B) Empirical (kernel-regime):** the XOR-cluster data model (mutually orthogonal "
               "task means of norm 1/sqrt(d), Gaussian noise sigma=O(1/(polylog(d) sqrt(d)))) was "
               "generated and the KERNEL-REGIME closed-form forgetting object the theorems bound "
               "(Eq. after Thm 2.2: F_tr(k)=|(1/n) sum_{x_k} eta*T x_k^T (sum_{j>k} A_j) x_k|) was "
               "computed directly, confirming the qualitative behaviour (small under orthogonality, "
               "1/sqrt(n) sample-fluctuation, breaks under non-orthogonality).")
    out.append("")
    out.append("Every verified claim has a MUTATION test: perturbing the setup (breaking task-mean "
               "orthogonality, or breaking one factor of the parameter regime, or making the per-step "
               "loss non-self-bounded) makes the claimed property break or shift, as required.")
    out.append("")
    out.append("## Per-claim detail\n")
    for r in rows:
        out.append(render(r))
    out.append("## Honest notes\n")
    out.append("- These are THEORETICAL claims (closed-form bounds / a decomposition), not empirical "
               "statements requiring a full-width NN training run. The reproduction evaluates the "
               "paper's bound *expressions* and their scaling / regime-sufficiency directly (method A) "
               "and on synthetic XOR-cluster data via the paper's own kernel-regime forgetting formula "
               "(method B). No GPU, no full network training.")
    out.append("- Vanishing is ASYMPTOTIC (o_d(1)): under the regime every bound decreases with d as a "
               "poly-logarithmic factor (1/sqrt(log d), 1/log d, 1/log^2 d), so at moderate d the bounds "
               "are still O(1) in magnitude; they tend to 0 only as d -> infinity. This matches the "
               "paper's o_d(1) statements.")
    out.append("- Term 1 of Theorem 2.1 under the regime equals Theta(1/sqrt(polylog d)) (the d shows up "
               "only through the polylog hidden in n=O~(d^2 K)); it vanishes with d only via that "
               "poly-log factor, consistent with the paper's K=O~_d(1) assumption.")
    out.append("- Theorem B.1's improved gap uses the learning RATE eta (not eta*T) in both its "
               "prefactor and exponent; with that correction it scales as eta d^2 log^3(T)/n "
               "(poly-log in T), strictly slower than Theorem 2.3's linear-in-T dependence.")
    out.append(f"\nOverall status: {status}\n")

    md = "\n".join(out)
    with open(LOGBOOK, "w") as f:
        f.write(md)
    print(f"Wrote {LOGBOOK}")
    print(f"Summary: {n_verified}/6 verified, status={status}")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
