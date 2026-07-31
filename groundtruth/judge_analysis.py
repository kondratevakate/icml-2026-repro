#!/usr/bin/env python3
"""Judge-behaviour and inter-participant agreement analysis.

Question: when several participants reproduce the SAME paper, how much do their
verdicts differ -- and is that spread real work quality or judge noise?

Design:
  * Unit of analysis = (paper, claim_index) slot. Only papers where every
    submitted solution reports the same n_claims AND that equals the number of
    anchored claims are used -- otherwise claim k of one logbook is not claim k
    of another and the comparison is meaningless.
  * Variance decomposition: between-paper vs within-paper (same paper, different
    participants). If spread were driven by paper difficulty alone, within-paper
    variance would be ~0.
  * Per-claim agreement: Fleiss' kappa over the 4 verdict categories, plus the
    modal-verdict share.
  * Judge-drift probe: does a solution's score depend on WHEN it was judged, and
    on how many logbooks the judge had already seen for that paper (anchoring)?

Writes groundtruth/out/judge_analysis.json
"""
import collections, json, math, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = f"{HERE}/data", f"{HERE}/out"
CATS = ["verified", "falsified", "toy", "inconclusive"]
PTS = {"verified": 2, "falsified": 2, "toy": 1, "inconclusive": 0}
MIN_SOL = 3


def fleiss_kappa(rows):
    """rows: list of category-count vectors, one per item; equal raters per item."""
    rows = [r for r in rows if sum(r) >= 2]
    if not rows:
        return None
    n = sum(rows[0])
    if any(sum(r) != n for r in rows):  # unequal raters -> fall back to mean pairwise
        return None
    N = len(rows)
    p_j = [sum(r[j] for r in rows) / (N * n) for j in range(len(CATS))]
    P_i = [(sum(c * c for c in r) - n) / (n * (n - 1)) for r in rows]
    P_bar, P_e = sum(P_i) / N, sum(p * p for p in p_j)
    if P_e >= 1:
        # every rating fell in one category: chance agreement is total.
        # Perfect observed agreement is then 1.0 by convention; anything less is 0.
        return 1.0 if P_bar >= 1 - 1e-12 else 0.0
    return (P_bar - P_e) / (1 - P_e)


def main():
    sols = json.load(open(f"{OUT}/solutions_by_paper.json"))
    anchored = json.load(open(f"{DATA}/claims_anchored.json"))

    aligned = {}
    for orid, es in sols.items():
        if len(es) < MIN_SOL:
            continue
        ns = {int(e["n_claims"]) for e in es}
        if len(ns) != 1:
            continue
        n = ns.pop()
        if n != len(anchored.get(orid, [])) or n == 0:
            continue
        if any(len(e["verdicts"]) != n for e in es):
            continue
        aligned[orid] = sorted(es, key=lambda e: e["judged_at"] or "")

    # ---- per-claim slots -----------------------------------------------------
    slots = []           # one entry per (paper, claim_idx)
    kappa_rows = collections.defaultdict(list)   # n_raters -> count vectors
    for orid, es in aligned.items():
        n = len(es[0]["verdicts"])
        for k in range(n):
            vs = [e["verdicts"][k] for e in es]
            cnt = [vs.count(c) for c in CATS]
            mode = max(CATS, key=lambda c: vs.count(c))
            slots.append({
                "orid": orid, "claim": k, "raters": len(vs),
                "unanimous": len(set(vs)) == 1,
                "modal_share": round(vs.count(mode) / len(vs), 4),
                "modal": mode,
                "counts": dict(zip(CATS, cnt)),
                "points": [PTS[v] for v in vs],
            })
            kappa_rows[len(vs)].append(cnt)

    kappas = {k: fleiss_kappa(v) for k, v in sorted(kappa_rows.items()) if len(v) >= 10}
    # A single pooled kappa is undefined across unequal rater counts, so report the
    # slot-weighted mean of the per-bucket kappas instead of silently dropping it.
    wk = [(len(kappa_rows[k]), v) for k, v in kappas.items() if v is not None]
    kappa_pooled = (round(sum(n * v for n, v in wk) / sum(n for n, _ in wk), 4)
                    if wk else None)

    # ---- variance decomposition on per-claim points --------------------------
    grand = statistics.mean(p for s in slots for p in s["points"])
    within = statistics.mean(statistics.pvariance(s["points"]) for s in slots)
    between = statistics.pvariance([statistics.mean(s["points"]) for s in slots])
    total = within + between

    # same, at paper level (normalized score per solution)
    paper_means, paper_within = [], []
    for orid, es in aligned.items():
        rates = [e["points"] / e["max_points"] for e in es if e["max_points"]]
        if len(rates) >= 2:
            paper_means.append(statistics.mean(rates))
            paper_within.append(statistics.pvariance(rates))

    # ---- judge drift ---------------------------------------------------------
    # within each paper, does the k-th submission judged score differently?
    by_order = collections.defaultdict(list)
    for orid, es in aligned.items():
        for i, e in enumerate(es):
            if e["max_points"]:
                by_order[min(i, 5)].append(e["points"] / e["max_points"])
    drift = {f"submission_{k}": {"n": len(v), "mean_rate": round(statistics.mean(v), 4)}
             for k, v in sorted(by_order.items())}

    # verdict mix by judging month
    by_month = collections.defaultdict(collections.Counter)
    for orid, es in aligned.items():
        for e in es:
            m = (e["judged_at"] or "")[:7]
            by_month[m].update(e["verdicts"])
    months = {m: {c: round(cnt[c] / sum(cnt.values()), 4) for c in CATS}
              | {"n_claims": sum(cnt.values())}
              for m, cnt in sorted(by_month.items()) if sum(cnt.values()) >= 50}

    # ---- disagreement hotspots ----------------------------------------------
    split = [s for s in slots if s["modal_share"] <= 0.5 and s["raters"] >= 4]
    split.sort(key=lambda s: (s["modal_share"], -s["raters"]))

    summary = {
        "papers_analyzed": len(aligned),
        "solutions": sum(len(v) for v in aligned.values()),
        "claim_slots": len(slots),
        "unanimous_slots": sum(s["unanimous"] for s in slots),
        "unanimous_share": round(sum(s["unanimous"] for s in slots) / len(slots), 4),
        "mean_modal_share": round(statistics.mean(s["modal_share"] for s in slots), 4),
        "fleiss_kappa_pooled": kappa_pooled,
        "fleiss_kappa_slots_covered": sum(n for n, _ in wk),
        "fleiss_kappa_by_rater_count": {k: (round(v, 4) if v is not None else None)
                                        for k, v in kappas.items()},
        "grand_mean_points": round(grand, 4),
        "variance_within_paper_claim": round(within, 4),
        "variance_between_paper_claim": round(between, 4),
        "within_share_of_total": round(within / total, 4) if total else None,
        "paper_level_within_variance": round(statistics.mean(paper_within), 5) if paper_within else None,
        "paper_level_between_variance": round(statistics.pvariance(paper_means), 5) if paper_means else None,
        "judge_order_effect": drift,
        "verdict_mix_by_month": months,
        "fully_split_slots": len(split),
    }

    json.dump({"summary": summary, "slots": slots, "hotspots": split[:40]},
              open(f"{OUT}/judge_analysis.json", "w"), indent=1)

    for k, v in summary.items():
        print(f"{k:34} {v}")
    return summary


if __name__ == "__main__":
    main()
