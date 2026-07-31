#!/usr/bin/env python3
"""Build a reproducibility ground-truth corpus from the ICML-2026-repro judged data.

Two strata:
  --mode all      : every paper with >=1 judged solution (multi-domain)
  --mode medical  : only the 703 medical/life-science papers (Kate's validation domain)

For each paper we derive a *consensus ground truth* per claim by aggregating the
participants' verdicts. A claim slot is included when enough participants judged it
and they agree beyond chance; slots with no agreement are flagged `disputed` (not
dropped) so the corpus is honest about where ground truth is weak.

Outputs (under out/):
  corpus_all.json / corpus_medical.json
  - papers: [{orid, area, n_solutions, consensus:[...], disputed:[...],
              negation_rate, zero_point_solutions, best_rate}]
  - strata_summary: verdict mix, consensus coverage, dispute rate, cost proxies
  - disputed_slots: the V-vs-F splits (rare falsification class)

The rare FALSIFIED class is tracked separately so a downstream model-eval knows
where the signal is thin.
"""
import argparse, csv, json, os, statistics, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, DATA, AUD = f"{HERE}/out", f"{HERE}/data", f"{HERE}/audit_snapshot"
CATS = ["verified", "falsified", "toy", "inconclusive"]
MIN_PART = 3          # need >=3 participants to call a consensus
AGREE = 0.6           # >=60% on one category to declare consensus


def stratify(rows):
    """Map orid -> (area, subarea) from the audit CSV."""
    return {r["orid"]: (r.get("area", ""), r.get("subarea", "")) for r in rows}


def build(sols, area_of, restrict=None):
    papers = []
    disputed = []
    for orid, es in sols.items():
        if restrict is not None and orid not in restrict:
            continue
        if not es:
            continue
        n_claims = int(es[0]["n_claims"])
        if n_claims == 0:
            continue
        # align verdict vectors to the same length (defensive; analysis already filters)
        vecs = [e["verdicts"] for e in es if len(e["verdicts"]) == n_claims]
        if not vecs:
            continue
        area, sub = area_of.get(orid, ("", ""))
        consensus, soft, disp = [], [], []
        for k in range(n_claims):
            vs = [v[k] for v in vecs]
            cnt = collections.Counter(vs)
            top, topn = cnt.most_common(1)[0]
            share = topn / len(vs)
            if len(vs) >= MIN_PART and share >= AGREE:
                consensus.append(top)
                soft.append(None)            # hard label only
            else:
                consensus.append("disputed")
                soft.append({c: n / len(vs) for c, n in cnt.items()})
                disp.append(k)
            if cnt["verified"] and cnt["falsified"]:
                disputed.append({
                    "orid": orid, "claim": k, "raters": len(vs),
                    "counts": dict(cnt), "area": area,
                })
        rates = [e["points"] / e["max_points"] for e in es if e["max_points"]]
        zero = sum(1 for e in es if e["points"] == 0)
        papers.append({
            "orid": orid,
            "area": area,
            "subarea": sub,
            "n_solutions": len(vecs),
            "consensus": consensus,
            "soft_gt": soft,                 # per-claim empirical distribution (None if hard)
            "n_consensus": sum(1 for c in consensus if c != "disputed"),
            "n_soft": sum(1 for c in consensus if c == "disputed"),
            "disputed_claims": disp,
            "n_claims": n_claims,
            "consensus_rate": round(sum(1 for c in consensus if c != "disputed") / n_claims, 4),
            "soft_rate": round(sum(1 for c in consensus if c == "disputed") / n_claims, 4),
            "zero_point_solutions": zero,
            "best_rate": round(max(rates), 4) if rates else None,
            "mean_rate": round(statistics.mean(rates), 4) if rates else None,
        })
    return papers, disputed


def summarize(papers, disputed):
    tot_claims = sum(p["n_claims"] for p in papers)
    cons_claims = sum(p["n_consensus"] for p in papers)
    vc = collections.Counter()
    for p in papers:
        for c in p["consensus"]:
            if c != "disputed":
                vc[c] += 1
    by_area = collections.defaultdict(lambda: {"papers": 0, "claims": 0, "consensus": 0})
    for p in papers:
        a = by_area[p["area"]]
        a["papers"] += 1
        a["claims"] += p["n_claims"]
        a["consensus"] += p["n_consensus"]
    return {
        "papers": len(papers),
        "total_claims": tot_claims,
        "consensus_claims": cons_claims,
        "soft_claims": tot_claims - cons_claims,
        "consensus_coverage": round(cons_claims / tot_claims, 4) if tot_claims else 0,
        "soft_coverage": round((tot_claims - cons_claims) / tot_claims, 4) if tot_claims else 0,
        "disputed_claims": tot_claims - cons_claims,
        "verdict_mix_consensus": {c: round(vc[c] / cons_claims, 4) if cons_claims else 0
                                  for c in CATS},
        "disputed_slots": len(disputed),
        "falsified_vs_verified_slots": sum(1 for d in disputed
                                            if d["counts"]["verified"] and d["counts"]["falsified"]),
        "zero_point_solutions": sum(p["zero_point_solutions"] for p in papers),
        "by_area": {a: dict(v) for a, v in sorted(by_area.items(), key=lambda kv: -kv[1]["papers"])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "medical"], default="all")
    args = ap.parse_args()

    sols = json.load(open(f"{OUT}/solutions_by_paper.json"))
    rows = list(csv.DictReader(open(f"{AUD}/public_repro_audit.csv")))
    area_of = stratify(rows)
    medical = {r["orid"] for r in rows}

    restrict = medical if args.mode == "medical" else None
    papers, disputed = build(sols, area_of, restrict)
    summ = summarize(papers, disputed)

    out = {"summary": summ, "papers": papers, "disputed_slots": disputed}
    fn = f"{OUT}/corpus_{args.mode}.json"
    json.dump(out, open(fn, "w"), indent=1)

    print(f"=== corpus --{args.mode} ===")
    for k, v in summ.items():
        if k != "by_area":
            print(f"{k:28} {v}")
    print("by_area (papers / consensus coverage):")
    for a, v in summ["by_area"].items():
        cov = round(v["consensus"] / v["claims"], 3) if v["claims"] else 0
        print(f"  {a:26} papers={v['papers']:4}  consensus_cov={cov}")
    return out


if __name__ == "__main__":
    main()
