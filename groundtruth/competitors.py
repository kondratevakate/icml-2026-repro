#!/usr/bin/env python3
"""Per-competitor reproducibility rates on the ICML 2026 challenge.

Answers: across all leaderboard participants, how accurate is each one under Kate's
stratified metric -- and is the field as a whole better or worse than her runs?

Strata (same screen as select_sample.py): medical vs non-medical, and
theory / simulation / open_data / other by anchored-claim text.

Writes groundtruth/out/competitors.json + competitors.md
"""
import collections, csv, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT, AUD = f"{HERE}/data", f"{HERE}/out", f"{HERE}/audit_snapshot"

THEORY = r"(theorem|proof|we prove|bound|convergence|lemma|proposition|asymptotic|minimax|identifiab|guarantee)"
SIM = r"(simulation|synthetic data|simulated|monte carlo|toy model|generative process|numerical experiment)"
OPEN = r"(MIMIC|PhysioNet|UK Biobank|OpenNeuro|TCGA|ADNI|eICU|public dataset|publicly available|open dataset|benchmark dataset|HCP|ABIDE)"
MIN_LOGBOOKS = 5


def stratum_of(orid, anchored, titles):
    t = titles.get(orid, "") + " " + " ".join(c.get("text", "") for c in anchored.get(orid, []))
    if re.search(THEORY, t, re.I):
        return "theory"
    if re.search(SIM, t, re.I):
        return "simulation"
    if re.search(OPEN, t, re.I):
        return "open_data"
    return "other"


def main():
    j = lambda p: json.load(open(p))
    sols = j(f"{OUT}/solutions_by_paper.json")
    best = j(f"{OUT}/best_solutions.json")
    anchored = j(f"{DATA}/claims_anchored.json")
    medical = {r["orid"] for r in csv.DictReader(open(f"{AUD}/public_repro_audit.csv"))}
    titles = {o: b["title"] for o, b in best.items()}

    strata = {o: stratum_of(o, anchored, titles) for o in sols}

    agg = collections.defaultdict(lambda: {
        "logbooks": 0, "points": 0, "max_points": 0, "papers": set(),
        "verdicts": collections.Counter(), "full_papers": 0, "medical_logbooks": 0,
        "by_stratum": collections.defaultdict(lambda: {"n": 0, "pts": 0, "max": 0}),
    })

    for orid, entries in sols.items():
        st = strata[orid]
        for e in entries:
            a = agg[e["author"]]
            a["logbooks"] += 1
            a["points"] += e["points"]
            a["max_points"] += e["max_points"]
            a["papers"].add(orid)
            a["verdicts"].update(e["verdicts"])
            a["full_papers"] += e["points"] == e["max_points"] and e["max_points"] > 0
            a["medical_logbooks"] += orid in medical
            s = a["by_stratum"][st]
            s["n"] += 1
            s["pts"] += e["points"]
            s["max"] += e["max_points"]

    rows = []
    for author, a in agg.items():
        mx = a["max_points"]
        v = a["verdicts"]
        nc = sum(v.values())
        rows.append({
            "author": author,
            "logbooks": a["logbooks"],
            "papers": len(a["papers"]),
            "points": a["points"],
            "max_points": mx,
            "rate": round(a["points"] / mx, 4) if mx else 0.0,
            "full_papers": a["full_papers"],
            "medical_logbooks": a["medical_logbooks"],
            "claims": nc,
            "verified_rate": round(v["verified"] / nc, 4) if nc else 0,
            "falsified_rate": round(v["falsified"] / nc, 4) if nc else 0,
            "toy_rate": round(v["toy"] / nc, 4) if nc else 0,
            "inconclusive_rate": round(v["inconclusive"] / nc, 4) if nc else 0,
            "by_stratum": {k: {**s, "rate": round(s["pts"] / s["max"], 4) if s["max"] else 0}
                           for k, s in a["by_stratum"].items()},
        })

    rows.sort(key=lambda r: -r["points"])
    field_pts = sum(r["points"] for r in rows)
    field_max = sum(r["max_points"] for r in rows)
    eligible = [r for r in rows if r["logbooks"] >= MIN_LOGBOOKS]
    eligible_by_rate = sorted(eligible, key=lambda r: -r["rate"])

    kate = next((r for r in rows if r["author"] == "kondratevakate"), None)
    better = sum(1 for r in eligible if r["rate"] > (kate["rate"] if kate else 1))

    st_field = collections.defaultdict(lambda: {"pts": 0, "max": 0, "n": 0})
    for r in rows:
        for k, s in r["by_stratum"].items():
            st_field[k]["pts"] += s["pts"]
            st_field[k]["max"] += s["max"]
            st_field[k]["n"] += s["n"]

    summary = {
        "competitors_total": len(rows),
        "competitors_with_min_logbooks": len(eligible),
        "min_logbooks": MIN_LOGBOOKS,
        "field_points": field_pts,
        "field_max": field_max,
        "field_rate": round(field_pts / field_max, 4),
        "median_rate_eligible": round(
            sorted(r["rate"] for r in eligible)[len(eligible) // 2], 4) if eligible else None,
        "kate_rate": kate["rate"] if kate else None,
        "kate_rank_by_rate": next((i + 1 for i, r in enumerate(eligible_by_rate)
                                   if r["author"] == "kondratevakate"), None),
        "kate_beaten_by": better,
        "field_by_stratum": {k: {**v, "rate": round(v["pts"] / v["max"], 4) if v["max"] else 0}
                             for k, v in st_field.items()},
    }

    json.dump({"summary": summary, "rows": rows}, open(f"{OUT}/competitors.json", "w"), indent=1)
    for k, v in summary.items():
        print(f"{k:32} {v}")
    return summary, rows


if __name__ == "__main__":
    main()
