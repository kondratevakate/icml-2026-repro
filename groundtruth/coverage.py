#!/usr/bin/env python3
"""Coverage report: of the 703 medical/life-science ICML 2026 papers,
how many are reproduced (per the live judged verdicts), and how many of those
are NOT covered by Kate's own logbooks on either branch (master / codex).

Reads:
  groundtruth/out/best_solutions.json      (live judged data, from build_index.py)
  groundtruth/audit_snapshot/public_repro_audit.csv  (703-paper medical population)
  groundtruth/out/kate_logbooks.json       (orids Kate covered, from scan_kate_logbooks.py)
Writes:
  groundtruth/out/coverage.json + coverage.md
"""
import csv, json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, AUD = os.path.join(HERE, "out"), os.path.join(HERE, "audit_snapshot")

KATE_USER = "kondratevakate"


def load_population():
    with open(os.path.join(AUD, "public_repro_audit.csv")) as f:
        return {r["orid"]: r for r in csv.DictReader(f)}


def main():
    pop = load_population()
    best = json.load(open(os.path.join(OUT, "best_solutions.json")))
    kate = set(json.load(open(os.path.join(OUT, "kate_logbooks.json")))["orids"])
    sols = json.load(open(os.path.join(OUT, "solutions_by_paper.json")))

    rows = []
    for orid, p in pop.items():
        b = best.get(orid)
        pts = b["best_points"] if b else 0
        mx = b["best_max"] if b else 0
        # papers Kate covered anywhere (branch dirs) OR via a judged HF space of hers
        kate_space = any(s["author"] == KATE_USER for s in sols.get(orid, []))
        rows.append({
            "orid": orid,
            "title": p["title"],
            "area": p["area"],
            "n_claims": int(p["n_claims"] or 0),
            "gpu": p["gpu_required_signal"] == "True",
            "data_access": p["data_access_signal"],
            "n_solutions": b["n_solutions"] if b else 0,
            "best_points": pts,
            "best_max": mx,
            "reproduced_any": pts > 0,
            "reproduced_full": mx > 0 and pts == mx,
            "kate_covered": orid in kate or kate_space,
        })

    n = len(rows)
    rep_any = [r for r in rows if r["reproduced_any"]]
    rep_full = [r for r in rows if r["reproduced_full"]]
    gap_any = [r for r in rep_any if not r["kate_covered"]]
    gap_full = [r for r in rep_full if not r["kate_covered"]]
    kate_rows = [r for r in rows if r["kate_covered"]]

    stats = {
        "population": n,
        "kate_covered": len(kate_rows),
        "reproduced_any_points": len(rep_any),
        "reproduced_full_points": len(rep_full),
        "reproduced_any_not_in_kate": len(gap_any),
        "reproduced_full_not_in_kate": len(gap_full),
        "gap_full_cpu_only": sum(1 for r in gap_full if not r["gpu"]),
        "gap_any_cpu_only": sum(1 for r in gap_any if not r["gpu"]),
        "never_attempted": sum(1 for r in rows if r["n_solutions"] == 0),
    }
    json.dump({"stats": stats, "rows": rows}, open(os.path.join(OUT, "coverage.json"), "w"), indent=1)

    for k, v in stats.items():
        print(f"{k:34} {v}")

    gap_full.sort(key=lambda r: (r["gpu"], -r["best_points"]))
    lines = ["| orid | best | claims | GPU | data | title |", "|---|---|---|---|---|---|"]
    for r in gap_full[:60]:
        lines.append(f"| `{r['orid']}` | {r['best_points']}/{r['best_max']} | {r['n_claims']} | "
                     f"{'yes' if r['gpu'] else 'no'} | {r['data_access']} | {r['title'][:70]} |")
    open(os.path.join(OUT, "coverage.md"), "w").write(
        f"# Coverage gap: fully-reproduced medical papers not in Kate's logbooks\n\n"
        + json.dumps(stats, indent=1) + "\n\n" + "\n".join(lines) + "\n")
    print("\nwrote out/coverage.json and out/coverage.md")


if __name__ == "__main__":
    main()
