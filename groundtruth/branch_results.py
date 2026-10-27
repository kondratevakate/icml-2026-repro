#!/usr/bin/env python3
"""Freeze the per-branch result of Kate's own reproduction runs.

master                              -> Claude (Opus) run
codex/medical-reproducibility-map   -> Codex run (hit the 20-notebook/day publish limit)

For every paper a branch covers, report whether the logbook was published AND judged,
and if so the judged score. Unpublished = produced locally but never got a verdict
(publishing quota exhausted), which is NOT the same as a zero score.

Writes groundtruth/out/branch_results.json and RESULTS_<branch>.md next to it.
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
KU = "kondratevakate"

BRANCHES = {
    "master": ("origin/master", "Claude (Opus)"),
    "codex": ("origin/codex/medical-reproducibility-map", "Codex"),
}


def main():
    kate = json.load(open(os.path.join(OUT, "kate_logbooks.json")))
    sols = json.load(open(os.path.join(OUT, "solutions_by_paper.json")))
    best = json.load(open(os.path.join(OUT, "best_solutions.json")))
    orid2dir = {o: d for d, orids in kate["per_top_dir"].items() for o in orids}

    report = {}
    for key, (ref, agent) in BRANCHES.items():
        rows = []
        for orid in sorted(kate["per_branch"][ref]):
            mine = [s for s in sols.get(orid, []) if s["author"] == KU]
            mine.sort(key=lambda s: (-s["points"], s["judged_at"] or ""))
            b = best.get(orid, {})
            top = mine[0] if mine else None
            rows.append({
                "orid": orid,
                "dir": orid2dir.get(orid, ""),
                "title": b.get("title", ""),
                "published_judged": bool(mine),
                "my_points": top["points"] if top else None,
                "my_max": top["max_points"] if top else None,
                "my_space": top["space_id"] if top else None,
                "my_verdicts": top["verdicts"] if top else None,
                "field_best_points": b.get("best_points"),
                "field_best_max": b.get("best_max"),
                "field_best_space": b.get("best_space"),
                "n_competitors": b.get("n_solutions", 0),
            })

        pub = [r for r in rows if r["published_judged"]]
        pts = sum(r["my_points"] for r in pub)
        mx = sum(r["my_max"] for r in pub)
        report[key] = {
            "ref": ref,
            "agent": agent,
            "logbooks_local": len(rows),
            "published_and_judged": len(pub),
            "unpublished": len(rows) - len(pub),
            "points": pts,
            "max_points": mx,
            "normalized": round(pts / mx, 4) if mx else None,
            "papers_at_full": sum(1 for r in pub if r["my_points"] == r["my_max"]),
            "papers_field_leader": sum(1 for r in pub if r["my_points"] == r["field_best_points"]),
            "claim_verdicts": dict(collections.Counter(
                v for r in pub for v in (r["my_verdicts"] or []))),
            "rows": rows,
        }
        r = report[key]
        print(f"{key:8} {agent:14} local={r['logbooks_local']:3} judged={r['published_and_judged']:3} "
              f"unpublished={r['unpublished']:3} score={pts}/{mx} "
              f"({r['normalized']}) full={r['papers_at_full']} leader={r['papers_field_leader']}")
        print(f"         claims: {r['claim_verdicts']}")

    json.dump(report, open(os.path.join(OUT, "branch_results.json"), "w"), indent=1)
    print("\nwrote out/branch_results.json")
    return report


if __name__ == "__main__":
    main()
