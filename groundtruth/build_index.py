#!/usr/bin/env python3
"""Ground-truth index for the ICML 2026 repro challenge (JSON-only, no sqlite).

Reads groundtruth/data/*.json, writes groundtruth/out/*.json + a markdown report.
Scoring per challenge leaderboard.js: verified/falsified = 2, toy = 1, inconclusive = 0.
"""
import json, os, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

FULL = {"verified", "falsified"}
TOY = {"toy"}


def pts(v):
    v = (v or "").strip().lower()
    return 2 if v in FULL else (1 if v in TOY else 0)


def load():
    j = lambda n: json.load(open(os.path.join(DATA, n)))
    verdicts = j("verdicts.json")
    papers = {p["orid"]: p for p in j("index.json")["papers"]}
    anchored = j("claims_anchored.json")
    return verdicts, papers, anchored


def build():
    verdicts, papers, anchored = load()
    by_paper = collections.defaultdict(list)
    for sid, v in verdicts.items():
        claims = v.get("claims", [])
        entry = {
            "space_id": sid,
            "author": sid.split("/")[0],
            "judged_at": v.get("judged_at"),
            "judge_model": v.get("model"),
            "n_claims": len(claims),
            "points": sum(pts(c.get("verdict")) for c in claims),
            "max_points": 2 * len(claims),
            "verdicts": [(c.get("verdict") or "").strip().lower() for c in claims],
        }
        orid = v.get("orid") or ""
        if orid:
            by_paper[orid].append(entry)

    best = {}
    for orid, sols in by_paper.items():
        sols.sort(key=lambda e: (-e["points"], e["judged_at"] or ""))
        b = sols[0]
        p = papers.get(orid, {})
        best[orid] = {
            "orid": orid,
            "title": p.get("title") or "",
            "area": p.get("area") or "",
            "sub": p.get("sub") or "",
            "openreview": p.get("or") or f"https://openreview.net/forum?id={orid}",
            "arxiv": p.get("arxiv") or "",
            "n_solutions": len(sols),
            "best_space": b["space_id"],
            "best_points": b["points"],
            "best_max": b["max_points"],
            "best_verdicts": b["verdicts"],
            "n_anchored_claims": len(anchored.get(orid, [])),
        }

    json.dump(best, open(os.path.join(OUT, "best_solutions.json"), "w"), indent=1)
    json.dump({k: v for k, v in by_paper.items()},
              open(os.path.join(OUT, "solutions_by_paper.json"), "w"))

    vocab = collections.Counter(v for sols in by_paper.values() for s in sols for v in s["verdicts"])
    stats = {
        "papers_indexed": len(papers),
        "judged_spaces": len(verdicts),
        "papers_with_judged_solution": len(by_paper),
        "papers_full_points": sum(1 for b in best.values()
                                  if b["best_max"] > 0 and b["best_points"] == b["best_max"]),
        "claims_judged": sum(vocab.values()),
        "verdict_counts": dict(vocab),
    }
    json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1)
    for k, v in stats.items():
        print(k, "=", v)
    return best


if __name__ == "__main__":
    build()
