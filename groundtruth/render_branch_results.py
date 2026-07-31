#!/usr/bin/env python3
"""Render RESULTS.md for one branch from out/branch_results.json.
Usage: python3 render_branch_results.py <master|codex> > RESULTS.md
"""
import json, os, sys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def render(key):
    r = json.load(open(os.path.join(OUT, "branch_results.json")))[key]
    L = []
    a = L.append
    a(f"# Reproduction results — {r['agent']}")
    a("")
    a(f"Branch `{r['ref'].replace('origin/','')}`. Snapshot of the ICML 2026 challenge "
      f"judged verdicts, frozen automatically by `groundtruth/branch_results.py`.")
    a("")
    a("## Summary")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| agent | **{r['agent']}** |")
    a(f"| logbooks produced locally | {r['logbooks_local']} |")
    a(f"| published **and judged** | {r['published_and_judged']} |")
    a(f"| produced but never judged | **{r['unpublished']}** |")
    a(f"| judged score | **{r['points']}/{r['max_points']}** ({r['normalized']:.1%}) |")
    a(f"| papers at full points | {r['papers_at_full']} |")
    a(f"| papers where this run leads the field | {r['papers_field_leader']} |")
    a("")
    cv = r["claim_verdicts"]
    tot = sum(cv.values())
    a("Claim verdicts across judged logbooks: " + ", ".join(
        f"**{cv.get(k,0)}** {k}" for k in ["verified", "falsified", "toy", "inconclusive"])
      + f" (n={tot}).")
    a("")
    a("> `unjudged` means the logbook exists in this branch but never received a verdict — "
      "the daily publishing quota (20 notebooks) ran out. It is **not** a zero score and "
      "must not be averaged in as one.")
    a("")
    a("## Per paper")
    a("")
    a("| paper | dir | my score | field best | competitors | status |")
    a("|---|---|---|---|---|---|")
    for x in sorted(r["rows"], key=lambda y: (not y["published_judged"],
                                              -(y["my_points"] or 0))):
        if x["published_judged"]:
            mine = f"**{x['my_points']}/{x['my_max']}**"
            lead = " 🥇" if x["my_points"] == x["field_best_points"] else ""
            status = "judged" + lead
        else:
            mine, status = "—", "**unjudged (quota)**"
        fb = (f"{x['field_best_points']}/{x['field_best_max']}"
              if x["field_best_points"] is not None else "—")
        a(f"| [{x['title'][:52] or x['orid']}](https://openreview.net/forum?id={x['orid']}) "
          f"| `{x['dir']}` | {mine} | {fb} | {x['n_competitors']} | {status} |")
    a("")
    a("Scoring: verified/falsified = 2, toy = 1, inconclusive = 0 (challenge `leaderboard.js`).")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.stdout.write(render(sys.argv[1]))
