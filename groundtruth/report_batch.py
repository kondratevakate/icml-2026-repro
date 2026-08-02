#!/usr/bin/env python3
"""report_batch.py — after Bundle 2 runs complete, collect scores + emit a table.
Reads <orid>_arm2b/_score.json for each paper in the list.

Usage: python3 report_batch.py --papers <orid>...
"""
import os, sys, argparse, json

COMMON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "experiments/run03_hermes_leaf_pilot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", nargs="+", required=True)
    a = ap.parse_args()
    print(f"{'orid':14} {'pts':5} {'verdicts'}")
    print("-" * 60)
    tot = 0
    for o in a.papers:
        d = os.path.join(COMMON, f"{o}_arm2b")
        sc = os.path.join(d, "_score.json")
        if not os.path.exists(sc):
            print(f"{o:14} {'?':5} (no _score.json yet)")
            continue
        s = json.load(open(sc))
        pc = s.get("per_claim_verdict") or {}
        pts = s.get("rubric_points", "?")
        tot += (pts if isinstance(pts, (int, float)) else 0)
        print(f"{o:14} {str(pts):5} {pc}")
    print("-" * 60)
    print(f"total points: {tot} / {len(a.papers)*10}")


if __name__ == "__main__":
    main()
