#!/usr/bin/env python3
"""compare_runs.py — compare two reproduction runs of the SAME paper (e.g. arm2 vs
arm2+compaction) by their _score.json per_claim_verdict. Reports whether compaction
preserved quality (verdicts identical) or regressed.

Usage: python3 compare_runs.py <dir_a> <dir_b>
"""
import json, os, sys, argparse


def load(d):
    p = os.path.join(d, "_score.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p)).get("per_claim_verdict") or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    a = ap.parse_args()
    va, vb = load(a), load(b)
    if va is None or vb is None:
        print("missing _score.json in one of the dirs")
        return
    keys = sorted(set(va) | set(vb))
    print(f"{'claim':8} {'A':14} {'B':14}  match")
    print("-" * 44)
    mism = 0
    for k in keys:
        x, y = va.get(k, "?"), vb.get(k, "?")
        m = "✓" if x == y else "✗"
        if x != y:
            mism += 1
        print(f"{k:8} {str(x):14} {str(y):14}  {m}")
    print(f"\nverdict mismatches: {mism}/{len(keys)}")
    print("COMPACTION PRESERVED QUALITY" if mism == 0 else "COMPACTION CHANGED VERDICTS")


if __name__ == "__main__":
    main()
