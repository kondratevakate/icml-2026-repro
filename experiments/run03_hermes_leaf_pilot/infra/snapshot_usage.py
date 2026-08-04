#!/usr/bin/env python3
"""
snapshot_usage.py — token metering for one run (replaces deprecated run_leaf.py).

Router /v1/usage tracks tokens PER PROXY_API_KEY. For a clean per-run delta we
snapshot BEFORE the run, then AFTER, and write/extend _score_usage.json in the
run dir. score_run.py reads that file to report tokens_used.

Usage:
  python3 snapshot_usage.py <orid> [--arm N] --before   # write 'before' snapshot
  python3 snapshot_usage.py <orid> [--arm N] --after    # read existing, snapshot, print delta
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_run import router_usage_snapshot

def main():
    args = sys.argv[1:]
    if len(args) < 1 or args[0].startswith("--"):
        print("usage: snapshot_usage.py <orid> [--arm N] [--before|--after]"); sys.exit(1)
    orid = args[0]
    arm = 1
    if "--arm" in args:
        arm = int(args[args.index("--arm") + 1])
    suffix = "" if arm == 1 else f"_arm{arm}"
    d = os.path.join(HERE, orid + suffix)
    if not os.path.isdir(d):
        print(f"no run dir: {d}"); sys.exit(1)
    path = os.path.join(d, "_score_usage.json")

    if "--before" in args:
        snap = router_usage_snapshot()
        json.dump({"before": list(snap[:2]), "before_ts": snap[2], "after": None},
                  open(path, "w"), ensure_ascii=False)
        print(f"[before] tokens={snap[0]:,} reqs={snap[1]:,} -> {path}")

    elif "--after" in args:
        if not os.path.exists(path):
            print("no _score_usage.json (run --before first)"); sys.exit(1)
        u = json.load(open(path))
        if not u.get("before"):
            print("before snapshot missing"); sys.exit(1)
        after = router_usage_snapshot()
        b = u["before"]
        tok = after[0] - b[0] if after[0] is not None and b[0] is not None else None
        req = (after[1] - b[1]) if after[1] is not None and b[1] is not None else None
        u["after"] = list(after[:2]); u["after_ts"] = after[2]
        json.dump(u, open(path, "w"), ensure_ascii=False)
        print(f"[after]  tokens_now={after[0]:,}")
        print(f"[delta] tokens_used={tok:,}  requests_used={req}")
    else:
        print("need --before or --after")

if __name__ == "__main__":
    main()
