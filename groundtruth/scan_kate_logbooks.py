#!/usr/bin/env python3
"""Scan BOTH branches (master, codex/medical-reproducibility-map) for the ICML paper
orids that Kate's own logbooks/notebooks cover.

An orid is 10 chars of [A-Za-z0-9]; we harvest them from tracked file contents
(openreview links, paper-<orid> tags, orid fields) plus HF spaces authored by her.
Writes groundtruth/out/kate_logbooks.json
"""
import json, os, re, subprocess, collections

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT = os.path.join(HERE, "out")
BRANCHES = ["origin/master", "origin/codex/medical-reproducibility-map"]
ORID = re.compile(r"(?:forum\?id=|paper-|\borid[\"'\s:=]+)([A-Za-z0-9]{10})\b")


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout


def main():
    known = set(json.load(open(os.path.join(OUT, "best_solutions.json"))))
    anchored = set(json.load(open(os.path.join(HERE, "data", "claims_anchored.json"))))
    valid = known | anchored

    per_branch, per_dir = {}, collections.defaultdict(set)
    for br in BRANCHES:
        files = [f for f in git("ls-tree", "-r", "--name-only", br).splitlines()
                 if re.search(r"\.(md|json|py|csv|txt|ipynb|yaml|yml)$", f)
                 and not f.startswith(("audit/", "docs/", "groundtruth/"))
                 and "papers" not in f.lower() and "all_papers" not in f.lower()]
        found = set()
        for f in files:
            hits = set(ORID.findall(git("show", f"{br}:{f}"))) & valid
            if hits:
                found |= hits
                per_dir[f.split("/")[0]] |= hits
        per_branch[br] = sorted(found)
        print(f"{br}: {len(files)} text files, {len(found)} distinct orids")

    orids = sorted(set().union(*per_branch.values()))
    json.dump({"orids": orids,
               "per_branch": {k: v for k, v in per_branch.items()},
               "per_top_dir": {k: sorted(v) for k, v in sorted(per_dir.items())}},
              open(os.path.join(OUT, "kate_logbooks.json"), "w"), indent=1)
    print(f"total distinct orids covered by Kate's branches: {len(orids)}")
    return orids


if __name__ == "__main__":
    main()
