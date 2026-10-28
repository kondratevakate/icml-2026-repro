"""Write results/download_manifests_hashed.json: aggregate digests of what was downloaded.

No per-Space entries are written. Space ids are public in the challenge's verdict dataset, so any
per-Space hash (and any revision id) could be mapped back to a Space; only counts and one digest per
source over all Spaces are kept. A full re-download can be checked against these digests.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import common as C


def digest(items) -> str:
    return hashlib.sha256(json.dumps(sorted(items), separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    load = lambda name: json.loads((C.DATA / name).read_text(encoding="utf-8"))
    pages, commits, trees = load("pages_manifest.json"), load("commits_manifest.json"), load("trees_manifest.json")
    out = dict(
        note="aggregate only: counts and one sha256 digest per source over sorted per-Space content digests",
        verdict_dataset=dict(repo="ICML-2026-agent-repro/verdicts", repo_type="dataset",
                             revision="f0c5a8b369d6d618ecc9fa060aa2e2f27f9e2602", file="verdicts.json",
                             lfs_sha256="652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526"),
        pages=dict(status=dict(Counter(e["status"] for e in pages.values())),
                   n_files=sum(len(e.get("files", {})) for e in pages.values()),
                   digest=digest([[s, e.get("sha"), sorted(e.get("files", {}).items())] for s, e in pages.items()])),
        commits=dict(status=dict(Counter(e["status"] for e in commits.values())),
                     n_commits=sum(len(e.get("commits", [])) for e in commits.values()),
                     digest=digest([[s, sorted(c[0] for c in e.get("commits", []))] for s, e in commits.items()])),
        trees=dict(status=dict(Counter(e["status"] for e in trees.values())),
                   n_files=sum(len(e.get("files", {})) for e in trees.values()),
                   digest=digest([[k, sorted([p, v[0]] for p, v in e.get("files", {}).items())] for k, e in trees.items()])),
    )
    path = C.HERE / "results" / "download_manifests_hashed.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k in ("pages", "commits", "trees")}))


if __name__ == "__main__":
    main()
