"""Write results/download_manifests_hashed.json: a shareable digest of what was downloaded.

Space ids are replaced by SHA-256 hashes (no usernames). For each Space or revision the file keeps the
status, the number of files and one digest over the sorted file hashes, so a re-download can be checked
against it without publishing who made which logbook.
"""
from __future__ import annotations

import hashlib
import json

import common as C


def h(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest(items) -> str:
    return h(json.dumps(sorted(items), separators=(",", ":")))


def main() -> None:
    load = lambda name: json.loads((C.DATA / name).read_text(encoding="utf-8"))
    pages, commits, trees = load("pages_manifest.json"), load("commits_manifest.json"), load("trees_manifest.json")
    out = dict(
        note="keys are sha256(space_id) or sha256(space_id@revision); digests are sha256 of sorted JSON lists",
        verdict_dataset=dict(repo="ICML-2026-agent-repro/verdicts", repo_type="dataset",
                             revision="f0c5a8b369d6d618ecc9fa060aa2e2f27f9e2602", file="verdicts.json",
                             lfs_sha256="652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526"),
        pages={h(s): dict(status=e["status"], revision=e.get("sha"), n_files=len(e.get("files", {})),
                          digest=digest(list(e.get("files", {}).items()))) for s, e in pages.items()},
        commits={h(s): dict(status=e["status"], n_commits=len(e.get("commits", [])),
                            digest=digest([c[0] for c in e.get("commits", [])])) for s, e in commits.items()},
        trees={h(k): dict(status=e["status"], n_files=len(e.get("files", {})),
                          digest=digest([[p, v[0]] for p, v in e.get("files", {}).items()])) for k, e in trees.items()},
    )
    path = C.HERE / "results" / "download_manifests_hashed.json"
    path.write_text(json.dumps(out, indent=0, sort_keys=True), encoding="utf-8")
    print({k: len(v) for k, v in out.items() if isinstance(v, dict) and k != "verdict_dataset"}, f"{path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
