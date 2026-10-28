"""Shared loading for Protocol 8 (PROTOCOL.md, Protocol 8): groups, attempted papers, claim verdicts."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
P8 = DATA / "p8"
CODES = HERE / "codes" / "p8"
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402

AUTHOR = "kondratevakate"
VERDICTS = ROOT / "data/verdicts_confirmation_2026-10-17/verdicts.json"
VERDICTS_SHA = "652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526"
SEED_SAMPLE, SEED_BOOT, B = 20261018, 20261017, 2000
DECISIVE = ("verified", "falsified")


def groups() -> dict[str, str]:
    """orid -> 'medical' | 'other' for ICML 2026 (protocol 7 label); excluded candidates are absent."""
    labels = {}
    with (HERE / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "cand" and r["unit"].startswith("icml-2026:"):
                labels[r["unit"].split(":", 1)[1]] = r["label"]
    out = {}
    for u in json.loads((DATA / "papers.json").read_text(encoding="utf-8")):
        if u["stratum"] != "icml-2026":
            continue
        if not u["candidate"]:
            out[u["forum_id"]] = "other"
        elif labels.get(u["forum_id"]) in ("clinical", "life_science"):
            out[u["forum_id"]] = "medical"
    return out


def clinical_ids() -> set[str]:
    with (HERE / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        return {r["unit"].split(":", 1)[1] for r in csv.DictReader(fh)
                if r["kind"] == "cand" and r["unit"].startswith("icml-2026:") and r["label"] == "clinical"}


def claim_verdicts() -> dict[str, list[dict]]:
    """orid -> claim verdicts in deduplicated logbooks, Kate's namespace excluded."""
    if A.sha256(VERDICTS) != VERDICTS_SHA:
        raise SystemExit("verdict snapshot hash mismatch")
    logs, _ = A.dedupe(list(A.load_json(VERDICTS).values()))
    out = defaultdict(list)
    for r in logs:
        if A.namespace(r["space_id"]) == AUTHOR:
            continue
        for c in r.get("claims") or []:
            out[r["orid"]].append(dict(space=r["space_id"], text=A.norm(c.get("claim", "")), raw=c.get("claim", ""),
                                       verdict=A.category(c.get("verdict")), evidence=c.get("evidence", "") or "",
                                       title=r.get("paper_title", "")))
        if not r.get("claims"):
            out[r["orid"]]  # attempted with no claims still counts as attempted
    return dict(out)
