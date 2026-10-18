"""Shared loading for the voters study (verdict_audit/voters/PROTOCOL.md)."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402  (frozen helpers, reused unchanged)

ROOT = A.ROOT
DATA = HERE / "data"
PAGES = DATA / "pages"
INPUTS = {
    "data/verdicts_confirmation_2026-10-17/verdicts.json": "652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526",
    "data/spaces_2026-10-17/spaces.json": "71fbe96bbc5d543a55aa69d19701fa5dbc34c2e6e0019719a154f41b53aa3c5a",
    "claims_anchored.json": "eb3f2d878646ca5c40da121741a613c1f9e1e10c14845f373db107e74a4ef439",
}
ORGANISER = "ICML-2026-agent-repro"
AUTHOR = "kondratevakate"


def check_inputs() -> dict:
    hashes = {}
    for name, expected in INPUTS.items():
        got = A.sha256(ROOT / name)
        if got != expected:
            sys.exit(f"Hash mismatch for {name}: {got}")
        hashes[name] = got
    return hashes


def _short(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def local_dir(space_id: str) -> Path:
    return PAGES / _short(space_id)


def file_key(path: str) -> str:
    return _short(path) + ".txt"


def read_pages(space_id: str, entry: dict) -> dict[str, str]:
    """{repo path: text} for one downloaded Space, verifying each stored hash."""
    out = {}
    for path, digest in entry.get("files", {}).items():
        data = (local_dir(space_id) / file_key(path)).read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            sys.exit(f"Page hash mismatch: {space_id} {path}")
        out[path] = data.decode("utf-8", errors="replace")
    return out


def load():
    """Deduplicated logbooks, anchored texts, createdAt and the multi-rated claim map."""
    verdicts = A.load_json(ROOT / "data/verdicts_confirmation_2026-10-17/verdicts.json")
    logbooks, dropped = A.dedupe(list(verdicts.values()))
    raw = json.loads((ROOT / "claims_anchored.json").read_text(encoding="utf-8-sig"))
    anchored = {k: {A.norm(c["text"]) for c in v} for k, v in raw.items()}
    spaces = json.loads((ROOT / "data/spaces_2026-10-17/spaces.json").read_text(encoding="utf-8-sig"))
    created = {s["id"]: s.get("createdAt") for s in spaces}
    return logbooks, dropped, raw, anchored, created


def claim_raters(logbooks, anchored) -> dict[tuple[str, str], dict[str, str]]:
    """(orid, claim text) -> {space_id: verdict}, first logbook per team, as in analyze.multi_rated."""
    per_claim: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in logbooks:
        ns = A.namespace(r["space_id"])
        for c in r.get("claims") or []:
            t = A.norm(c.get("claim", ""))
            key = (r["orid"], t)
            if t in anchored.get(r["orid"], set()) and ns not in seen[key]:
                seen[key].add(ns)
                per_claim[key][r["space_id"]] = A.category(c.get("verdict"))
    return dict(per_claim)


def multi_rated_claims(logbooks, anchored):
    return {k: v for k, v in claim_raters(logbooks, anchored).items() if len(v) >= 2}
