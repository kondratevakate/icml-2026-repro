"""Event construction for Protocol 3 (PROTOCOL.md). Shared by the tree download and the analysis."""
from __future__ import annotations

import json
from collections import defaultdict

import common as C
from analyze_exposure import SNAPSHOTS, ts

A = C.A


def load_records() -> dict[str, dict[tuple[str, str], dict]]:
    """space_id -> {(judged_at, sha): record} over the three snapshots (identical judgements merge)."""
    for name, expected in SNAPSHOTS.items():
        got = A.sha256(C.ROOT / name)
        if got != expected:
            raise SystemExit(f"Hash mismatch for {name}: {got}")
    recs: dict[str, dict] = defaultdict(dict)
    for name in SNAPSHOTS:
        for sid, r in A.load_json(C.ROOT / name).items():
            recs[sid][(r["judged_at"], r["sha"])] = r
    return dict(recs)


def commit_times() -> dict[str, dict[str, float]]:
    cm = json.loads((C.DATA / "commits_manifest.json").read_text(encoding="utf-8"))
    return {s: {cid: ts(t) for cid, t in e["commits"]} for s, e in cm.items() if e.get("status") == "ok"}


def events(recs) -> list[dict]:
    """Consecutive observed judgements per Space."""
    out = []
    for sid, rs in recs.items():
        order = sorted(rs)
        for k1, k2 in zip(order, order[1:]):
            out.append(dict(space=sid, team=A.namespace(sid), r1=rs[k1], r2=rs[k2],
                            kind="same_sha" if k1[1] == k2[1] else "new_sha"))
    return out


def matched_claims(ev: dict, anchored: dict[str, set[str]]) -> list[tuple[str, str, str, str, str]]:
    """[(norm claim, verdict1, verdict2, evidence1, evidence2)] for anchored claims present in both judgements."""
    r1, r2 = ev["r1"], ev["r2"]
    if r1.get("orid") != r2.get("orid"):
        return []
    c1 = {A.norm(c.get("claim", "")): c for c in r1.get("claims") or []}
    out = []
    for c in r2.get("claims") or []:
        t = A.norm(c.get("claim", ""))
        if t in c1 and t in anchored.get(r2["orid"], set()):
            out.append((t, A.category(c1[t].get("verdict")), A.category(c.get("verdict")),
                        c1[t].get("evidence", ""), c.get("evidence", "")))
    return out
