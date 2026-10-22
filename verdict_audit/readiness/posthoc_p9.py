"""Protocol 9, Note 9-9 (post hoc, descriptive): is the ICML 2026 decided gap carried by claims or by reproduction depth?

PH9-a: decided difference (life science minus other) unadjusted and MH-adjusted for teams, claim count, both;
claim-level decided shares. PH9-b: adjusted for the P9 `claims` code.

Usage: python posthoc_p9.py   -> results/p9_posthoc.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from analyze_p9 import B, SEED_BOOT, boot_ci, final_codes, mh_rd  # noqa: E402
from build_p9_sheets import KEYS  # noqa: E402


def tertile(values: list[int]):
    lo, hi = np.percentile(values, [100 / 3, 200 / 3])
    return lambda v: 0 if v <= lo else 1 if v <= hi else 2


def est(papers, strat, keep=lambda p: True) -> dict:
    rows = [(strat(p), p["ls"], p["decided"]) for p in papers if keep(p)]
    return dict(n=len(rows), rd=mh_rd(rows), ci=boot_ci(rows, mh_rd, cell=lambda r: r[1]))


def accounts(adj: dict, unadj: dict) -> bool:
    return bool(adj["ci"] and adj["ci"][0] <= 0 <= adj["ci"][1] and abs(adj["rd"]) < abs(unadj["rd"]) / 2)


def claim_share(papers) -> dict:
    """Share of claim verdict rows that are decisive, per group; paper bootstrap within group."""
    rng = np.random.default_rng(SEED_BOOT)
    out = {}
    for g, ls in (("life_science", True), ("other", False)):
        sel = sorted((p["rows"], p["dec_rows"]) for p in papers if p["ls"] == ls and p["rows"])
        n, k = np.array([s[0] for s in sel]), np.array([s[1] for s in sel])
        boots = []
        for _ in range(B):
            i = rng.integers(0, len(sel), len(sel))
            boots.append(k[i].sum() / n[i].sum())
        out[g] = dict(papers=len(sel), claim_rows=int(n.sum()), share=float(k.sum() / n.sum()),
                      ci=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
    out["difference"] = out["life_science"]["share"] - out["other"]["share"]
    return out


def main() -> dict:
    cv = P8.claim_verdicts()
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    final = final_codes()
    papers = []
    for i in sorted(final):
        k = keys[i]
        if k["stratum"] != "icml-2026" or k["forum_id"] not in cv:
            continue
        cl = cv[k["forum_id"]]
        papers.append(dict(item=i, ls=k["group"] == "life_science", claims_code=final[i]["claims"],
                           teams=len({P8.A.namespace(c["space"]) for c in cl}), n_claims=len({c["text"] for c in cl}),
                           rows=len(cl), dec_rows=sum(c["verdict"] in P8.DECISIVE for c in cl),
                           decided=int(any(c["verdict"] in P8.DECISIVE for c in cl))))
    ct = tertile([p["n_claims"] for p in papers])
    team = lambda p: min(p["teams"], 3)
    res = dict(units=dict(Counter("life_science" if p["ls"] else "other" for p in papers)))
    res["descriptives"] = {g: dict(mean_teams=float(np.mean([p["teams"] for p in sel])),
                                   mean_claims=float(np.mean([p["n_claims"] for p in sel])),
                                   decided=float(np.mean([p["decided"] for p in sel])))
                           for g, sel in (("life_science", [p for p in papers if p["ls"]]),
                                          ("other", [p for p in papers if not p["ls"]]))}
    u = est(papers, lambda p: 0)
    a = dict(unadjusted=u, teams=est(papers, team), claim_count=est(papers, lambda p: ct(p["n_claims"])),
             both=est(papers, lambda p: (team(p), ct(p["n_claims"]))), claim_level=claim_share(papers))
    for k in ("teams", "claim_count", "both"):
        a[k]["accounts_for_gap"] = accounts(a[k], u)
    res["PH9_a"] = a
    b = est(papers, lambda p: p["claims_code"], keep=lambda p: p["claims_code"] != "unclear")
    b["accounts_for_gap"] = accounts(b, u)
    b["distribution"] = {g: dict(Counter(p["claims_code"] for p in papers if p["ls"] == ls))
                         for g, ls in (("life_science", True), ("other", False))}
    res["PH9_b"] = b
    (HERE / "results" / "p9_posthoc.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
