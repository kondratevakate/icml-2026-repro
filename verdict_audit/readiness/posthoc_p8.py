"""Protocol 8 post hoc analyses PH8-a, PH8-b, PH8-c (Note 8-2): composition of the H8.5 policy cells.

Usage: python posthoc_p8.py   -> results/p8_posthoc.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

import p8_common as P
from analyze_p8 import H85, THEORY_RE, consensus, wilson

RESTRICTED = ("credentialed_or_request", "private")


def mh(papers: list[dict], expo: str) -> float | None:
    """Mantel-Haenszel risk difference (exposed minus unexposed) over the two groups."""
    t = defaultdict(lambda: [0, 0, 0, 0])
    for p in papers:
        c = t[p["g"]]
        if p[expo]:
            c[0] += 1; c[1] += p["decided"]
        else:
            c[2] += 1; c[3] += p["decided"]
    num = den = 0.0
    for ne, ae, nu, au in t.values():
        if ne and nu:
            num += (ae * nu - au * ne) / (ne + nu); den += ne * nu / (ne + nu)
    return num / den if den else None


def boot(papers: list[dict], cell_key, stat) -> list[float] | None:
    """Resample papers within cells, as in analyze_p8.py H8.5."""
    rng = np.random.default_rng(P.SEED_BOOT)
    by = defaultdict(list)
    for p in papers:
        by[cell_key(p)].append(p)
    vals = [stat([x for v in by.values() for x in (v[i] for i in rng.integers(0, len(v), len(v)))]) for _ in range(P.B)]
    vals = [v for v in vals if v is not None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def cell(sel: list[int]) -> dict:
    return dict(n=len(sel), decided=sum(sel) / len(sel) if sel else None, wilson=wilson(sum(sel), len(sel)))


def main() -> dict:
    grp, cv = P.groups(), P.claim_verdicts()
    keys = json.loads((P.P8 / "keys.json").read_text(encoding="utf-8"))
    f85, _, pending = consensus("h85", tuple(H85), H85, False)
    if pending:
        raise SystemExit(f"h85 adjudication incomplete: {len(pending)} pending")
    papers = []
    for i, v in f85.items():
        o = keys[i]["orid"]
        claims = cv[o]
        papers.append(dict(orid=o, g=keys[i]["group"], access=v["main_data_access"], ode=v["open_data_experiment"],
                           decided=int(any(c["verdict"] in P.DECISIVE for c in claims)),
                           empirical=int(any(not THEORY_RE.search(c["text"]) for c in claims))))
    if any(grp.get(p["orid"]) != p["g"] for p in papers):
        raise SystemExit("group mismatch between keys.json and protocol 7 labels")
    res = {}

    # PH8-a: composition of the policy cells
    res["PH8_a"] = {f"{g}_open_{v}": dict(n=len(s), synthetic_only=sum(p["access"] == "synthetic_only" for p in s),
                                          no_empirical_claim=sum(not p["empirical"] for p in s))
                    for g in ("medical", "other") for v, lab in ((1, "yes"), (0, "no"))
                    for s in [[p for p in papers if p["g"] == g and p["ode"] == lab]]}

    # PH8-b: decided share by main data access, open versus restricted
    acc = [dict(p, restricted=int(p["access"] in RESTRICTED)) for p in papers if p["access"] in ("open",) + RESTRICTED]
    cells = {f"{g}_{lab}": cell([p["decided"] for p in acc if p["g"] == g and p["restricted"] == v])
             for g in ("medical", "other") for v, lab in ((0, "open"), (1, "restricted"))}
    med = [p for p in acc if p["g"] == "medical"]

    def diff(s):
        r, o = [p["decided"] for p in s if p["restricted"]], [p["decided"] for p in s if not p["restricted"]]
        return (sum(r) / len(r) - sum(o) / len(o)) if r and o else None
    res["PH8_b"] = dict(cells=cells, medical_restricted_minus_open=diff(med),
                        medical_restricted_minus_open_ci=boot(med, lambda p: p["restricted"], diff),
                        access_counts={g: dict(Counter(p["access"] for p in papers if p["g"] == g)) for g in ("medical", "other")})

    # PH8-c: policy test among empirical, non-synthetic-only papers
    pol = [dict(p, open=int(p["ode"] == "yes")) for p in papers
           if p["ode"] in ("yes", "no") and p["empirical"] and p["access"] != "synthetic_only"]
    res["PH8_c"] = dict(cells={f"{g}_open_{v}": cell([p["decided"] for p in pol if p["g"] == g and p["open"] == v])
                               for g in ("medical", "other") for v in (1, 0)},
                        mh_diff=mh(pol, "open"), mh_ci=boot(pol, lambda p: (p["g"], p["open"]), lambda s: mh(s, "open")))
    (P.HERE / "results" / "p8_posthoc.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
