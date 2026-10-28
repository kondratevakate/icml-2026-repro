"""Protocol 9, Note 9-15: claim-text coding — agreement, adjudication sheets, merge, and analyses K1-K3.

Usage: python analyze_p9_claims.py score | build_adj | merge | run
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p8_common as P8  # noqa: E402
from analyze_p8 import kappa  # noqa: E402
from analyze_p9 import B, SEED_BOOT, mh_rd  # noqa: E402
from build_p9_sheets import P9  # noqa: E402
from posthoc_p9_areas import boot_papers  # noqa: E402

CODES = HERE / "codes" / "p9" / "claims"
KEYS = P9 / "claim_keys.json"
ADJ_SHEETS, ADJ_BATCHES = P9 / "claim_adj_sheets", P9 / "claim_adj_batches.json"
ADJ = CODES / "claim_adjudication.csv"
FIELDS = ("check_cost", "quantitative", "comparison")
LABELS = dict(check_cost=("inference_only", "retrain", "large_scale", "analytic", "unclear"),
              quantitative=("yes", "no"), comparison=("yes", "no"))
DEFINITION = dict(
    check_cost="check_cost: inference_only (one run of released or easily rebuilt artifacts settles it), retrain (the "
               "method must be trained or fitted), large_scale (many runs, large models or datasets, broad sweeps or "
               "ablations, many benchmarks), analytic (proof, derivation or definition; no experiment), unclear",
    quantitative="quantitative: yes (the claim states a number or metric value), no",
    comparison="comparison: yes (against a baseline, prior work or the state of the art), no")
COSTLY = ("retrain", "large_scale")


def read(role: str) -> dict:
    out = {}
    for f in sorted(CODES.glob(f"p9_{role}_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                vals = {k: (r.get(k) or "").strip() for k in FIELDS}
                if any(vals[k] not in LABELS[k] for k in FIELDS):
                    raise SystemExit(f"{f.name}: invalid label for {r['item_id']}: {vals}")
                out[r["item_id"]] = vals
    return out


def adjudications() -> dict:
    out = {}
    if ADJ.exists():
        with ADJ.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                out[(r["item_id"], r["field"])] = r["label"]
    return out


def score() -> dict:
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    c1, c2 = read("claim1"), read("claim2")
    shared = sorted(set(c1) & set(c2))
    res = dict(items=len(keys), coded1=len(c1), coded2=len(c2), shared=len(shared),
               agreement={f: kappa([c1[i][f] for i in shared], [c2[i][f] for i in shared], LABELS[f]) for f in FIELDS})
    res["disagreements"] = {f: sum(c1[i][f] != c2[i][f] for i in shared) for f in FIELDS}
    print(json.dumps(res, indent=1))
    return res


def build_adj() -> None:
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    c1, c2 = read("claim1"), read("claim2")
    if set(c1) != set(keys) or set(c2) != set(keys):
        raise SystemExit("coding incomplete")
    sheets = {}
    for f in sorted(P9.glob("claim_sheets/p9_claim1_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            sheets.update({r["item_id"]: r["claim"] for r in csv.DictReader(fh)})
    todo = [(i, f) for i in sorted(keys) for f in FIELDS if c1[i][f] != c2[i][f]]
    rng = random.Random(20261023)
    ADJ_SHEETS.mkdir(parents=True, exist_ok=True)
    batches = []
    for n, start in enumerate(range(0, len(todo), 40)):
        chunk = todo[start:start + 40]
        with (ADJ_SHEETS / f"p9_claimadj_batch_{n:03d}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, ["item_id", "field", "definition", "label_a", "label_b", "claim"])
            w.writeheader()
            for i, f in chunk:
                a, b = (c1[i][f], c2[i][f]) if rng.random() < 0.5 else (c2[i][f], c1[i][f])
                w.writerow(dict(item_id=i, field=f, definition=DEFINITION[f], label_a=a, label_b=b, claim=sheets[i]))
        batches.append(dict(batch=n, items=chunk))
    ADJ_BATCHES.write_text(json.dumps(batches), encoding="utf-8")
    print(len(todo), "disagreements in", len(batches), "batches")


def merge() -> None:
    rows = []
    for b in json.loads(ADJ_BATCHES.read_text(encoding="utf-8")):
        with (CODES / f"p9_claimadj_batch_{b['batch']:03d}.csv").open(newline="", encoding="utf-8") as fh:
            got = {(r["item_id"], r["field"]): r for r in csv.DictReader(fh)}
        for i, f in b["items"]:
            r = got.get((i, f))
            if r is None or r["label"].strip() not in LABELS[f] or not r.get("reason", "").strip():
                raise SystemExit(f"adjudication batch {b['batch']}: bad or missing {i} {f}")
            rows.append(dict(item_id=i, field=f, label=r["label"].strip(), reason=" ".join(r["reason"].split())))
    with ADJ.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["item_id", "field", "label", "reason"])
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "adjudications merged")


def final() -> dict:
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    c1, c2, adj = read("claim1"), read("claim2"), adjudications()
    out = {}
    for i in sorted(keys):
        out[i] = {}
        for f in FIELDS:
            if c1[i][f] == c2[i][f]:
                out[i][f] = c1[i][f]
            elif (i, f) in adj:
                out[i][f] = adj[(i, f)]
            else:
                raise SystemExit(f"not adjudicated: {i} {f}")
    return out


def run() -> dict:
    from posthoc_p9_adversarial import icml_units
    from posthoc_p9_more import load
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    codes = final()
    code_of = {(v["orid"], v["text"]): codes[i] for i, v in keys.items()}
    sampled = {v["orid"] for v in keys.values()}
    papers = [p for p in load(icml_units()) if p["orid"] in sampled]
    cv = P8.claim_verdicts()
    res = dict(score=score(), papers=dict(Counter("medical" if p["med"] else "other" for p in papers)))

    # K1: claim-level shares by group, bootstrap over papers
    per_paper = defaultdict(list)
    for i, v in keys.items():
        per_paper[v["orid"]].append(codes[i])
    k1 = {}
    for name, test in (("costly", lambda c: c["check_cost"] in COSTLY), ("analytic", lambda c: c["check_cost"] == "analytic"),
                       ("quantitative", lambda c: c["quantitative"] == "yes"),
                       ("comparison", lambda c: c["comparison"] == "yes")):
        for p in papers:
            p["_k"], p["_n"] = sum(test(c) for c in per_paper[p["orid"]]), len(per_paper[p["orid"]])
        share = lambda s, m: (sum(p["_k"] for p in s if p["med"] == m) / max(1, sum(p["_n"] for p in s if p["med"] == m)))
        diff = lambda s: share(s, True) - share(s, False)
        k1[name] = dict(medical=share(papers, True), other=share(papers, False), diff=diff(papers),
                        ci=boot_papers(papers, diff))
    k1["check_cost_distribution"] = {g: dict(Counter(codes[i]["check_cost"] for i, v in keys.items() if v["group"] == g))
                                     for g in ("medical", "other")}
    res["K1"] = k1

    # K2: within-team claim-level decisive, adjusted for check_cost
    for p in papers:
        p["claim_rows"] = [(P8.A.namespace(c["space"]), code_of[(p["orid"], c["text"])]["check_cost"],
                            int(c["verdict"] in P8.DECISIVE)) for c in cv.get(p["orid"], [])
                           if (p["orid"], c["text"]) in code_of]
    team = lambda s: mh_rd([(ns, p["med"], d) for p in s for ns, _, d in p["claim_rows"]])
    team_cost = lambda s: mh_rd([((ns, cc), p["med"], d) for p in s for ns, cc, d in p["claim_rows"]])
    raw = lambda s: mh_rd([(0, p["med"], d) for p in s for _, _, d in p["claim_rows"]])
    k2 = dict(unadjusted=dict(rd=raw(papers), ci=boot_papers(papers, raw)),
              within_team=dict(rd=team(papers), ci=boot_papers(papers, team)),
              within_team_and_cost=dict(rd=team_cost(papers), ci=boot_papers(papers, team_cost)))
    by_cost = {}
    for cc in LABELS["check_cost"]:
        rows = [(0, p["med"], d) for p in papers for _, c, d in p["claim_rows"] if c == cc]
        by_cost[cc] = dict(n=len(rows), medical_decisive=float(np.mean([d for _, m, d in rows if m])) if any(m for _, m, _ in rows) else None,
                           other_decisive=float(np.mean([d for _, m, d in rows if not m])) if any(not m for _, m, _ in rows) else None)
    k2["decisive_by_check_cost"] = by_cost
    res["K2"] = k2

    # K3: fully verified, adjusted for first-team strength and the paper's share of costly claims
    for p in papers:
        cs = [c["check_cost"] in COSTLY for c in per_paper[p["orid"]]]
        sh = sum(cs) / len(cs) if cs else 0
        p["cost_bin"] = 0 if sh == 0 else 1 if sh <= 0.5 else 2
    g = lambda strat: (lambda s: mh_rd([(strat(p), p["med"], p["fully_verified"]) for p in s]))
    k3 = {}
    for name, strat in (("unadjusted", lambda p: 0), ("strength", lambda p: p["s_bin"]),
                        ("strength_and_cost", lambda p: (p["s_bin"], p["cost_bin"]))):
        f = g(strat)
        k3[name] = dict(rd=f(papers), ci=boot_papers(papers, f))
    k3["cost_accounts"] = bool(abs(k3["strength_and_cost"]["rd"]) < abs(k3["strength"]["rd"]) / 2
                               and k3["strength_and_cost"]["ci"][0] <= 0 <= k3["strength_and_cost"]["ci"][1])
    res["K3"] = k3
    k2["cost_accounts"] = bool(abs(k2["within_team_and_cost"]["rd"]) < abs(k2["within_team"]["rd"]) / 2
                               and k2["within_team_and_cost"]["ci"][0] <= 0 <= k2["within_team_and_cost"]["ci"][1])
    (HERE / "results" / "p9_claims.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    {"score": score, "build_adj": build_adj, "merge": merge, "run": lambda: print(json.dumps(run(), indent=1)[:4000])}[sys.argv[1]]()
