"""Protocol 9 analysis: coder agreement and adjudication, then A9.3 (prevalence, Mantel-Haenszel, labels L9.1-L9.3).

Usage: python analyze_p9.py score  -> results/p9_coding.json, codes/p9/to_adjudicate.csv (item ids only)
       python analyze_p9.py run    -> results/p9.json (+ p9_manifest.json); needs the gate met or adjudication complete
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from analyze_p8 import kappa, wilson
from build_p9_sheets import CODES, FIELDS, KEYS, LABELS

HERE = Path(__file__).resolve().parent
PRIMARY = ("code", "main_data_access", "weights_release", "variance", "compute")
GATE, B, SEED_BOOT = 0.80, 2000, 20261017
ADJ = CODES / "adjudication.csv"


def read(role: str) -> dict:
    out = {}
    for f in sorted(CODES.glob(f"p9_{role}_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["item_id"] in out:
                    raise SystemExit(f"{f.name}: duplicate {r['item_id']}")
                out[r["item_id"]] = {k: r[k].strip() for k in FIELDS}
    return out


def adjudications() -> dict:
    out = {}
    if ADJ.exists():
        with ADJ.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("reason", "").strip():
                    raise SystemExit(f"adjudication without reason: {r['item_id']} {r['field']}")
                if r["label"] not in LABELS[r["field"]]:
                    raise SystemExit(f"adjudication label not in codebook: {r}")
                out[(r["item_id"], r["field"])] = r["label"]
    return out


def score() -> dict:
    c1, c2 = read("coder1"), read("coder2")
    shared = sorted(set(c1) & set(c2))
    agreement = {f: kappa([c1[i][f] for i in shared], [c2[i][f] for i in shared], LABELS[f]) for f in FIELDS}
    gate = {f: (agreement[f]["kappa"] or 0) >= GATE for f in PRIMARY}
    adj = adjudications()
    pending = [(i, f, c1[i][f], c2[i][f]) for i in shared for f in FIELDS
               if c1[i][f] != c2[i][f] and (i, f) not in adj]
    with (CODES / "to_adjudicate.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "field", "coder1", "coder2"])
        w.writerows(pending)
    out = dict(n_coder1=len(c1), n_coder2=len(c2), n_shared=len(shared), agreement=agreement, gate=gate,
               disagreements=dict(Counter(f for _, f, _, _ in pending)), adjudicated=len(adj))
    (HERE / "results" / "p9_coding.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ---------- analyses (Note 9-5) ----------

INDICATOR = dict(claims="listed_and_linked", code="released", main_data_access="open", open_data_experiment="yes",
                 synthetic_or_sample_release="yes", weights_release="yes", outputs_release="yes", variance="yes",
                 environment="yes", compute="amount")
LABELS_A93 = {"L9.1": "main_data_access", "L9.2": "weights_release", "L9.3": "variance"}


def mh_rd(rows) -> float | None:
    """Mantel-Haenszel risk difference (life science minus other); rows = (stratum, is_ls, y)."""
    t = defaultdict(lambda: [0, 0, 0, 0])
    for s, ls, y in rows:
        c = t[s]
        if ls:
            c[0] += 1; c[1] += y
        else:
            c[2] += 1; c[3] += y
    num = den = 0.0
    for n1, a1, n0, a0 in t.values():
        if n1 and n0:
            num += (a1 * n0 - a0 * n1) / (n1 + n0); den += n1 * n0 / (n1 + n0)
    return num / den if den else None


def boot_ci(rows, stat, cell=lambda r: (r[0], r[1])) -> list[float] | None:
    """Paper bootstrap within cells (default: venue-year x group), sorted input, B resamples, fixed seed."""
    rng = np.random.default_rng(SEED_BOOT)
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: tuple(map(str, r))):
        by[cell(r)].append(r)
    cells = [by[k] for k in sorted(by, key=str)]
    vals = []
    for _ in range(B):
        s = [x for v in cells for x in (v[i] for i in rng.integers(0, len(v), len(v)))]
        v = stat(s)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def prevalence(papers: list[dict], field: str) -> dict:
    pos = INDICATOR.get(field)
    out = {}
    for g in ("life_science", "other"):
        sel = [p for p in papers if p["group"] == g]
        dist = Counter(p[field] for p in sel)
        entry = dict(n=len(sel), distribution=dict(sorted(dist.items())), unclear=dist.get("unclear", 0))
        if pos:
            known = [p for p in sel if p[field] != "unclear"]
            k = sum(p[field] == pos for p in known)
            entry.update(share=k / len(known) if known else None, wilson=wilson(k, len(known)), n_known=len(known))
            entry["by_venue_year"] = {}
            entry["by_year"] = {}
            for key, sub in (("by_venue_year", lambda p: p["stratum"]), ("by_year", lambda p: p["stratum"][-4:])):
                for s in sorted({sub(p) for p in known}):
                    kk = [p for p in known if sub(p) == s]
                    a = sum(p[field] == pos for p in kk)
                    entry[key][s] = dict(n=len(kk), share=a / len(kk), wilson=wilson(a, len(kk)))
        out[g] = entry
    if pos:
        rows = [(p["stratum"], p["group"] == "life_science", int(p[field] == pos)) for p in papers if p[field] != "unclear"]
        out["mh_rd"], out["mh_ci"] = mh_rd(rows), boot_ci(rows, mh_rd)
    return out


def a95(papers: list[dict]) -> dict:
    sys.path.insert(0, str(HERE))
    import p8_common as P8
    cv = P8.claim_verdicts()
    icml = [p for p in papers if p["stratum"] == "icml-2026"]
    att = [dict(p, decided=int(any(c["verdict"] in P8.DECISIVE for c in cv[p["forum_id"]]))) for p in icml if p["forum_id"] in cv]
    out = dict(icml2026_coded=len(icml), attempted=dict(Counter(p["group"] for p in att)), cells={})
    for g in ("life_science", "other"):
        sel = [p["decided"] for p in att if p["group"] == g]
        out["cells"][g] = dict(n=len(sel), decided=sum(sel) / len(sel) if sel else None, wilson=wilson(sum(sel), len(sel)))

    def est(strat, keep):
        rows = [(strat(p), p["group"] == "life_science", p["decided"]) for p in att if keep(p)]
        return dict(n=len(rows), rd=mh_rd(rows), ci=boot_ci(rows, mh_rd, cell=lambda r: r[1]))
    out["unadjusted"] = est(lambda p: 0, lambda p: True)
    out["compute_tier"] = est(lambda p: p["compute_tier"], lambda p: p["compute_tier"] != "unclear")
    out["data_scale"] = est(lambda p: p["data_scale"], lambda p: p["data_scale"] != "unclear")
    out["both"] = est(lambda p: (p["compute_tier"], p["data_scale"]),
                      lambda p: "unclear" not in (p["compute_tier"], p["data_scale"]))
    u, b = out["unadjusted"]["rd"], out["both"]
    out["label_scale_accounts_for_gap"] = bool(b["ci"] and b["ci"][0] <= 0 <= b["ci"][1] and u is not None
                                                and b["rd"] is not None and abs(b["rd"]) < abs(u) / 2)
    return out


def a96() -> dict:
    from analyze_p8 import H84, H85, consensus
    keys = json.loads((HERE / "data" / "p8" / "keys.json").read_text(encoding="utf-8"))
    f84, _, _ = consensus("h84", ("label",), {"label": H84}, True)
    f85, _, _ = consensus("h85", tuple(H85), H85, False)
    access = {keys[i]["orid"]: v["main_data_access"] for i, v in f85.items() if keys[i]["group"] == "medical"}
    out = {}
    for name, cats in (("restricted", ("credentialed_or_request", "private")), ("open", ("open",))):
        papers = {o for o, a in access.items() if a in cats}
        claims = [v["label"] for i, v in f84.items() if keys[i]["orid"] in papers]
        k = sum(c in ("theory", "mechanism") for c in claims)
        out[name] = dict(papers=len(papers), papers_with_coded_claims=len({keys[i]["orid"] for i in f84 if keys[i]["orid"] in papers}),
                         claims=len(claims), theory_or_mechanism=k, share=k / len(claims) if claims else None,
                         wilson=wilson(k, len(claims)))
    return out


def run() -> dict:
    sc = score()
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    c1 = read("coder1")
    papers = [dict(item=i, group=keys[i]["group"], stratum=keys[i]["stratum"], forum_id=keys[i]["forum_id"], **c1[i])
              for i in sorted(c1)]
    coverage = defaultdict(lambda: dict(frame=0, coded=0))
    for i, v in keys.items():
        coverage[f"{v['group']}|{v['stratum']}"]["frame"] += 1
        coverage[f"{v['group']}|{v['stratum']}"]["coded"] += i in c1
    res = dict(agreement=sc["agreement"], gate=sc["gate"], coverage=dict(sorted(coverage.items())),
               coded=dict(Counter(p["group"] for p in papers)), A9_3={})
    for f in FIELDS:
        res["A9_3"][f] = prevalence(papers, f)
        res["A9_3"][f]["provisional"] = f in PRIMARY and not sc["gate"][f]
    res["labels"] = {}
    for lab, f in LABELS_A93.items():
        ci = res["A9_3"][f]["mh_ci"]
        res["labels"][lab] = dict(field=f, rd=res["A9_3"][f]["mh_rd"], ci=ci,
                                  met=None if res["A9_3"][f]["provisional"] else bool(ci and ci[1] < 0))
    res["A9_5"] = a95(papers)
    res["A9_6"] = a96()
    out = HERE / "results" / "p9.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (HERE / "results" / "p9_manifest.json").write_text(json.dumps(dict(
        protocol_sha256=sha(HERE / "PROTOCOL.md"),
        code_sha256={f: sha(HERE / f) for f in ("analyze_p9.py", "build_p9_sheets.py", "build_p9_excerpts.py", "fetch_p9.py")},
        keys_sha256=sha(KEYS), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        seed_boot=SEED_BOOT, resamples=B, results_sha256=sha(out)), indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    print(json.dumps(score() if mode == "score" else run(), indent=1)[:8000])
