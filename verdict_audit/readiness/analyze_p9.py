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
BELOW_GATE = ("variance", "compute")  # Note 9-5; coder 2 completes them for all papers (Note 9-6)
REF_FIELDS = ("code_used", "data_obtained", "weights_used")
REF_LABELS = {f: ("yes", "no", "unclear") for f in REF_FIELDS}
GATE, B, SEED_BOOT = 0.80, 2000, 20261017
ADJ = CODES / "adjudication.csv"


def read(role: str, fields=FIELDS) -> dict:
    out = {}
    for f in sorted(CODES.glob(f"p9_{role}_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["item_id"] in out:
                    raise SystemExit(f"{f.name}: duplicate {r['item_id']}")
                out[r["item_id"]] = {k: r[k].strip() for k in fields}
    return out


def adjudications() -> dict:
    out = {}
    cats = {**LABELS, **REF_LABELS}
    if ADJ.exists():
        with ADJ.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("reason", "").strip():
                    raise SystemExit(f"adjudication without reason: {r['item_id']} {r['field']}")
                if r["label"] not in cats[r["field"]]:
                    raise SystemExit(f"adjudication label not in codebook: {r}")
                out[(r["item_id"], r["field"])] = r["label"]
    return out


def read_complete(role: str) -> dict:
    """Codes from batch files that cover their batch exactly with valid labels (skips files still being written)."""
    from build_p9_sheets import BATCHES
    out = {}
    for b in json.loads(BATCHES.read_text(encoding="utf-8")).get(role, []):
        f = CODES / f"p9_{role}_batch_{b['batch']:03d}.csv"
        if not f.exists():
            continue
        with f.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if sorted(r["item_id"] for r in rows) == sorted(b["items"]) and all(
                (r.get(k) or "").strip() in LABELS[k] for r in rows for k in FIELDS):
            out.update({r["item_id"]: {k: r[k].strip() for k in FIELDS} for r in rows})
    return out


def coder2_all() -> dict:
    """Coder 2 on every coded paper: its 25% sample plus the Note 9-6 completion (role coder2full)."""
    c2, full = read("coder2"), read_complete("coder2full")
    if set(c2) & set(full):
        raise SystemExit("coder2 and coder2full overlap")
    return {**c2, **full}


def pending_items() -> list[tuple]:
    """Disagreements still to adjudicate: variance and compute over all papers, reference fields over all units."""
    c1, c2 = read("coder1"), coder2_all()
    r1, r2 = read("ref1", REF_FIELDS), read("ref2", REF_FIELDS)
    adj = adjudications()
    out = [(i, f, c1[i][f], c2[i][f]) for i in sorted(set(c1) & set(c2)) for f in BELOW_GATE
           if c1[i][f] != c2[i][f] and (i, f) not in adj]
    out += [(i, f, r1[i][f], r2[i][f]) for i in sorted(set(r1) & set(r2)) for f in REF_FIELDS
            if r1[i][f] != r2[i][f] and (i, f) not in adj]
    return out


def score() -> dict:
    c1, c2 = read("coder1"), read("coder2")
    shared = sorted(set(c1) & set(c2))
    agreement = {f: kappa([c1[i][f] for i in shared], [c2[i][f] for i in shared], LABELS[f]) for f in FIELDS}
    gate = {f: (agreement[f]["kappa"] or 0) >= GATE for f in PRIMARY}
    c2a = coder2_all()
    shared_all = sorted(set(c1) & set(c2a))
    agreement_all = {f: kappa([c1[i][f] for i in shared_all], [c2a[i][f] for i in shared_all], LABELS[f])
                     for f in BELOW_GATE}
    r1, r2 = read("ref1", REF_FIELDS), read("ref2", REF_FIELDS)
    rs = sorted(set(r1) & set(r2))
    agreement_ref = {f: kappa([r1[i][f] for i in rs], [r2[i][f] for i in rs], REF_LABELS[f]) for f in REF_FIELDS}
    pending = pending_items()
    with (CODES / "to_adjudicate.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "field", "label_a", "label_b"])
        w.writerows(pending)
    out = dict(n_coder1=len(c1), n_coder2=len(c2), n_shared=len(shared), agreement=agreement, gate=gate,
               n_coder2_all=len(c2a), agreement_all_below_gate=agreement_all, n_reference=len(rs),
               agreement_reference=agreement_ref, pending=dict(Counter(f for _, f, _, _ in pending)),
               adjudicated=len(adjudications()))
    (HERE / "results" / "p9_coding.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def final_codes() -> dict:
    """Coder 1 for fields above the gate; for variance and compute coder 1 where coders agree, else adjudicated."""
    c1, c2a, adj = read("coder1"), coder2_all(), adjudications()
    if set(c1) - set(c2a):
        raise SystemExit(f"coder 2 has not coded {len(set(c1) - set(c2a))} papers yet (Note 9-6)")
    out = {}
    for i in sorted(c1):
        out[i] = dict(c1[i])
        for f in BELOW_GATE:
            if c1[i][f] != c2a[i][f]:
                if (i, f) not in adj:
                    raise SystemExit(f"not adjudicated: {i} {f}")
                out[i][f] = adj[(i, f)]
    return out


def reference_codes() -> dict:
    r1, r2, adj = read("ref1", REF_FIELDS), read("ref2", REF_FIELDS), adjudications()
    out = {}
    for i in sorted(set(r1) | set(r2)):
        if i not in r1 or i not in r2:
            raise SystemExit(f"reference coding incomplete: {i}")
        out[i] = {f: r1[i][f] if r1[i][f] == r2[i][f] else adj[(i, f)] for f in REF_FIELDS}
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


def binary_agreement(pairs) -> dict:
    """pairs of (full-text positive, reference positive) booleans: sensitivity, specificity, kappa."""
    tp = sum(a and b for a, b in pairs); fn = sum((not a) and b for a, b in pairs)
    tn = sum((not a) and (not b) for a, b in pairs); fp = sum(a and (not b) for a, b in pairs)
    k = kappa(["y" if a else "n" for a, _ in pairs], ["y" if b else "n" for _, b in pairs], ("y", "n"))
    return dict(n=len(pairs), tp=tp, fp=fp, fn=fn, tn=tn,
                sensitivity=tp / (tp + fn) if tp + fn else None, specificity=tn / (tn + fp) if tn + fp else None,
                kappa=k["kappa"])


def a94(papers: list[dict]) -> dict:
    ref = reference_codes()
    by = {p["item"]: p for p in papers}
    out = dict(units=len(ref), reference_distribution={f: dict(Counter(v[f] for v in ref.values())) for f in REF_FIELDS})
    for ft, pos, rf in (("code", "released", "code_used"), ("main_data_access", "open", "data_obtained"),
                        ("weights_release", "yes", "weights_used")):
        pairs = [(by[i][ft] == pos, r[rf] == "yes") for i, r in sorted(ref.items())
                 if by[i][ft] != "unclear" and r[rf] != "unclear"]
        out[f"{ft}_vs_{rf}"] = binary_agreement(pairs)
    return out




def checklist_item5(papers: list[dict]) -> dict:
    """NeurIPS 2025/2026: checklist item 5 answer in the excerpt versus code `released` (Note 9-7)."""
    import re
    from build_p9_sheets import excerpts
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    ex = excerpts()
    pat = re.compile(r"\(item 5\) Question:.*?Answer:\s*\[?\s*(yes|no|n/?a)\b", re.IGNORECASE | re.DOTALL)
    pairs, answers = [], Counter()
    for p in papers:
        if not p["stratum"].startswith("neurips"):
            continue
        r = ex.get(keys[p["item"]]["key"])
        m = pat.search(r["excerpt"]) if r else None
        if not m:
            answers["absent"] += 1
            continue
        a = m.group(1).lower().replace("/", "")
        answers[a] += 1
        if a in ("yes", "no") and p["code"] != "unclear":
            pairs.append((p["code"] == "released", a == "yes"))
    return dict(answers=dict(answers), **binary_agreement(pairs))


def run() -> dict:
    sc = score()
    keys = json.loads(KEYS.read_text(encoding="utf-8"))
    final = final_codes()
    papers = [dict(item=i, group=keys[i]["group"], stratum=keys[i]["stratum"], forum_id=keys[i]["forum_id"], **final[i])
              for i in sorted(final)]
    coverage = defaultdict(lambda: dict(frame=0, coded=0))
    for i, v in keys.items():
        coverage[f"{v['group']}|{v['stratum']}"]["frame"] += 1
        coverage[f"{v['group']}|{v['stratum']}"]["coded"] += i in c1
    res = dict(agreement=sc["agreement"], gate=sc["gate"], agreement_all_below_gate=sc["agreement_all_below_gate"],
               agreement_reference=sc["agreement_reference"], adjudicated=sc["adjudicated"],
               coverage=dict(sorted(coverage.items())), coded=dict(Counter(p["group"] for p in papers)), A9_3={})
    for f in FIELDS:
        res["A9_3"][f] = prevalence(papers, f)
        res["A9_3"][f]["provisional"] = False  # below-gate fields are adjudicated in final_codes (Note 9-6)
    res["labels"] = {}
    for lab, f in LABELS_A93.items():
        ci = res["A9_3"][f]["mh_ci"]
        res["labels"][lab] = dict(field=f, rd=res["A9_3"][f]["mh_rd"], ci=ci,
                                  met=None if res["A9_3"][f]["provisional"] else bool(ci and ci[1] < 0))
    res["A9_4"] = a94(papers)
    res["A9_4"]["neurips_checklist_item5_vs_code"] = checklist_item5(papers)
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
