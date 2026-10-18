"""Protocol 8 scoring (kappa, consensus, adjudication lists) and analyses H8.1-H8.5.

Usage: python analyze_p8.py score   -> results/p8_coding.json, codes/p8/to_adjudicate_*.csv
       python analyze_p8.py run     -> results/p8.json (+ p8_manifest.json); needs complete adjudication
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import p8_common as P

sys.path.insert(0, str(P.HERE.parent / "e1_adjusted"))
sys.path.insert(0, str(P.HERE.parent / "falsifiability"))
from analyze_e1 import THEORY_RE  # noqa: E402  (protocol 4 dictionary, validated)

H83 = ("data_unavailable", "artifacts_missing", "code_missing_or_broken", "reduced_scale_or_compute", "not_attempted",
       "underspecified_paper", "other")
H84 = ("theory", "mechanism", "data_protocol", "headline_performance", "secondary_empirical")
H85 = dict(main_data_access=("open", "credentialed_or_request", "private", "synthetic_only", "unclear"),
           open_data_experiment=("yes", "no", "unclear"), synthetic_or_sample_release=("yes", "no", "unclear"),
           weights_release=("yes", "no", "unclear"))
GATE = 0.80


def read_codes(task: str, role: str, fields: tuple[str, ...] | None = None) -> dict:
    out = {}
    for f in sorted(P.CODES.glob(f"{task}_{role}_batch_*.csv")):
        with f.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["item_id"] in out:
                    raise SystemExit(f"{f.name}: duplicate {r['item_id']}")
                out[r["item_id"]] = {k: r[k].strip().lower() for k in (fields or ("label",))}
    return out


def kappa(a: list[str], b: list[str], cats) -> dict:
    n = len(a)
    if not n:
        return dict(n=0, po=None, kappa=None)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / n ** 2
    return dict(n=n, po=po, kappa=(po - pe) / (1 - pe) if pe < 1 else None)


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, c - h), min(1.0, c + h)]


def adjudications(task: str) -> dict:
    path = P.CODES / f"adjudication_{task}.csv"
    out = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if not r.get("reason", "").strip():
                    raise SystemExit(f"adjudication without reason: {r}")
                out[(r["item_id"], r.get("field", "label"))] = r["label"]
    return out


def consensus(task: str, fields, cats_by_field, overlap_only: bool) -> tuple[dict, dict, list]:
    """Return (final codes {item: {field: label}}, agreement {field: kappa}, pending [(item, field, c1, c2)])."""
    c1, c2 = read_codes(task, "coder1", fields), read_codes(task, "coder2", fields)
    keys = {k for k, v in json.loads((P.P8 / "keys.json").read_text(encoding="utf-8")).items() if v["task"] == task}
    if set(c1) != keys:
        raise SystemExit(f"{task}: coder 1 coverage {len(c1)} of {len(keys)}")
    if not overlap_only and set(c2) != keys:
        raise SystemExit(f"{task}: coder 2 coverage {len(c2)} of {len(keys)}")
    adj = adjudications(task)
    shared = sorted(set(c1) & set(c2))
    agreement, pending, final = {}, [], {}
    for f in fields:
        bad = {v[f] for v in list(c1.values()) + list(c2.values())} - set(cats_by_field[f])
        if bad:
            raise SystemExit(f"{task}.{f}: unknown labels {bad}")
        agreement[f] = kappa([c1[i][f] for i in shared], [c2[i][f] for i in shared], cats_by_field[f])
    for i in keys:
        final[i] = {}
        for f in fields:
            if i in c2 and c1[i][f] != c2[i][f]:
                lab = adj.get((i, f))
                if lab is None:
                    pending.append((i, f, c1[i][f], c2[i][f]))
                final[i][f] = lab
            else:
                final[i][f] = c1[i][f]
    return final, agreement, pending


def score() -> dict:
    out = {}
    for task, fields, cats, overlap in (("h83", ("label",), {"label": H83}, False),
                                        ("h84", ("label",), {"label": H84}, True),
                                        ("h85", tuple(H85), H85, False)):
        try:
            final, agr, pending = consensus(task, fields, cats, overlap)
        except SystemExit as exc:
            out[task] = dict(status=str(exc))
            continue
        with (P.CODES / f"to_adjudicate_{task}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["item_id", "field", "coder1", "coder2"])
            w.writerows(pending)
        out[task] = dict(agreement=agr, pending=len(pending),
                         gate_met=(agr["label"]["kappa"] or 0) >= GATE if task == "h84" else None)
    (P.HERE / "results" / "p8_coding.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# ---------- analyses ----------

def boot_paper(rows_by_paper: dict, stat, rng) -> list[float]:
    """Paper-cluster bootstrap within group: rows_by_paper = {group: {orid: [rows]}}."""
    vals = []
    for _ in range(P.B):
        sample = {}
        for g, papers in rows_by_paper.items():
            ids = list(papers)
            pick = [ids[i] for i in rng.integers(0, len(ids), len(ids))]
            sample[g] = [r for o in pick for r in papers[o]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def share(rows, cond=lambda r: True):
    sel = [r["decisive"] for r in rows if cond(r)]
    return (sum(sel) / len(sel)) if sel else None


def gap(sample, cond):
    m, o = share(sample.get("medical", []), cond), share(sample.get("other", []), cond)
    return None if m is None or o is None else m - o


def run() -> dict:
    sc = score()
    for t in ("h83", "h84", "h85"):
        if sc[t].get("pending") or "status" in sc[t]:
            raise SystemExit(f"{t} not ready: {sc[t]}")
    if not sc["h84"]["gate_met"]:
        raise SystemExit("H8.4 gate not met: coder 2 must code all items (protocol)")
    grp, cv = P.groups(), P.claim_verdicts()
    clin = P.clinical_ids()
    attempted = {o: g for o, g in grp.items() if o in cv}
    keys = json.loads((P.P8 / "keys.json").read_text(encoding="utf-8"))
    res = dict(attempted=dict(Counter(attempted.values())))

    # claim-verdict rows
    rows = defaultdict(lambda: defaultdict(list))
    for o, g in attempted.items():
        for c in cv[o]:
            rows[g][o].append(dict(orid=o, text=c["text"], decisive=int(c["verdict"] in P.DECISIVE),
                                   theory=bool(THEORY_RE.search(c["text"]))))

    def h81(rb, label_rule=True):
        rng = np.random.default_rng(P.SEED_BOOT)
        flat = {g: [r for p in v.values() for r in p] for g, v in rb.items()}
        ge = gap(flat, lambda r: not r["theory"])
        gt = gap(flat, lambda r: r["theory"])
        did = None if ge is None or gt is None else ge - gt
        out = dict(gap_empirical=ge, gap_empirical_ci=boot_paper(rb, lambda s: gap(s, lambda r: not r["theory"]), rng),
                   gap_theory=gt, gap_theory_ci=boot_paper(rb, lambda s: gap(s, lambda r: r["theory"]), rng),
                   did=did, did_ci=boot_paper(rb, lambda s: (lambda a, b: None if a is None or b is None else a - b)(
                       gap(s, lambda r: not r["theory"]), gap(s, lambda r: r["theory"])), rng),
                   counts={g: dict(theory=sum(r["theory"] for r in flat[g]), empirical=sum(not r["theory"] for r in flat[g]))
                           for g in flat})
        if label_rule:
            out["label_gap_concentrated_in_empirical"] = bool(out["did_ci"] and out["did_ci"][1] < 0)
        return out

    res["H8_1"] = h81(rows)
    res["H8_1_clinical_only"] = h81({"medical": {o: v for o, v in rows["medical"].items() if o in clin},
                                     "other": rows["other"]}, label_rule=False)

    # H8.2: restricted-data mention (protocol 7 O3) and paper decidability
    from analyze_readiness import outcomes
    units = {u["forum_id"]: u for u in json.loads((P.DATA / "papers.json").read_text(encoding="utf-8"))
             if u["stratum"] == "icml-2026"}
    papers = [dict(orid=o, g=g, o3=outcomes(units[o])["o3"], decided=int(any(r["decisive"] for r in rows[g][o])))
              for o, g in attempted.items()]

    def mh(papers_, strat_key):
        t = defaultdict(lambda: [0, 0, 0, 0])
        for p in papers_:
            cell = t[p[strat_key]]
            if p["g"] == "medical":
                cell[0] += 1; cell[1] += p["decided"]
            else:
                cell[2] += 1; cell[3] += p["decided"]
        num = den = 0.0
        for ne, ae, nu, au in t.values():
            if ne and nu:
                num += (ae * nu - au * ne) / (ne + nu); den += ne * nu / (ne + nu)
        return num / den if den else None

    def boot_papers(papers_, stat):
        rng = np.random.default_rng(P.SEED_BOOT)
        by = defaultdict(list)
        for p in papers_:
            by[(p["g"], p.get("stratum_boot", 0))].append(p)
        vals = []
        for _ in range(P.B):
            s = [x for v in by.values() for x in (v[i] for i in rng.integers(0, len(v), len(v)))]
            v = stat(s)
            if v is not None:
                vals.append(v)
        return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None

    cells = {}
    for g in ("medical", "other"):
        for v in (1, 0):
            sel = [p["decided"] for p in papers if p["g"] == g and p["o3"] == v]
            cells[f"{g}_o3_{v}"] = dict(n=len(sel), decided=sum(sel) / len(sel) if sel else None,
                                        wilson=wilson(sum(sel), len(sel)))
    for p in papers:
        p["all"] = 0
    res["H8_2"] = dict(cells=cells, unadjusted=mh(papers, "all"), unadjusted_ci=boot_papers(papers, lambda s: mh(s, "all")),
                       adjusted=mh(papers, "o3"), adjusted_ci=boot_papers(papers, lambda s: mh(s, "o3")))

    # H8.3: judge's reasons
    f83, a83, _ = consensus("h83", ("label",), {"label": H83}, False)
    by = defaultdict(list)
    for i, v in f83.items():
        by[keys[i]["group"]].append((keys[i]["orid"], v["label"]))
    h83 = dict(agreement=a83["label"], shares={})
    for g, lst in by.items():
        c = Counter(l for _, l in lst)
        h83["shares"][g] = {k: dict(n=c[k], share=c[k] / len(lst), wilson=wilson(c[k], len(lst))) for k in H83}
    for reason_ in ("data_unavailable", "artifacts_missing"):
        rng = np.random.default_rng(P.SEED_BOOT)
        clus = {g: defaultdict(list) for g in by}
        for g, lst in by.items():
            for o, l in lst:
                clus[g][o].append(int(l == reason_))

        def stat(sample):
            m, o_ = sample.get("medical", []), sample.get("other", [])
            return (sum(m) / len(m) - sum(o_) / len(o_)) if m and o_ else None
        vals = []
        for _ in range(P.B):
            s = {}
            for g, papers_ in clus.items():
                ids = list(papers_)
                s[g] = [x for i in rng.integers(0, len(ids), len(ids)) for x in papers_[ids[i]]]
            v = stat(s)
            if v is not None:
                vals.append(v)
        obs = h83["shares"]["medical"][reason_]["share"] - h83["shares"]["other"][reason_]["share"]
        h83[f"diff_{reason_}"] = dict(diff=obs, ci=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))])
    res["H8_3"] = h83

    # H8.4: claim-type map and asymmetry
    f84, a84, _ = consensus("h84", ("label",), {"label": H84}, True)
    ctype = {(keys[i]["orid"], keys[i]["text"]): v["label"] for i, v in f84.items()}
    rb = defaultdict(lambda: defaultdict(list))
    for g in rows:
        for o, lst in rows[g].items():
            for r in lst:
                t = ctype.get((o, r["text"]))
                if t:
                    rb[g][o].append(dict(r, ctype=t))
    rng = np.random.default_rng(P.SEED_BOOT)
    flat = {g: [r for p in v.values() for r in p] for g, v in rb.items()}
    by_type = {}
    for t in H84:
        by_type[t] = dict(share_medical=share(flat["medical"], lambda r, t=t: r["ctype"] == t),
                          share_other=share(flat["other"], lambda r, t=t: r["ctype"] == t),
                          n_medical=sum(r["ctype"] == t for r in flat["medical"]),
                          n_other=sum(r["ctype"] == t for r in flat["other"]),
                          gap=gap(flat, lambda r, t=t: r["ctype"] == t),
                          gap_ci=boot_paper(rb, lambda s, t=t: gap(s, lambda r: r["ctype"] == t), rng))
    data_dep = lambda r: r["ctype"] in ("headline_performance", "secondary_empirical")
    checkable = lambda r: r["ctype"] in ("theory", "mechanism")
    g_ck = gap(flat, checkable)
    g_dd = gap(flat, data_dep)
    contrast_ci = boot_paper(rb, lambda s: (lambda a, b: None if a is None or b is None else a - b)(gap(s, data_dep), gap(s, checkable)), rng)
    g_ck_ci = boot_paper(rb, lambda s: gap(s, checkable), rng)
    res["H8_4"] = dict(agreement=a84["label"], by_type=by_type, gap_checkable=g_ck, gap_checkable_ci=g_ck_ci,
                       gap_data_dependent=g_dd, contrast=None if g_ck is None or g_dd is None else g_dd - g_ck,
                       contrast_ci=contrast_ci,
                       label_checkable_decided_equally_data_dependent_not=bool(
                           g_ck_ci and g_ck_ci[0] <= 0 <= g_ck_ci[1] and contrast_ci and contrast_ci[1] < 0))

    # H8.5: full text coding, prevalence, policy test
    f85, a85, _ = consensus("h85", tuple(H85), H85, False)
    h85_papers = json.loads((P.P8 / "h85_papers.json").read_text(encoding="utf-8"))
    status = Counter()
    for p in h85_papers:
        path = P.P8 / "fulltext" / f"{p['orid']}.json"
        status[json.loads(path.read_text(encoding="utf-8"))["status"] if path.exists() else "missing"] += 1
    coded = [dict(orid=keys[i]["orid"], g=keys[i]["group"], **v) for i, v in f85.items()]
    dec = {o: int(any(r["decisive"] for r in rows[g][o])) for o, g in attempted.items()}
    prev = {}
    for g in ("medical", "other"):
        sel = [c for c in coded if c["g"] == g]
        prev[g] = {f: dict(Counter(c[f] for c in sel)) for f in H85}
        prev[g]["n"] = len(sel)
    hidden = [c for c in coded if c["g"] == "medical" and c["main_data_access"] == "private" and c["open_data_experiment"] == "no"]
    pol = [dict(g=c["g"], ode=1 if c["open_data_experiment"] == "yes" else 0, decided=dec[c["orid"]])
           for c in coded if c["open_data_experiment"] in ("yes", "no")]
    cells = {}
    for g in ("medical", "other"):
        for v in (1, 0):
            sel = [p["decided"] for p in pol if p["g"] == g and p["ode"] == v]
            cells[f"{g}_open_{v}"] = dict(n=len(sel), decided=sum(sel) / len(sel) if sel else None, wilson=wilson(sum(sel), len(sel)))

    def mh_open(s):
        t = defaultdict(lambda: [0, 0, 0, 0])
        for p in s:
            c = t[p["g"]]
            if p["ode"]:
                c[0] += 1; c[1] += p["decided"]
            else:
                c[2] += 1; c[3] += p["decided"]
        num = den = 0.0
        for ne, ae, nu, au in t.values():
            if ne and nu:
                num += (ae * nu - au * ne) / (ne + nu); den += ne * nu / (ne + nu)
        return num / den if den else None
    rng = np.random.default_rng(P.SEED_BOOT)
    byc = defaultdict(list)
    for p in pol:
        byc[(p["g"], p["ode"])].append(p)
    vals = [mh_open([x for v in byc.values() for x in (v[i] for i in rng.integers(0, len(v), len(v)))]) for _ in range(P.B)]
    vals = [v for v in vals if v is not None]
    ci = [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None
    res["H8_5"] = dict(agreement=a85, fulltext_status=dict(status), prevalence=prev,
                       hidden_science=dict(n=len(hidden), share_of_coded_medical=len(hidden) / prev["medical"]["n"] if prev["medical"]["n"] else None,
                                           wilson=wilson(len(hidden), prev["medical"]["n"])),
                       policy_cells=cells, policy_mh_diff=mh_open(pol), policy_mh_ci=ci,
                       unclear_excluded=sum(1 for c in coded if c["open_data_experiment"] == "unclear"),
                       label_open_data_experiment_decided_more_often=bool(ci and ci[0] > 0))

    out = P.HERE / "results" / "p8.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (P.HERE / "results" / "p8_manifest.json").write_text(json.dumps(dict(
        protocol_sha256=sha(P.HERE / "PROTOCOL.md"), code_sha256={f: sha(P.HERE / f) for f in ("analyze_p8.py", "p8_common.py", "build_p8.py", "fetch_p8_fulltext.py")},
        keys_sha256=sha(P.P8 / "keys.json"), git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=P.ROOT, text=True).strip(),
        seed_boot=P.SEED_BOOT, seed_sample=P.SEED_SAMPLE, resamples=P.B, results_sha256=sha(out)), indent=2), encoding="utf-8")
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    print(json.dumps(score() if mode == "score" else run(), indent=1)[:6000])
