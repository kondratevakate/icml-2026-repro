"""Protocol 4 analysis: E0 label validity, dictionary gates, H1 model, sensitivities.

Stage 1 (`python analyze_e1.py validate`): coder agreement, disagreement list for
adjudication, dictionary gates, E0 PPV/NPV. Needs codes_A_*, codes_B_* and, for
consensus, adjudication_*.csv (item,label,reason) covering every disagreement.
Stage 2 (`python analyze_e1.py models`): H1 and all prespecified secondary and
sensitivity analyses. Writes results_e1.json; refuses to overwrite it.
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.conditional_models import ConditionalLogit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
import analyze as A  # noqa: E402

SNAPSHOT = ROOT / "data/verdicts_confirmation_2026-10-17/verdicts.json"
SNAPSHOT_SHA = "652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526"
CUTOFF = "2026-07-28T12:51:33+00:00"
ORGANISER = "ICML-2026-agent-repro"
SEED = 20261017
B = 2000

THEORY_RE = re.compile(
    r"\b(theorem|lemma|proposition|corollary|proof|prove[sd]?|provabl\w*|bound(s|ed)?|guarantee\w*|converge\w*"
    r"|regret|minimax|consisten(t|cy)|identifiab\w*|asymptotic\w*|sample complexity|o\(|theta\(|omega\()",
    re.I)
REASONS = [
    ("artifact_absent", re.compile(
        r"not (been )?released|unreleased|not (publicly )?available|unavailable|absent|withheld|not provided"
        r"|no (official )?(code|checkpoint|weights|data|dataset|predictions|outputs)|proprietary|restricted|access", re.I)),
    ("not_attempted", re.compile(
        r"not attempted|not (re)?run|did not run|never (ran|run)|no experiments?|blocked|not executed"
        r"|no (numerical )?results", re.I)),
    ("reduced_scale", re.compile(r"toy|small[- ]scale|subset|reduced|synthetic|smaller|fewer", re.I)),
]


def reason(text: str) -> str:
    return next((name for name, rx in REASONS if rx.search(text or "")), "other")


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n ** 2
    return dict(n=n, po=round(po, 4), kappa=round((po - pe) / (1 - pe), 4) if pe < 1 else None)


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def consensus(task):
    a = {r["item"]: r["label"] for r in read(HERE / f"codes_A_{task}.csv")}
    b = {r["item"]: r["label"] for r in read(HERE / f"codes_B_{task}.csv")}
    assert set(a) == set(b) and len(a) == 200
    adj_path = HERE / f"adjudication_{task}.csv"
    adj = {r["item"]: r["label"] for r in read(adj_path)} if adj_path.exists() else {}
    items = sorted(a, key=int)
    disagreements = [i for i in items if a[i] != b[i]]
    missing = [i for i in disagreements if i not in adj]
    cons = {i: (a[i] if a[i] == b[i] else adj.get(i)) for i in items}
    return a, b, cons, disagreements, missing


def validate():
    out = {}
    sheets = {t: {r["item"]: r for r in read(HERE / f"sheets/{f}")}
              for t, f in [("e0", "e0_papers.csv"), ("claims", "e1_claims.csv"), ("evidence", "e1_evidence.csv")]}
    for task in ("e0", "claims", "evidence"):
        a, b, cons, dis, missing = consensus(task)
        items = sorted(a, key=int)
        res = dict(coder_agreement=kappa([a[i] for i in items], [b[i] for i in items]),
                   disagreements=len(dis), unadjudicated=missing)
        if not missing:
            if task == "claims":
                dic = ["theory" if THEORY_RE.search(sheets[task][i]["claim"]) else "empirical" for i in items]
                res["dictionary_vs_consensus"] = kappa(dic, [cons[i] for i in items])
                res["gate_passed"] = res["dictionary_vs_consensus"]["kappa"] >= 0.60
            if task == "evidence":
                dic = [reason(sheets[task][i]["evidence"]) for i in items]
                res["dictionary_vs_consensus"] = kappa(dic, [cons[i] for i in items])
                res["gate_passed"] = res["dictionary_vs_consensus"]["kappa"] >= 0.60
            if task == "e0":
                key = {r["item"]: r["audit_label"] for r in read(HERE / "keys/e0_key.csv")}
                true_med = {i: cons[i] in ("clinical", "life_science") for i in items}
                med_items = [i for i in items if key[i] == "medical"]
                oth_items = [i for i in items if key[i] == "other"]
                tp = sum(true_med[i] for i in med_items)
                tn = sum(not true_med[i] for i in oth_items)
                res["ppv"] = dict(k=tp, n=len(med_items), est=round(tp / len(med_items), 4), ci=wilson(tp, len(med_items)))
                res["npv"] = dict(k=tn, n=len(oth_items), est=round(tn / len(oth_items), 4), ci=wilson(tn, len(oth_items)))
                res["consensus_counts_by_audit_label"] = {
                    lab: dict(Counter(cons[i] for i in items if key[i] == lab)) for lab in ("medical", "other")}
        res["disagreement_items"] = dis
        out[task] = res
    (HERE / "validation_e1.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def frame():
    assert hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest() == SNAPSHOT_SHA
    logbooks, _ = A.dedupe(list(A.load_json(SNAPSHOT).values()))
    medical = A.load_orids(ROOT / "audit/public_repro_audit.csv")
    narrow = A.load_orids(ROOT / "medical_neuro_papers.csv")
    rows = []
    for r in logbooks:
        ns = A.namespace(r["space_id"])
        for j, c in enumerate(r.get("claims") or []):
            v = A.category(c.get("verdict"))
            rows.append(dict(reproducer=ns, paper=r["orid"], verified=int(v == "verified"), verdict=v,
                             medical=int(r["orid"] in medical), narrow=int(r["orid"] in narrow),
                             theory=int(bool(THEORY_RE.search(A.norm(c.get("claim", ""))))),
                             discovery=int(r["judged_at"] <= CUTOFF), organiser=int(ns == ORGANISER),
                             evidence=str(c.get("evidence") or "")))
    df = pd.DataFrame(rows)
    df["w_paper"] = 1.0 / df.groupby("paper")["verified"].transform("size")
    return df


def lpm(df, med_col="medical", theory=True, weights=None):
    mixed = df.groupby("reproducer")[med_col].transform(lambda s: s.nunique() > 1)
    d = df[mixed].copy()
    cols = [med_col] + (["theory"] if theory else [])
    X = pd.concat([d[cols].astype(float), pd.get_dummies(d["reproducer"], drop_first=False, dtype=float)], axis=1)
    groups = np.column_stack([pd.factorize(d["paper"])[0], pd.factorize(d["reproducer"])[0]])
    model = sm.WLS(d["verified"].astype(float), X, weights=weights[mixed] if weights is not None else 1.0)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    ci = fit.conf_int().loc[med_col].tolist()
    return dict(estimate=round(float(fit.params[med_col]), 4), ci=[round(float(x), 4) for x in ci],
                p=float(fit.pvalues[med_col]), n_claims=int(len(d)), n_reproducers=int(d["reproducer"].nunique()),
                n_papers=int(d["paper"].nunique()), medical_claims=int(d[med_col].sum()),
                theory_coef=round(float(fit.params["theory"]), 4) if theory else None)


def clogit(df, theory=True):
    """Amendment 1: logit with reproducer fixed effects, two-way cluster-robust CI."""
    mixed = df.groupby("reproducer")["medical"].transform(lambda s: s.nunique() > 1)
    varying = df.groupby("reproducer")["verified"].transform(lambda s: s.nunique() > 1)
    d = df[mixed & varying].reset_index(drop=True)
    cols = ["medical"] + (["theory"] if theory else [])
    X = pd.concat([d[cols].astype(float), pd.get_dummies(d["reproducer"], drop_first=False, dtype=float)], axis=1)
    groups = np.column_stack([pd.factorize(d["paper"])[0], pd.factorize(d["reproducer"])[0]])
    fit = sm.Logit(d["verified"].astype(float), X).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": groups})
    lo, hi = fit.conf_int().loc["medical"].tolist()
    return dict(model="fixed-effects logit (amendment 1)", log_or=round(float(fit.params["medical"]), 4),
                odds_ratio=round(float(np.exp(fit.params["medical"])), 4),
                or_ci=[round(float(np.exp(lo)), 4), round(float(np.exp(hi)), 4)], p=float(fit.pvalues["medical"]),
                n_claims=int(len(d)), converged=bool(fit.mle_retvals.get("converged", False)))


def reasons(df):
    d = df[df["verdict"].isin(["inconclusive", "toy"])].copy()
    d["reason"] = d["evidence"].map(reason)
    cats = ["artifact_absent", "not_attempted", "reduced_scale", "other"]
    tab = d.groupby(["paper", "reason"]).size().unstack(fill_value=0).reindex(columns=cats, fill_value=0)
    med = df.groupby("paper")["medical"].first().reindex(tab.index).values.astype(bool)
    counts = tab.values.astype(float)

    def shares(rows):
        s = rows.sum(0)
        return s / s.sum()

    rng = np.random.default_rng(SEED)
    mi, oi = np.where(med)[0], np.where(~med)[0]
    boots = np.array([shares(counts[rng.choice(mi, len(mi))]) - shares(counts[rng.choice(oi, len(oi))]) for _ in range(B)])
    diff = shares(counts[mi]) - shares(counts[oi])
    return dict(share_medical=dict(zip(cats, shares(counts[mi]).round(4).tolist())),
                share_other=dict(zip(cats, shares(counts[oi]).round(4).tolist())),
                diff=dict(zip(cats, diff.round(4).tolist())),
                diff_ci={c: [round(float(x), 4) for x in np.percentile(boots[:, k], [2.5, 97.5])] for k, c in enumerate(cats)},
                n_verdicts=int(len(d)))


def label_corrected(est, validation):
    # Observed difference = true difference * (PPV + NPV - 1); bootstrap PPV and NPV from E0 counts.
    ppv, npv = validation["e0"]["ppv"], validation["e0"]["npv"]
    rng = np.random.default_rng(SEED)
    f = rng.binomial(ppv["n"], ppv["k"] / ppv["n"], B) / ppv["n"] + rng.binomial(npv["n"], npv["k"] / npv["n"], B) / npv["n"] - 1
    point = est["estimate"] / (ppv["est"] + npv["est"] - 1)
    corrected = est["estimate"] / f[f > 0.05]
    return dict(factor=round(ppv["est"] + npv["est"] - 1, 4), corrected_estimate=round(point, 4),
                factor_uncertainty_interval=[round(float(x), 4) for x in np.percentile(corrected, [2.5, 97.5])],
                note="propagates E0 sampling error only, not the model CI")


def models():
    out_path = HERE / "results_e1.json"
    if out_path.exists():
        sys.exit("results_e1.json exists; refusing to overwrite")
    validation = json.loads((HERE / "validation_e1.json").read_text(encoding="utf-8"))
    use_theory = bool(validation["claims"].get("gate_passed"))
    df = frame()
    res = dict(theory_covariate_used=use_theory, n_claims=int(len(df)), n_reproducers=int(df["reproducer"].nunique()))
    res["H1_primary"] = lpm(df, theory=use_theory)
    res["secondary_clogit"] = clogit(df, theory=use_theory)
    res["sens_no_theory"] = lpm(df, theory=False)
    res["sens_exclude_organiser"] = lpm(df[df["organiser"] == 0], theory=use_theory)
    res["sens_narrow_label"] = lpm(df, med_col="narrow", theory=use_theory)
    res["sens_equal_paper_weight"] = lpm(df, theory=use_theory, weights=df["w_paper"])
    res["sens_discovery_period"] = lpm(df[df["discovery"] == 1], theory=use_theory)
    res["sens_after_discovery"] = lpm(df[df["discovery"] == 0], theory=use_theory)
    res["sens_label_corrected"] = label_corrected(res["H1_primary"], validation)
    res["reasons_exploratory"] = reasons(df)
    res["reasons_validated"] = bool(validation["evidence"].get("gate_passed"))
    res["decision_rule_medical_gap_headline"] = bool(res["H1_primary"]["ci"][1] < 0 or res["H1_primary"]["ci"][0] > 0)
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    {"validate": validate, "models": models}[sys.argv[1]]()
