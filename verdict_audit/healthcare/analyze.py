"""Protocol 3 analysis: coder agreement and prespecified quantities Q1-Q4.

Usage: python analyze.py [coder_tag ...]
Without arguments, computes agreement between coder1 and coder2 and runs
Q1-Q4 for each coder and, if present, for the resolved file.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261017
B = 2000
ARTIFACT = {"derived_artifacts_missing", "weights_missing"}


def read(name):
    with (HERE / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cohen_kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = Counter(a), Counter(b)
    pe = sum(pa[c] * pb[c] for c in cats) / n ** 2
    return dict(n=n, po=round(po, 4), pe=round(pe, 4), kappa=round((po - pe) / (1 - pe), 4) if pe < 1 else None)


def agreement(tag1="coder1", tag2="coder2"):
    p1 = {r["folder"]: r for r in read(f"papers_{tag1}.csv")}
    p2 = {r["folder"]: r for r in read(f"papers_{tag2}.csv")}
    c1 = {(r["folder"], r["claim_no"]): r for r in read(f"claims_{tag1}.csv")}
    c2 = {(r["folder"], r["claim_no"]): r for r in read(f"claims_{tag2}.csv")}
    count1, count2 = Counter(f for f, _ in c1), Counter(f for f, _ in c2)
    count_mismatch = {f: (count1[f], count2[f]) for f in set(count1) | set(count2) if count1[f] != count2[f]}
    keys = sorted(k for k in c1 if k in c2 and k[0] not in count_mismatch)
    folders = sorted(set(p1) & set(p2))
    out = dict(papers=len(folders), aligned_claims=len(keys), claim_count_mismatch=count_mismatch)
    out["healthcare"] = cohen_kappa([p1[f]["healthcare"] for f in folders], [p2[f]["healthcare"] for f in folders])
    out["code_release"] = cohen_kappa([p1[f]["code_release"] for f in folders], [p2[f]["code_release"] for f in folders])
    for field in ("claim_type", "outcome", "primary_blocker", "divergence"):
        out[field] = cohen_kappa([c1[k][field] for k in keys], [c2[k][field] for k in keys])
    out["score_mismatches"] = [list(k) for k in keys if c1[k]["score"] != c2[k]["score"]]
    out["paper_number_mismatches"] = {
        f: {fld: (p1[f][fld], p2[f][fld]) for fld in ("forecast_score", "prepared_score", "prepared_max")
            if p1[f][fld] != p2[f][fld]} for f in folders}
    out["paper_number_mismatches"] = {f: v for f, v in out["paper_number_mismatches"].items() if v}
    out["primary_blocker_disagreements"] = [
        dict(folder=k[0], claim=k[1], coder1=c1[k]["primary_blocker"], coder2=c2[k]["primary_blocker"])
        for k in keys if c1[k]["primary_blocker"] != c2[k]["primary_blocker"]]
    return out


def q_analyses(tag):
    papers = {r["folder"]: r for r in read(f"papers_{tag}.csv")}
    claims = read(f"claims_{tag}.csv")
    hc = [r for r in claims if papers[r["folder"]]["healthcare"] == "yes"]
    out = dict(tag=tag, papers_healthcare=sum(p["healthcare"] == "yes" for p in papers.values()),
               claims_healthcare=len(hc))

    # Q1
    eligible = [r for r in hc if r["outcome"] != "full" and papers[r["folder"]]["code_release"] in ("full", "partial")]
    by_paper = defaultdict(lambda: [0, 0])
    for r in eligible:
        hit = bool(ARTIFACT & set(filter(None, r["blockers"].split(";"))))
        by_paper[r["folder"]][0] += hit
        by_paper[r["folder"]][1] += 1
    table = np.array(list(by_paper.values()), float)
    share = table[:, 0].sum() / table[:, 1].sum()
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(B):
        t = table[rng.integers(0, len(table), len(table))]
        boots.append(t[:, 0].sum() / t[:, 1].sum())
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    out["Q1"] = dict(eligible_claims=len(eligible), papers=len(table), share_artifact_blocked=round(float(share), 4),
                     ci=[round(x, 4) for x in ci],
                     label_code_release_did_not_make_claims_checkable=bool(share >= 0.5 and ci[0] > 0.3))

    # Q2
    cross = defaultdict(Counter)
    for r in hc:
        cross[r["claim_type"]][r["primary_blocker"]] += 1
    out["Q2_primary_blocker_by_claim_type"] = {k: dict(v) for k, v in sorted(cross.items())}
    out["Q2_any_blocker_counts"] = dict(Counter(b for r in hc if r["outcome"] != "full"
                                                for b in filter(None, r["blockers"].split(";"))))
    outcome_by_type = defaultdict(Counter)
    for r in hc:
        outcome_by_type[r["claim_type"]][r["outcome"]] += 1
    out["Q2_outcome_by_claim_type"] = {k: dict(v) for k, v in sorted(outcome_by_type.items())}

    # Q3
    div = defaultdict(list)
    for r in claims:
        if r["divergence"] == "yes":
            div[r["folder"]].append(f'C{r["claim_no"]}: {r["pointer"]}')
    out["Q3"] = dict(papers_with_divergence=len(div), of_papers=len(papers),
                     healthcare_papers_with_divergence=sum(papers[f]["healthcare"] == "yes" for f in div),
                     list={f: v for f, v in sorted(div.items())})

    # Q4
    deltas = {f: int(p["prepared_score"]) - int(p["forecast_score"]) for f, p in papers.items()}
    out["Q4"] = dict(mean_prepared_minus_forecast=round(float(np.mean(list(deltas.values()))), 3),
                     range=[min(deltas.values()), max(deltas.values())],
                     exact=sum(d == 0 for d in deltas.values()), below=sum(d < 0 for d in deltas.values()),
                     above=sum(d > 0 for d in deltas.values()), per_paper=deltas)
    return out


def main():
    results = {}
    if (HERE / "claims_coder2.csv").exists():
        results["agreement"] = agreement()
        k = results["agreement"]["primary_blocker"]["kappa"]
        results["taxonomy_reliable"] = bool(k is not None and k >= 0.60)
    for tag in ("coder1", "coder2", "resolved"):
        if (HERE / f"claims_{tag}.csv").exists():
            results[tag] = q_analyses(tag)
    (HERE / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
