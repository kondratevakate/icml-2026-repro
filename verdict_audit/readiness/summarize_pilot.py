"""Protocol 9 toy pilot (Q9.7): summary of one wave from pilot/*/INPUT.json, results.json and confirm.json.

Reports, as the protocol fixes: the share of theory and mechanism claims that were checkable on synthetic data
(consistent or inconsistent) with a Wilson interval; time to a running pipeline; reasons for blocked claims;
every inconsistent outcome in full; the negative control (other claim types); the confirmations; and, now that
the executors are done, the challenge verdicts for the same claims (descriptive). No label.

Usage: python summarize_pilot.py   -> results/p9_pilot_wave1.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

import p8_common as P
from analyze_p8 import wilson

PILOT = P.HERE / "pilot"
CHECKABLE_TYPES = ("theory", "mechanism")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main() -> dict:
    cv = P.claim_verdicts()
    papers, rows = [], []
    for d in sorted(p for p in PILOT.iterdir() if (p / "results.json").exists()):
        inp, res, conf = load(d / "INPUT.json"), load(d / "results.json"), load(d / "confirm.json")
        confirmed = {c["claim_id"]: c for c in (conf or {}).get("claims", [])}
        texts = {c["claim_id"]: c["text"] for c in inp["claims"]}
        verdicts = {}
        for c in cv.get(inp["forum_id"], []):
            verdicts.setdefault(c["text"], Counter())[c["verdict"]] += 1
        for c in res["claims"]:
            if c["outcome"] in ("consistent", "inconsistent") and c["claim_id"] not in confirmed:
                raise SystemExit(f"{d.name}: {c['claim_id']} has no confirmation yet")
            k = confirmed.get(c["claim_id"], {})
            rows.append(dict(paper=d.name, claim_id=c["claim_id"], type=c["type"], outcome=c["outcome"],
                             rerun_matches=k.get("rerun_matches"), check_tests_claim=k.get("check_tests_claim"),
                             challenge_verdicts=dict(verdicts.get(P.A.norm(texts[c["claim_id"]]), {}))))
        papers.append(dict(paper=d.name, status=res["status"], licence=res["licence"], commit=res["commit"],
                           time_to_running_pipeline_min=res["time_to_running_pipeline_min"],
                           container_seconds=res["container_seconds_total"], blocked_reason=res.get("blocked_reason")))
    target = [r for r in rows if r["type"] in CHECKABLE_TYPES]
    control = [r for r in rows if r["type"] not in CHECKABLE_TYPES]
    checkable = sum(r["outcome"] in ("consistent", "inconsistent") for r in target)
    out = dict(
        wave=1, papers=papers, claims=rows,
        theory_mechanism=dict(n=len(target), outcomes=dict(Counter(r["outcome"] for r in target)),
                              checkable=checkable, share=checkable / len(target) if target else None,
                              wilson=wilson(checkable, len(target))),
        negative_control=dict(n=len(control), outcomes=dict(Counter(r["outcome"] for r in control))),
        confirmations=dict(n=sum(r["rerun_matches"] is not None for r in rows),
                           rerun_matches=sum(bool(r["rerun_matches"]) for r in rows),
                           check_tests_claim=sum(bool(r["check_tests_claim"]) for r in rows)),
        time_to_running_pipeline_min=dict(median=median(p["time_to_running_pipeline_min"] for p in papers),
                                          range=[min(p["time_to_running_pipeline_min"] for p in papers),
                                                 max(p["time_to_running_pipeline_min"] for p in papers)]),
        container_minutes_total=round(sum(p["container_seconds"] for p in papers) / 60, 1),
        inconsistent=[dict(r, claim=next(c["text"] for c in load(PILOT / r["paper"] / "INPUT.json")["claims"]
                                         if c["claim_id"] == r["claim_id"]),
                           rationale=next(c["rationale"] for c in load(PILOT / r["paper"] / "results.json")["claims"]
                                          if c["claim_id"] == r["claim_id"]))
                      for r in rows if r["outcome"] == "inconsistent"],
        blocked=[dict(r, rationale=next(c["rationale"] for c in load(PILOT / r["paper"] / "results.json")["claims"]
                                        if c["claim_id"] == r["claim_id"]))
                 for r in rows if r["outcome"] == "blocked"],
    )
    # Note 9-2: per-paper material stays private (data/ is git-ignored); the public file holds aggregates only.
    private = P.DATA / "p9" / "pilot_private"
    private.mkdir(parents=True, exist_ok=True)
    (private / "p9_pilot_wave1_per_paper.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    public = dict(
        wave=1, papers=len(papers), claims=len(rows), note="aggregate counts only (protocol 9, Note 9-2)",
        all_claims=dict(Counter(r["outcome"] for r in rows)),
        theory_mechanism=out["theory_mechanism"], other_claim_types=out["negative_control"],
        confirmations=out["confirmations"], time_to_running_pipeline_min=out["time_to_running_pipeline_min"],
        container_minutes_total=out["container_minutes_total"],
        # No cross-tabulation with the public challenge verdicts: with the public pilot inputs it can point at a
        # single claim (Note 9-2). It stays in the private per-paper file.
        blocked_reasons_summary="training-dependent behaviour did not leave chance level within the CPU budget")
    (P.HERE / "results" / "p9_pilot_wave1.json").write_text(json.dumps(public, indent=1), encoding="utf-8")
    return public


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
