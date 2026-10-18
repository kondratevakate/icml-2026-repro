"""POST HOC diagnostics used in the paper draft (not part of any frozen protocol).

Counts only: judge-latency tail, concentration of H4 dissent pairs, resubmission structure,
empty-evidence (back-filled) verdicts, and local trace indexes. Writes results/review_checks.json.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

import common as C
import resubmission_events as R
from analyze_exposure import ts

A = C.A


def main() -> None:
    C.check_inputs()
    logbooks, _, _, anchored, _ = C.load()
    by_sid = {r["space_id"]: r for r in logbooks}
    multi = C.multi_rated_claims(logbooks, anchored)
    recs = R.load_records()
    ctime = R.commit_times()
    start = {s: min(v.values()) for s, v in ctime.items() if v}

    lat = []
    for sid, rs in recs.items():
        if sid in by_sid:
            j, sha = min(rs)
            if sha in ctime.get(sid, {}):
                lat.append((ts(j) - ctime[sid][sha]) / 3600)
    lat = np.array(lat)

    pairs, papers = Counter(), Counter()
    for (orid, t), raters in multi.items():
        dated = sorted((s for s in raters if s in start), key=lambda s: (start[s], s))
        for b in dated[1:]:
            if raters[b] == "falsified" and raters[dated[0]] != "falsified":
                pairs[(orid, t)] += 1
                papers[orid] += 1
    n_pairs = sum(pairs.values())

    events = R.events(recs)
    team_events = Counter(e["team"] for e in events if e["kind"] == "new_sha")
    tot = sum(team_events.values())
    top = team_events.most_common()

    empty = Counter()
    total_claims = 0
    for r in logbooks:
        for c in r.get("claims") or []:
            total_claims += 1
            if not (c.get("evidence") or "").strip():
                empty[A.category(c.get("verdict"))] += 1

    skipped = []
    indexes = []
    for root, dirs, files in os.walk(C.ROOT, onerror=skipped.append):
        dirs[:] = [d for d in dirs if d != ".git"]
        if os.path.basename(root) == "traces" and "index.json" in files:
            indexes.append(os.path.join(root, "index.json"))
    empty_idx = sum(json.loads(open(f, encoding="utf-8").read()).get("sessions") == [] for f in indexes)

    out = dict(
        note="post hoc diagnostics; counts only",
        judge_latency_hours=dict(n=len(lat), share_gt_24h=float((lat > 24).mean()), share_gt_7d=float((lat > 168).mean())),
        h4_falsified_dissent=dict(n_pairs=n_pairs, n_claims=len(pairs), n_papers=len(papers),
                                  top_claim_share=max(pairs.values()) / n_pairs,
                                  top10_papers_share=sum(v for _, v in papers.most_common(10)) / n_pairs),
        resubmission=dict(events=dict(Counter(e["kind"] for e in events)),
                          spaces_new_sha=len({e["space"] for e in events if e["kind"] == "new_sha"}),
                          teams_new_sha=len(team_events), top1_share=top[0][1] / tot,
                          top5_share=sum(v for _, v in top[:5]) / tot, top10_share=sum(v for _, v in top[:10]) / tot),
        empty_evidence=dict(total_claims=total_claims, by_verdict=dict(empty)),
        local_trace_indexes=dict(n=len(indexes), empty_sessions=empty_idx, dirs_skipped_unreadable=len(skipped)),
    )
    (C.HERE / "results" / "review_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
