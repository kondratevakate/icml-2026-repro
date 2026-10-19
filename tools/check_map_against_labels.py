"""Check the medical reproducibility map (docs/, audit/public_repro_audit.csv, July 2026) against the validated
labels and codes of verdict_audit (post hoc, descriptive; no label).

* Medical set: the map's 703 keyword-selected papers versus the protocol 7 label for ICML 2026 (two blind coders,
  kappa 0.96): agreement over all 6,341 papers, and where the map's extra and missing papers come from.
* Fields, on papers in both sets: the map's GitHub signal versus the declared code link (protocol 7 O1); its
  private/restricted signal versus the restricted-data mention (protocol 7 O3) and versus the full-text data-access
  codes (protocol 8 H8.5, where coded); its weights signal versus the protocol 8 weights code.
* Contest: the map's July leaderboard counts versus the 2026-10-17 verdict snapshot (all namespaces).

Usage: python tools/check_map_against_labels.py   -> audit/map_label_check.json
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READINESS = ROOT / "verdict_audit" / "readiness"
sys.path.insert(0, str(READINESS))
sys.path.insert(0, str(ROOT / "verdict_audit"))
import p8_common as P  # noqa: E402
from analyze_p8 import H85, consensus, kappa  # noqa: E402
from analyze_readiness import outcomes  # noqa: E402

LIFE = ("clinical", "life_science")


def table(pairs) -> dict:
    """2x2 counts and kappa for boolean pairs (map, reference)."""
    c = Counter((bool(a), bool(b)) for a, b in pairs)
    k = kappa(["y" if a else "n" for a, _ in pairs], ["y" if b else "n" for _, b in pairs], ("y", "n"))
    return dict(n=len(pairs), both=c[(True, True)], map_only=c[(True, False)], reference_only=c[(False, True)],
                neither=c[(False, False)], kappa=k["kappa"])


def main() -> dict:
    amap = {r["orid"]: r for r in csv.DictReader((ROOT / "audit" / "public_repro_audit.csv").open(encoding="utf-8"))}
    units = {u["forum_id"]: u for u in json.loads((P.DATA / "papers.json").read_text(encoding="utf-8"))
             if u["stratum"] == "icml-2026"}
    labels = {}
    with (READINESS / "codes" / "labels.csv").open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "cand" and r["unit"].startswith("icml-2026:"):
                labels[r["unit"].split(":", 1)[1]] = r["label"]
    ref = {o: (labels.get(o) in LIFE) for o in units}
    status = {o: ("life_science" if ref[o] else "candidate_labelled_no" if units[o]["candidate"] else "not_a_candidate")
              for o in units}
    res = dict(papers=len(units), map_papers=len(amap), map_papers_not_in_icml_2026=sorted(set(amap) - set(units)),
               protocol7_life_science=sum(ref.values()))
    res["medical_set"] = table([(o in amap, ref[o]) for o in units])
    res["map_papers_by_protocol7"] = dict(Counter(status[o] for o in amap if o in units))
    res["map_only_by_area"] = dict(Counter(amap[o]["area"] for o in amap if o in units and not ref[o]).most_common(8))
    both = [o for o in amap if o in units and ref[o]]
    res["fields_on_shared_papers"] = dict(
        n=len(both),
        code=table([(int(amap[o]["github_repo_count"] or 0) > 0, outcomes(units[o])["o1"]) for o in both]),
        restricted_vs_abstract_mention=table([(amap[o]["private_restricted_data_signal"] == "True", outcomes(units[o])["o3"])
                                              for o in both]))
    keys = json.loads((P.P8 / "keys.json").read_text(encoding="utf-8"))
    f85, _, _ = consensus("h85", tuple(H85), H85, False)
    coded = {keys[i]["orid"]: v for i, v in f85.items()}
    sub = [o for o in both if o in coded]
    access = [(o, coded[o]["main_data_access"]) for o in sub if coded[o]["main_data_access"] != "unclear"]
    res["fields_on_shared_papers"]["full_text_subset"] = dict(
        n=len(sub),
        restricted_vs_full_text=table([(amap[o]["private_restricted_data_signal"] == "True",
                                        a in ("credentialed_or_request", "private")) for o, a in access]),
        map_access_by_full_text=dict(Counter(f"{amap[o]['data_access_signal']} | {a}" for o, a in access).most_common(12)),
        weights_vs_full_text=table([(amap[o]["weights_signal"] == "True", coded[o]["weights_release"] == "yes")
                                    for o in sub if coded[o]["weights_release"] != "unclear"]))
    logs, _ = P.A.dedupe(list(P.A.load_json(P.VERDICTS).values()))
    attempted, decided = Counter(), Counter()
    for r in logs:
        attempted[r["orid"]] += 1
        decided[r["orid"]] += any(P.A.category(c.get("verdict")) in P.DECISIVE for c in r.get("claims") or [])
    res["contest"] = dict(
        map_july=dict(attempted=sum(int(amap[o]["leaderboard_logbooks"] or 0) > 0 for o in amap),
                      decided=sum((int(amap[o]["leaderboard_verified_claim_events"] or 0)
                                   + int(amap[o]["leaderboard_falsified_claim_events"] or 0)) > 0 for o in amap)),
        snapshot_oct_on_map_set=dict(attempted=sum(attempted[o] > 0 for o in amap),
                                     decided=sum(decided[o] > 0 for o in amap)),
        snapshot_oct_on_protocol7_set=dict(n=sum(ref.values()), attempted=sum(attempted[o] > 0 for o in units if ref[o]),
                                           decided=sum(decided[o] > 0 for o in units if ref[o])))
    (ROOT / "audit" / "map_label_check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
