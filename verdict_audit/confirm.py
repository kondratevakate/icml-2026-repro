"""Confirmation run for verdict_audit (PROTOCOL.md, Amendment 1).

Reuses analyze.py functions unchanged on logbooks new after the discovery snapshot.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import analyze as A

NEW = A.ROOT / "data/verdicts_confirmation_2026-10-17/verdicts.json"
NEW_SHA = "652f1c162462d7ad142ff32e2bd13d5853e21c639681ef416971776fdd33c526"
CUTOFF = "2026-07-28T12:51:33+00:00"


def main() -> None:
    started = time.monotonic()
    out_dir = A.HERE / "results_confirmation"
    out_dir.mkdir(exist_ok=False)
    hashes = {"new_snapshot": A.sha256(NEW)}
    if hashes["new_snapshot"] != NEW_SHA:
        sys.exit("Hash mismatch for confirmation snapshot")
    for name, expected in A.INPUTS.items():
        if A.sha256(A.ROOT / name) != expected:
            sys.exit(f"Hash mismatch for {name}")
        hashes[name] = expected

    new = A.load_json(NEW)
    disc = A.load_json(A.ROOT / "verdicts_latest_tmp.json")
    early = A.load_json(A.ROOT / "verdicts_live.json")
    raw_anch = json.loads((A.ROOT / "claims_anchored.json").read_text(encoding="utf-8-sig"))
    raw_loose = json.loads((A.ROOT / "claims.json").read_text(encoding="utf-8-sig"))
    anch = {k: {A.norm(c["text"]) for c in v} for k, v in raw_anch.items()}
    loose = {k: {A.norm(c["text"]) for c in v} for k, v in raw_loose.items()}
    medical = A.load_orids(A.ROOT / "audit/public_repro_audit.csv")

    records = [r for k, r in new.items() if k not in disc and k not in early and r["judged_at"] > CUTOFF]
    logbooks, dropped = A.dedupe(records)

    def rng():
        return np.random.default_rng(A.SEED)

    results = dict(
        snapshot=dict(logbooks_total=len(records), logbooks_dedup=len(logbooks), dropped_by_dedup=dropped,
                      papers=len({r["orid"] for r in records}),
                      reproducers=len({A.namespace(r["space_id"]) for r in records}),
                      judged_from=min(r["judged_at"] for r in records), judged_to=max(r["judged_at"] for r in records),
                      judge_models=sorted({r.get("model") for r in records})),
        C1_anchoring=A.anchoring(logbooks, anch, loose),
        C2_agreement=A.agreement(A.multi_rated(logbooks, anch, A.norm), rng(), permute=True),
        C3_medical_gap=A.medical_gap(logbooks, medical, rng()),
        C4_judge_retest=A.retest(disc, new, rng()),
    )
    c2, c3, c4 = results["C2_agreement"], results["C3_medical_gap"], results["C4_judge_retest"]
    results["prespecified_labels"] = dict(
        substantial_reproducer_disagreement=bool(c2["kappa_ci"][1] < 0.60 and c2["kappa_ci"][1] < c4["kappa_ci"][0]),
        medical_gap_in_verified_share=bool((c3["diff_ci"]["verified"][0] > 0 or c3["diff_ci"]["verified"][1] < 0)
                                           and c3["perm_p_verified"] < 0.05),
    )
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    manifest = dict(protocol_sha256=A.sha256(A.HERE / "PROTOCOL.md"), code_sha256=A.sha256(Path(__file__)),
                    analyze_sha256=A.sha256(Path(A.__file__)),
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=A.ROOT, text=True).strip(),
                    inputs=hashes, cutoff=CUTOFF, seed=A.SEED, resamples=A.B,
                    seconds=round(time.monotonic() - started, 1), results_sha256=A.sha256(out_dir / "results.json"))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(results["prespecified_labels"]), f"{manifest['seconds']}s")


if __name__ == "__main__":
    main()
