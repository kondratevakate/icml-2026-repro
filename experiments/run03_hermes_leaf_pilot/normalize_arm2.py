#!/usr/bin/env python3
"""
normalize_arm2.py — bring Vv4XRZDMM0_arm2 artifacts to the SAME canonical schema
as arm1, so the scorer (score_run.py v2) treats both arms identically.

PROBLEM (flagged 2026-08-01, user: "форма должна быть единообразная между армами"):
  arm2 wrote verdicts into `VERDICT_CAP` (not the canonical `verdict` field), and
  used a richer structure (arms/mutation/non_vacuity). The scorer only knew `verdict`,
  so arm2 was scored as all-inconclusive (0/12) even though it had real executed
  experiments + mutation tests.

ALSO fixes a self-report trap in arm2 claim4:
  claim4 wrote VERDICT_CAP="verified" but measured reduction_pct_mean=5.32% vs the
  paper's 34.39%. That is NOT a reproduction — it is a false positive. We downgrade
  it to `inconclusive` with an explicit note, exactly like the artifact-grounded judge
  should.

This script is IDEMPOTENT and NON-DESTRUCTIVE:
  - adds canonical `verdict` field = (cross-checked) VERDICT_CAP, never deletes numbers
  - does NOT touch arm1
  - re-running is a no-op for already-normalized files

Run:  python3 normalize_arm2.py
"""
import json, os, glob, re

ARM2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Vv4XRZDMM0_arm2")

# paper-reported anchor numbers for data/experimental claims (from input_bundle)
PAPER_VALUE = {
    4: 34.39,   # size reduction % (linear classification)
    5: 22.44,   # size reduction % (linear regression)
}
# tolerance: measured must be within this fraction of paper value to count as "verified"
TOL_FRAC = 0.10  # 10%


def normalize_claim(path):
    d = json.load(open(path))
    if not isinstance(d, dict):
        return False
    changed = []
    cap = d.get("VERDICT_CAP")
    # canonical verdict already present?
    if "verdict" not in d and cap is not None:
        d["verdict"] = cap
        changed.append(f"verdict<-VERDICT_CAP:{cap}")

    # data-claim cross-check: if verdict is 'verified' but measured << paper value,
    # downgrade to inconclusive (self-report trap guard)
    claim_no = None
    m = re.search(r"claim(\d+)\.json", os.path.basename(path))
    if m:
        claim_no = int(m.group(1))
    if d.get("verdict") == "verified" and claim_no in PAPER_VALUE:
        measured = None
        if claim_no == 4 and isinstance(d.get("measured"), dict):
            measured = d["measured"].get("reduction_pct_mean")
        # (claim5 uses reduction_pct_mean in measured too; arm2 hasn't produced claim5 yet)
        if measured is not None:
            pv = PAPER_VALUE[claim_no]
            if measured < pv * (1 - TOL_FRAC):
                d["verdict"] = "inconclusive"
                note = (f"self-report 'verified' OVERRIDDEN: measured reduction "
                        f"{measured:.2f}% << paper {pv}% (tol {TOL_FRAC*100:.0f}%)")
                d["verdict_note"] = note
                changed.append(f"DOWNGRADED verified->inconclusive: {note}")

    if changed:
        json.dump(d, open(path, "w"), indent=2, ensure_ascii=False)
        print(f"  {os.path.basename(path)}: " + "; ".join(changed))
        return True
    return False


def main():
    if not os.path.isdir(ARM2):
        print(f"arm2 dir not found: {ARM2}")
        return
    files = sorted(glob.glob(os.path.join(ARM2, "results", "claim*.json")))
    print(f"[normalize_arm2] {len(files)} claim files in {ARM2}/results")
    n = 0
    for f in files:
        if normalize_claim(f):
            n += 1
    print(f"DONE. normalized={n} (already-canonical skipped)")


if __name__ == "__main__":
    main()
