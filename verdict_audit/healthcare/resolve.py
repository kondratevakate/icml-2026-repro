"""Resolve coder disagreements after agreement was computed (protocol 3).

Starts from coder 1 and applies the decisions below; every other code is
unchanged. Each decision cites the codebook definition it follows.
"""
import csv

D, W, C = "derived_artifacts_missing", "weights_missing", "code_missing_or_broken"
PRECEDENCE = ["paper_code_divergence", C, "raw_data_unavailable", W, D,
              "compute_or_scale", "underspecified_paper", "other"]

# (folder, claim_no): (new blockers or None to keep, new divergence or None, reason)
DECISIONS = {
    ("dpsurv", "1"): (None, None, "Keep coder 1: TCGA WSIs are public; the missing inputs are derived UNI2/PANTHER features."),
    ("flow_density_ratio", "6"): ([W, D], None, "Accept coder 2: the trained model is absent, so weights_missing applies and takes precedence."),
    ("glean", "2"): (None, None, "Keep coder 1: no author code is released at all, a reason the audit states for every empirical claim."),
    ("glean", "3"): (None, None, "Keep coder 1: as glean C2."),
    ("glean", "4"): (None, None, "Keep coder 1: as glean C2."),
    ("glean", "5"): (None, None, "Keep coder 1: as glean C2."),
    ("globalhealthatlas", "3"): (None, None, "Keep coder 1: the 100 expert labels are unreleased data, which precede derived predictions."),
    ("lvcg", "6"): ([C], "no", "Accept coder 2: the paper's MIMIC path is absent rather than contradicted; code_missing, not divergence."),
    ("medcrp_cl", "1"): ([D, "other"], "no", "Accept coder 2: released code states alpha 5.0 as in the paper; only the checkpoint differs, so not a code divergence."),
    ("ophiuchus", "4"): ([C, W, D], None, "Accept coder 2: the complete evaluation code is absent, and code_missing takes precedence."),
}


def main():
    with open("claims_coder1.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        key = (r["folder"], r["claim_no"])
        if key in DECISIONS:
            blockers, div, _ = DECISIONS[key]
            if blockers is not None:
                r["blockers"] = ";".join(blockers)
                r["primary_blocker"] = next(b for b in PRECEDENCE if b in blockers)
            if div is not None:
                r["divergence"] = div
    with open("claims_resolved.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open("papers_coder1.csv", encoding="utf-8") as src, open("papers_resolved.csv", "w", encoding="utf-8", newline="") as dst:
        dst.write(src.read())
    print("resolved", len(DECISIONS), "disagreements")


if __name__ == "__main__":
    main()
