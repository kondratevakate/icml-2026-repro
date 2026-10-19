#!/usr/bin/env python3
"""Build the Trackio logbook from frozen StCP evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = CANDIDATE_DIR / ".trackio"
EVIDENCE_PATH = SCRIPT_DIR / "evidence" / "claims_audit.json"
SPACE_ID = "kondratevakate/repro-stcp-transduction"
TITLE = (
    "Reproduction: Stable Localized Conformal Prediction via Transduction"
)
PAGE_TITLES = {
    "C1": "Claim 1: set-stability criterion",
    "C2": "Claim 2: marginal-coverage characterization",
    "C3": "Claim 3: stability-rate theorem",
    "C4": "Claim 4: selected-lambda coverage",
    "C5": "Claim 5: medical stability gains",
    "C6": "Claim 6: LogAbs stability gains",
}


def create_isolated_logbook() -> Path:
    if PROJECT_DIR.exists():
        raise RuntimeError(f"Logbook already exists: {PROJECT_DIR}")
    original_cwd = Path.cwd()
    original_find = lb.find_project_dir
    try:
        os.chdir(CANDIDATE_DIR)
        lb.find_project_dir = lambda *args, **kwargs: None
        project = lb.create_logbook(title=TITLE, space_id=SPACE_ID)
    finally:
        lb.find_project_dir = original_find
        os.chdir(original_cwd)
    return Path(project)


def claim_body(claim_id: str, claim: dict) -> str:
    if claim_id == "C1":
        data = claim["evidence"]
        return f"""**Verdict - {claim["verdict"]}.**

The independent experiment generated {data["repeats"]} calibration datasets
and {data["test_points_per_construction"]} test covariates per construction.
The variance of conditional mean set size was
`{data["stability_variance"]:.6f}`. Raw per-test variance was
`{data["raw_per_test_variance_mutation"]:.6f}` because it additionally
contains `{data["mean_within_construction_variance"]:.6f}` of within-
construction test-point noise.

The law-of-total-variance residual was
`{data["law_total_variance_error"]:.3e}`. This verifies the paper's criterion
and falsifies the mutation that substitutes raw set-size variance.
"""
    if claim_id == "C2":
        data = claim["evidence"]
        return f"""**Verdict - {claim["verdict"]}.**

Theorem 4.2 defines `delta_S` from a random source-based estimator, but its
proof bounds an unconditional expectation by the realized `delta_S`. For a
Uniform[0,2] target score and a source quantile equal to 1 or 2 with equal
probability, the unconditional coverage deviation is
`{data["unconditional_coverage_deviation"]:.2f}`. On the positive-probability
realization where the source quantile is 1, `delta_S=0`, so the proof's right
side for this step is zero.

Replacing realized `delta_S` with `E[delta_S]` repairs this specific step.
This is a concrete proof counterexample, not a claim that every corrected
version of the theorem is false.
"""
    if claim_id == "C3":
        data = claim["evidence"]
        return f"""**Verdict - {claim["verdict"]}.**

An independent Gaussian surrogate reproduced the stated
`m^-1 + [n(1+lambda)^2]^-1` algebra with maximum relative error
`{data["max_relative_error"]:.4f}` over {data["repeats"]:,} repetitions.
Ignoring lambda removed the variance gain, as expected.

The theorem proof itself explicitly invokes local second-order smoothness not
listed in its assumptions. It also promotes an `O_p` fixed-point rate to a
second-moment rate without a tail or uniform-integrability argument.
Therefore the rate is supported as algebra but not established under the
theorem's stated assumptions.
"""
    if claim_id == "C4":
        data = claim["evidence"]
        return f"""**Verdict - {claim["verdict"]}.**

For iid continuous exchangeable scores, coverage of the `k`-th calibration
order statistic is exactly `k/(n+1)`. The proof uses
`k=ceil(n p)` and incorrectly asserts `k/(n+1) >= p`.

At the paper's `n={data["n"]}`, `alpha={data["alpha"]}`, and
`alpha_tol={data["alpha_tol"]}`, the lower endpoint has `p=0.88`,
`k=27`, and exact coverage `27/31 =
{data["exact_coverage_at_q_L"]:.6f}`, below the claimed 0.88. The audit found
{data["exhaustive_grid_failure_count"]} failures over its finite `(n,p)`
grid. Replacing `ceil(n p)` by `ceil((n+1)p)` repairs this lower-endpoint
arithmetic in the tested configuration.
"""
    if claim_id == "C5":
        return """**Verdict - PENDING FRESH MEDICAL RUN.**

DermaMNIST and TissueMNIST data plus author checkpoints are present in the
pinned repository. They are not treated as reproduced results. A fresh
multi-repeat execution is still required before assigning a verdict to the
medical stability claim. No patient-level or image-level data are included
in this logbook.
"""
    return f"""**Verdict - {claim["verdict"]}.**

The pinned author LogAbs pipeline completed a local end-to-end smoke run, but
one repeat and reduced training are insufficient to estimate stability.
The publishable runner labels such output `NO CLAIM VERDICT`; only a fresh
canonical 50-repeat run is eligible for the final claim verdict.

The paper defines Std with denominator `R-1`, whereas the released synthetic
aggregator uses NumPy's population denominator `R`. The runner reports both
the released and sample-corrected values.
"""


def main() -> int:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    project = create_isolated_logbook()
    if project.resolve() != PROJECT_DIR.resolve():
        raise RuntimeError(f"Unexpected Trackio project: {project}")

    metadata = lb.read_metadata(PROJECT_DIR)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-lSMTccAN61"],
            "paper": {"arxiv_id": "2605.01452"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT_DIR, metadata)

    executive_slug = lb.ensure_page(PROJECT_DIR, "Executive summary")
    claim_slugs = {
        claim_id: lb.ensure_page(PROJECT_DIR, title)
        for claim_id, title in PAGE_TITLES.items()
    }
    conclusion_slug = lb.ensure_page(PROJECT_DIR, "Conclusion")

    executive = """**Outcome.** Six claims from arXiv `2605.01452v1` were
audited without leaderboard or peer-reproduction evidence. C1 is
**VERIFIED**. C4 is **FALSIFIED AS WRITTEN** by an exact exchangeable-score
counterexample. C2 and C3 are **NOT ESTABLISHED AS WRITTEN** because their
proofs require a corrected random-error term or unstated regularity. C5 and
C6 remain pending fresh canonical experiments.

## Scope and cost

| Item | Value |
| --- | --- |
| GPU | None |
| Protected medical data | None |
| Independent stability constructions | 4,000 |
| Rate-surrogate repetitions | 200,000 |
| Unit tests | 5 passing |
| Prepared score | 6/12 |
| Conservative expected score | 6/12 |

The paper is medically relevant through its DermaMNIST dermatoscopy and
TissueMNIST kidney-cortex microscopy experiments.

Provenance: [OpenReview](https://openreview.net/forum?id=lSMTccAN61),
[arXiv](https://arxiv.org/abs/2605.01452), and the pinned
[author repository](https://github.com/OswinMin/SLCP) at commit
`84118ddf19efe211250a5f024c1be5a3c8f47b4b`.
"""
    lb.add_markdown_cell(
        PROJECT_DIR, executive_slug, executive, title="Executive summary"
    )
    executive_id = lb.last_cell_id(PROJECT_DIR, page=executive_slug)
    if executive_id:
        lb.set_cell_pinned(
            PROJECT_DIR, executive_id, pinned=True, page=executive_slug
        )

    poster = """<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #cfd7df;padding:20px;background:#fff;color:#17202a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">StCP: independent theorem and stability audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | lSMTccAN61 | arXiv 2605.01452v1 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Independent evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1 set stability</td><td>Total-variance audit, 4,000 constructions</td><td style="color:#137a46;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">C2 marginal bound</td><td>Random-delta proof counterexample</td><td style="color:#7a5c13;font-weight:700">NOT ESTABLISHED</td></tr>
<tr><td style="padding:6px">C3 variance rate</td><td>Rate sanity passes; assumptions and moment gap remain</td><td style="color:#7a5c13;font-weight:700">NOT ESTABLISHED</td></tr>
<tr><td style="padding:6px">C4 selected lambda</td><td>27/31 &lt; 0.88 under continuous exchangeability</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">C5-C6 experiments</td><td>Fresh canonical runs pending</td><td style="color:#7a5c13;font-weight:700">PENDING</td></tr>
</table>
<p style="font-size:12px;margin:12px 0 0">Prepared 6/12. No peer reproductions used.</p>
</div>"""
    lb.add_figure_cell(
        PROJECT_DIR,
        executive_slug,
        html=poster,
        title="Reproduction poster",
    )
    poster_id = lb.last_cell_id(PROJECT_DIR, page=executive_slug)
    if poster_id:
        lb.set_cell_pinned(
            PROJECT_DIR, poster_id, pinned=True, page=executive_slug
        )

    for claim_id, slug in claim_slugs.items():
        claim = evidence["claims"][claim_id]
        lb.add_markdown_cell(
            PROJECT_DIR,
            slug,
            claim_body(claim_id, claim),
            title=PAGE_TITLES[claim_id],
        )
        lb.add_code_cell(
            PROJECT_DIR,
            slug,
            output=json.dumps(claim, indent=2, sort_keys=True),
            title=f"{claim_id} machine-readable evidence",
            code_paths=["repro_stcp/audit_claims.py"],
            language="python",
        )

    conclusion = """The current publishable result prepares six conservative
points without relying on another participant's outcome. The set-stability
definition behaves as claimed, and the selected-lambda finite-sample theorem
fails at its lower endpoint because `ceil(n p)/(n+1)` need not exceed `p`.

The marginal and stability-rate theorems need additional qualifications
before they can be considered established. The medical and full LogAbs claims
remain open until fresh canonical executions complete.
"""
    lb.add_markdown_cell(
        PROJECT_DIR, conclusion_slug, conclusion, title="Conclusion"
    )

    pages_root = PROJECT_DIR / "logbook" / "pages"
    index_lines = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive_slug}) |",
    ]
    index_lines.extend(
        f"| [{PAGE_TITLES[claim_id]}](#/{claim_slugs[claim_id]}) |"
        for claim_id in PAGE_TITLES
    )
    index_lines.extend([f"| [Conclusion](#/{conclusion_slug}) |", ""])
    (pages_root / "index.md").write_text(
        "\n".join(index_lines), encoding="utf-8"
    )
    lb.write_site_files(PROJECT_DIR)
    print(PROJECT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
