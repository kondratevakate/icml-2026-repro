#!/usr/bin/env python3
"""Build the canonical Trackio logbook from frozen PEQ-Net evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_DIR = SCRIPT_DIR.parent
PROJECT_DIR = CANDIDATE_DIR / ".trackio"
EVIDENCE_PATH = SCRIPT_DIR / "evidence" / "claims_audit.json"
CLAIMS_PATH = SCRIPT_DIR / "claims.json"
SPACE_ID = (
    "kondratevakate/"
    "repro-smooth-multi-policy-causal-effect-estimation-in-longitudinal-settings"
)
TITLE = (
    "Reproduction: Smooth Multi-Policy Causal Effect Estimation in "
    "Longitudinal Settings"
)

PAGE_TITLES = {
    "C1": "Claim 1: limited-confounding RMSE",
    "C2": "Claim 2: expanded-confounding RMSE",
    "C3": "Claim 3: dynamic-policy RMSE",
    "C4": "Claim 4: policy embedding pipeline",
    "C5": "Claim 5: remainder Lipschitz theorem",
    "C6": "Claim 6: MIMIC-IV sepsis result",
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


def empirical_body(
    claim_text: str, claim: dict, *, expanded: bool = False
) -> str:
    dgp_note = (
        "The expanded recurrence is specified, but it delegates treatment and "
        "outcome generation to the same undefined limited-DGP equations."
        if expanded
        else "The printed lag sum starts at `i=1` but contains "
        "`(-1)^i/(1-i)`, so its first coefficient divides by zero."
    )
    return f"""**Paper claim.** "{claim_text}"

**Verdict - {claim["verdict"]}.**

The table was not treated as self-validating evidence. No author PEQ-Net code
or author DeepLTMLE implementation was found in the paper artifacts. The run
also requires processed MIMIC-III trajectories, 500 training epochs, selected
hyperparameters, and 20 independent seeds.

{dgp_note}

The paper's numerical values are therefore preserved only as provenance. A
faithful independent comparison cannot be executed from the released
artifacts, so no performance conclusion is claimed.

Machine-readable readiness:

```json
{json.dumps(claim["blockers"], indent=2, sort_keys=True)}
```
"""


def claim_body(claim_id: str, claim_text: str, claim: dict) -> str:
    if claim_id == "C1":
        return empirical_body(claim_text, claim)
    if claim_id == "C2":
        return empirical_body(claim_text, claim, expanded=True)
    if claim_id == "C3":
        return empirical_body(claim_text, claim)
    if claim_id == "C4":
        embedding = claim["embedding"]
        checks = embedding["checks"]
        return f"""**Paper claim.** "{claim_text}"

**Verdict - {claim["verdict"]} WITH QUALIFICATION.**

An independent implementation applied five threshold policies to
{embedding["n_histories"]} fixed synthetic histories, formed history-action
samples, computed biased empirical Gaussian-kernel MMD, and ran metric MDS on
the resulting dissimilarity matrix.

| Diagnostic | Result |
| --- | ---: |
| Distance-order rank correlation | {embedding["distance_ordering_correlation"]:.3f} |
| Relative MDS stress against supplied `MMD^2` | {embedding["relative_stress_against_mmd2"]:.3f} |
| Max error if embedding distance is called exact `MMD` | {embedding["max_error_if_treated_as_exact_mmd"]:.3f} |
| Finite, symmetric, zero diagonal | {checks["all_values_finite"] and checks["symmetric_dissimilarity"] and checks["zero_diagonal"]} |

The policy-distance ordering is perfectly preserved in this controlled run,
and the source algorithm independently confirms the reverse-time policy-tail
encoder and shared-Q conditioning path.

**Qualification.** The method defines the MDS input as `MMD^2` and promises
approximate preservation. The theorem proof later asserts exact equality
between embedding distance and `MMD`. The reproduced pipeline supports the
architecture claim but not that exact mathematical identity.
"""
    if claim_id == "C5":
        counter = claim["targeting_counterexample"]
        return f"""**Paper claim.** "{claim_text}"

**Verdict - {claim["verdict"]}.**

The appendix proof has three independent failures as a uniform Lipschitz
argument:

1. The subsequence-MMD lemma defines `C` as the numerator/denominator ratio
   after fixing each policy pair. That proves only a pair-specific tautology,
   not one finite constant uniform over policies.
2. The method specifies approximate metric-MDS preservation of `MMD^2`; the
   proof substitutes exact equality to `MMD`.
3. Strict positivity and bounded LTMLE fluctuations do not imply that the
   targeting map is Lipschitz in the initial Q estimates.

For the third point, use identical initial values `Q_i=Q_j=0.5`, clever
covariate one, strictly positive treatment probability 0.5, and bounded
policy-specific fluctuations `epsilon_i=0.8`, `epsilon_j=-0.8`. Logistic
targeting gives `{counter["targeted_q_values"][0]:.6f}` and
`{counter["targeted_q_values"][1]:.6f}`. The initial difference and the claimed
right-hand side are zero, while the targeted difference is
`{counter["targeted_difference"]:.6f}`.

The theorem may be repairable with a uniform marginal-to-trajectory MMD bridge,
an exact or controlled embedding-distortion bound, and a Lipschitz condition
on the policy-specific targeting map. Those assumptions are absent.
"""
    return f"""**Paper claim.** "{claim_text}"

**Verdict - {claim["verdict"]}.**

The paper reports 999 adult ICU patients satisfying Sepsis-3 criteria,
hypotension (`MAP <= 65 mmHg`), and vasopressor initiation within 24 hours. It
does not release cohort SQL, MIMIC item IDs, missing-data rules, preprocessing,
model code, checkpoints, or random seeds.

The local protected-data download is not complete. Reconstructing a different
cohort and comparing it to the paper figure would not verify the anchored
claim. This page will remain inconclusive until the exact cohort path is
auditable. No patient-level data are included in this logbook.

```json
{json.dumps(claim["blockers"], indent=2, sort_keys=True)}
```
"""


def main() -> int:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    claims = {
        item["id"]: item["text"]
        for item in json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    }
    project = create_isolated_logbook()
    if project.resolve() != PROJECT_DIR.resolve():
        raise RuntimeError(f"Trackio created unexpected project: {project}")

    metadata = lb.read_metadata(PROJECT_DIR)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-bIcz7bIZSo"],
            "paper": {"arxiv_id": "2605.14284"},
            "private": False,
            "repos_public": False,
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

    executive = """**Outcome.** Six claims were audited against arXiv
`2605.14284v2` and independent CPU checks. C4 is **VERIFIED with
qualification**: the MMD-to-MDS policy representation works and preserves
distance ordering, but does not provide the exact identity used later in the
proof. C5 is **FALSIFIED as a uniform Lipschitz guarantee** under the stated
assumptions. C1-C3 and C6 are **INCONCLUSIVE** because code and exact protected
data pipelines are unavailable.

## Scope and cost

| Item | Value |
| --- | --- |
| GPU | None |
| Protected medical data | None used or published |
| Independent policies | 5 |
| Synthetic histories | 240 |
| Unit tests | 4 passing |
| Prepared score | 4/12 |
| Conservative expected score | 4/12 |

The paper is medically relevant: its motivating and real-data setting is
longitudinal treatment-policy evaluation in MIMIC-III/MIMIC-IV.
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
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">PEQ-Net: independent embedding and theorem audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | bIcz7bIZSo | arXiv 2605.14284v2 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claims</th><th>Independent evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1-C3 empirical RMSE</td><td>No author code; protected input; printed DGP singular at i=1</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
<tr><td style="padding:6px">C4 policy embedding</td><td>5 policies; independent MMD + metric MDS; rank correlation 1.0</td><td style="color:#137a46;font-weight:700">VERIFIED*</td></tr>
<tr><td style="padding:6px">C5 remainder theorem</td><td>Pair-specific constant, MDS mismatch, bounded-targeting counterexample</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">C6 MIMIC-IV case</td><td>No cohort SQL/item map; complete restricted data pending</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table>
<p style="font-size:12px;margin:12px 0 0">*Pipeline verified; exact MDS=MMD identity is not. Prepared 4/12.</p>
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
            claim_body(claim_id, claims[claim_id], claim),
            title=PAGE_TITLES[claim_id],
        )
        lb.add_code_cell(
            PROJECT_DIR,
            slug,
            output=json.dumps(claim, indent=2, sort_keys=True),
            title=f"{claim_id} machine-readable evidence",
            code_paths=["repro_peqnet/audit_claims.py"],
            language="python",
        )

    conclusion = """The publishable result is narrow and reproducible. The
policy representation can be rebuilt locally and preserves the intended
ordering on a controlled policy family. The exact identity imported into the
theorem proof does not follow from metric MDS, and the theorem's targeting-map
step has a direct bounded counterexample.

The empirical tables and MIMIC-IV trend are not assigned a verdict from the
paper's own numbers. They remain inconclusive until the author implementation
or an exact independent restricted-data pipeline is available.
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
