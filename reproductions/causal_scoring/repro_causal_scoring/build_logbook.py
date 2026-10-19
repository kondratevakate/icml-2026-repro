#!/usr/bin/env python3
"""Build the Trackio logbook from frozen causal-scoring evidence."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "claims_audit.json"
SPACE_ID = "kondratevakate/repro-tailored-scoring-rules-causal-inference"
TITLE = (
    "Reproduction: Tailoring Strictly Proper Scoring Rules for Downstream "
    "Tasks: An Application to Causal Inference"
)
PAGES = {
    "C1": "Claim 1: IPW MSE bound",
    "C2": "Claim 2: task curvature",
    "C3": "Claim 3: strictly proper loss",
    "C4": "Claim 4: quartic canonical mapping",
    "C5": "Claim 5: Kang-Schafer mechanism",
    "C6": "Claim 6: broad benchmark superiority",
}


def create_logbook() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"Logbook already exists: {PROJECT}")
    original_cwd = Path.cwd()
    original_find = lb.find_project_dir
    try:
        os.chdir(CANDIDATE)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = original_find
        os.chdir(original_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected logbook path: {created}")


def write_space_readme() -> None:
    content = f"""---
title: "{TITLE}"
emoji: "\\U0001F4C8"
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-JTwryHNicJ
 - arxiv:2606.03332
---

# {TITLE}

An open experiment logbook, published with [Trackio](https://github.com/gradio-app/trackio).

The complete CPU-only reproduction bundle is included in
[`repro_causal_scoring/`](./repro_causal_scoring/README.md).
"""
    (PROJECT / "logbook" / "README.md").write_text(content, encoding="utf-8")


def copy_reproduction_bundle() -> None:
    destination = PROJECT / "logbook" / "repro_causal_scoring"
    destination.mkdir()
    for relative in (
        "README.md",
        "requirements.txt",
        "claims_list.json",
        "audit_claims.py",
        "prepare_official.py",
        "evidence/claims_audit.json",
        "tests/test_audit_claims.py",
    ):
        source = HERE / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def claim_markdown(claim: dict) -> str:
    evidence = claim["evidence"]
    if claim["id"] == "C1":
        return f"""**Verdict - PARTIALLY VERIFIED, CONDITIONAL ON FIXED PROPENSITY.**

The theorem-specific pointwise bias inequality holds in 5,000 deterministic
trials; its maximum residual is
`{evidence["conditional_proof_max_bias_residual"]:.6g}`. A separate sanity
check of the generic `Var(A+B) <= 2 Var(A) + 2 Var(B)` step has maximum
residual `{evidence["generic_sum_variance_inequality_max_residual"]:.6g}`.
Weakening the bias constant produces
`{evidence["weakened_constant_mutation_violations"]}` violations.

The scope step is incomplete for ordinary K-fold cross-fitting. Exact
enumeration of four two-fold states gives contribution covariance
`{evidence["crossfit_contribution_covariance"]}` and mean-variance residual
`{evidence["iid_formula_residual"]}` relative to `Var(Z)/N`. The proof is valid
conditional on a fixed propensity function, but cross-fitting alone does not
make all fold contributions unconditionally independent.

Paper anchors: Theorem 3.5 at source lines 415-434; cross-fitting argument at
lines 1267-1269.
"""
    if claim["id"] == "C2":
        return f"""**Verdict - VERIFIED.**

SymPy simplifies the difference between the task-divergence Hessian and the
declared curvature to `{evidence["symbolic_difference"]}`. A
`{evidence["grid_size"]}`-point, 60-digit boundary-aware grid has maximum
relative error `{evidence["max_relative_error"]:.3e}`. Changing a cubic
boundary term to quadratic gives maximum relative error
`{evidence["mutation_max_relative_error"]:.3f}`.

Paper anchor: Equation 15, source lines 449-453.
"""
    if claim["id"] == "C3":
        return f"""**Verdict - VERIFIED.**

Both proper-loss differential residuals simplify exactly to zero. The declared
weight is positive and equals `{evidence["minimum_declared_weight_at_half"]}`
at its symmetry point. Across `{evidence["truth_grid_size"]}` truths and
`{evidence["prediction_grid_size"]}` candidate predictions, maximum argmin
error is `{evidence["max_argmin_abs_error"]:.3e}`. Swapping label-specific
partial losses moves the minimizer by
`{evidence["label_swap_mutation_min_mean_argmin_error"]:.3f}` on average.

Paper anchor: Definition 3.7, source lines 456-463.
"""
    if claim["id"] == "C4":
        return f"""**Verdict - PARTIALLY VERIFIED; ASYMPTOTIC NO-VANISH INTERPRETATION FALSIFIED.**

The integrated-link derivative and quartic residual both simplify to zero.
The independent inverse and largest-root implementations agree within
`{evidence["quartic_root_vs_inverse_link_max_abs_error"]:.3e}` over
`{evidence["logit_grid_size"]}` logits. The mapping is strictly monotone and
symmetric.

The canonical gradient is `p-y`; it is nonzero for every finite logit and
binary label. However, the paper's statement that it cannot vanish is false
in the standard asymptotic vanishing-gradient sense. At `z=10^6`, `y=1`, its
magnitude is already
`{evidence["canonical_gradient_abs_at_z_1e6_y1"]:.6f}` and tends to zero as
`z` grows. In contrast, the sigmoid-link mutation reaches gradient magnitude
`{evidence["mismatched_sigmoid_max_abs_gradient"]:.3e}`.

Paper anchors: source lines 487-506.
"""
    if claim["id"] == "C5":
        metrics = evidence["metrics"]
        return f"""**Verdict - PARTIALLY VERIFIED IN A SCOPED INDEPENDENT SIMULATION.**

All `{evidence["expected_fit_count_per_method"] * 4}` propensity fits converged
in a 30-seed, 10-fold panel. On the paper's nonlinear observed features,
log-loss IPW RMSE is `{metrics["observed_log"]["rmse"]:.3f}` versus
`{metrics["observed_tailored"]["rmse"]:.3f}` for the tailored pair, a
`{evidence["observed_rmse_ratio_log_over_tailored"]:.2f}x` ratio; tailored wins
`{100 * evidence["observed_tailored_abs_error_win_rate"]:.1f}%` of seeds.

Replacing observed features with the correctly specified latent variables
shrinks the ratio to `{evidence["latent_rmse_ratio_log_over_tailored"]:.2f}x`,
a specification ablation supporting the misspecification mechanism.
Standard-normal
outcome noise is assumed because the paper does not specify its distribution.

Paper anchors: DGP at source lines 794-797; protocol at lines 545-547.
"""
    return """**Verdict - INCONCLUSIVE, NOT EXECUTED.**

The complete IHDP, Jobs, and 32-DGP ACIC evaluation was not rerun. The paper
does not release code, per-run inputs, random seeds, split assignments, or the
full tuning grids. Values printed in figures and tables are not treated as
reproduction evidence.
"""


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-JTwryHNicJ"],
            "paper": {"arxiv_id": "2606.03332"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive_slug = lb.ensure_page(PROJECT, "Executive summary")
    claim_slugs = {
        cid: lb.ensure_page(PROJECT, title) for cid, title in PAGES.items()
    }
    conclusion_slug = lb.ensure_page(PROJECT, "Conclusion")

    executive = """**Outcome.** This CPU-only reproduction independently audits
six claims using exact algebra, high-precision grids, destructive mutations,
and a 30-seed 10-fold synthetic panel.

| Claim | Result | Score |
| --- | --- | ---: |
| IPW MSE bound | Partial: fixed-propensity scope | 1/2 |
| Task curvature | Verified | 2/2 |
| Strictly proper loss | Verified | 2/2 |
| Quartic mapping | Partial: no-vanish clause false | 1/2 |
| Kang-Schafer mechanism | Partial independent simulation | 1/2 |
| Broad benchmarks | Inconclusive | 0/2 |
| **Prepared** |  | **7/12** |

No leaderboard or peer reproduction is used as evidence.

Provenance: [arXiv](https://arxiv.org/abs/2606.03332) and
[OpenReview](https://openreview.net/forum?id=JTwryHNicJ).
"""
    lb.add_markdown_cell(
        PROJECT, executive_slug, executive, title="Executive summary"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    poster = """<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">Tailored causal scoring rules: independent audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | JTwryHNicJ | arXiv 2606.03332v1 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">IPW bound</td><td>Closes conditionally; cross-fold dependence omitted</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">Curvature</td><td>Exact symbolic identity plus mutation</td><td style="color:#067647;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Proper loss</td><td>Exact derivatives and exhaustive risk minima</td><td style="color:#067647;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Canonical map</td><td>Quartic holds; no-vanish clause false</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">Kang-Schafer</td><td>7.10x RMSE ratio; 93.3% seed wins</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">Broad benchmarks</td><td>Unreleased execution details</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 7/12. Six tests pass.</p></div>"""
    lb.add_figure_cell(
        PROJECT, executive_slug, html=poster, title="Reproduction poster"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    for cid, slug in claim_slugs.items():
        claim = next(item for item in evidence["claims"] if item["id"] == cid)
        lb.add_markdown_cell(PROJECT, slug, claim_markdown(claim), title=PAGES[cid])
        lb.add_code_cell(
            PROJECT,
            slug,
            output=json.dumps(claim, indent=2, sort_keys=True),
            title=f"{cid} machine-readable evidence",
            code_paths=["audit_claims.py"],
            language="python",
        )

    boundary = """## Evidence boundary

- C1 does not dispute the pointwise IPW inequalities. It
limits the paper's cross-fitting justification for unconditional variance.
- C4 verifies the canonical construction. The gradient is nonzero at every
finite logit; only the asymptotic no-vanishing interpretation is falsified.
- C5 uses the stated DGP with an explicitly recorded standard-normal noise
assumption, frozen seeds, one regularization value, and IPW only. It is not an
exact reconstruction of the paper table.
- C6 remains inconclusive. No result is inferred from paper-rendered numbers.
"""

    conclusion = """The paper's central closed-form construction is
mathematically sound: the task curvature, proper loss, and quartic canonical
mapping survive independent symbolic, numerical, and mutation tests. Two
wording/proof-scope qualifications remain, and the independent Kang-Schafer
panel supports the intended weak-overlap mechanism without claiming exact
table replication.
"""
    lb.add_markdown_cell(
        PROJECT, conclusion_slug, f"{conclusion}\n\n{boundary}", title="Conclusion"
    )

    index = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive_slug}) |",
    ]
    index.extend(f"| [{PAGES[cid]}](#/{claim_slugs[cid]}) |" for cid in PAGES)
    index.extend(
        [
            f"| [Conclusion](#/{conclusion_slug}) |",
            "",
        ]
    )
    (PROJECT / "logbook" / "pages" / "index.md").write_text(
        "\n".join(index), encoding="utf-8"
    )
    copy_reproduction_bundle()
    lb.write_site_files(PROJECT)
    write_space_readme()
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
