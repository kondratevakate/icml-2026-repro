#!/usr/bin/env python3
"""Build the Trackio logbook from frozen CAME-Grad evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path

from trackio import logbook as lb


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent
PROJECT = CANDIDATE / ".trackio"
EVIDENCE = HERE / "evidence" / "claims_audit.json"
SPACE_ID = "kondratevakate/repro-came-grad-equation-audit"
TITLE = "Reproduction: CAME-Grad gradient dynamics audit"
PAGE_TITLES = {
    "C1": "Claim 1: three-stage optimizer mechanics",
    "C2": "Claim 2: SDE mechanism and convergence",
    "C3": "Claim 3: eight-backbone clinical gains",
    "C4": "Claim 4: multi-task optimizer comparison",
    "C5": "Claim 5: stage ablations",
}


def create_logbook() -> None:
    if PROJECT.exists():
        raise RuntimeError(f"Logbook already exists: {PROJECT}")
    old_cwd = Path.cwd()
    old_find = lb.find_project_dir
    try:
        os.chdir(CANDIDATE)
        lb.find_project_dir = lambda *args, **kwargs: None
        created = Path(lb.create_logbook(title=TITLE, space_id=SPACE_ID))
    finally:
        lb.find_project_dir = old_find
        os.chdir(old_cwd)
    if created.resolve() != PROJECT.resolve():
        raise RuntimeError(f"Unexpected logbook path: {created}")


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-T7y2wavrFM", "radiology"],
            "paper": {"arxiv_id": "2605.22635"},
            "private": False,
            "repos_public": True,
            "embed_content": False,
        }
    )
    lb.write_metadata(PROJECT, metadata)

    executive = lb.ensure_page(PROJECT, "Executive summary")
    claims = {
        cid: lb.ensure_page(PROJECT, title)
        for cid, title in PAGE_TITLES.items()
    }
    conclusion = lb.ensure_page(PROJECT, "Conclusion")

    mechanics = evidence["mechanics"]
    release = evidence["release"]
    source = evidence["source"]["published_table_arithmetic"]
    summary = f"""**Outcome.** CAME-Grad's printed equations pass an independent
mechanics audit, preparing **2/10 points**, exactly as forecast. The SDE causal
claims are not established by those equations, and all clinical claims remain
inconclusive because the official release omits the optimizer, dataset module,
weights, requirements, and runnable evaluation bundle.

| Item | Result |
| --- | --- |
| Random equation checks | {mechanics["repeats"]} |
| Maximum trust-region residual | `{mechanics["max_trust_region_boundary_residual"]:.2e}` |
| Fusion residual | `{mechanics["max_fusion_equation_residual"]:.1f}` |
| Unit/mutation tests | 5 passing |
| GPU / medical data | none |
| Prepared score | 2/10 |

Primary sources: [arXiv](https://arxiv.org/abs/2605.22635),
[OpenReview](https://openreview.net/forum?id=T7y2wavrFM), and the
[official repository](https://github.com/vpsg-research/CAME-Grad). No
leaderboard or third-party reproduction verdict contributes evidence.
"""
    lb.add_markdown_cell(PROJECT, executive, summary, title="Executive summary")
    summary_id = lb.last_cell_id(PROJECT, page=executive)
    if summary_id:
        lb.set_cell_pinned(PROJECT, summary_id, pinned=True, page=executive)

    poster = """<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;border-bottom:3px solid #a13b2a;padding-bottom:8px">CAME-Grad: independent equation and release audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | T7y2wavrFM | arXiv 2605.22635v2 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#f8efed"><th style="padding:6px;text-align:left">Claim</th><th>Evidence</th><th>Verdict</th></tr>
<tr><td style="padding:6px">C1 mechanics</td><td>500 equation checks; trust/norm/fusion invariants</td><td style="color:#137a46;font-weight:700">VERIFIED WITH QUALIFICATIONS</td></tr>
<tr><td style="padding:6px">C2 SDE mechanism</td><td>Equal norms produce covariance trace 0 vs 0.5</td><td style="color:#b36b00;font-weight:700">NOT ESTABLISHED</td></tr>
<tr><td style="padding:6px">C3-C5 experiments</td><td>Core optimizer and runnable artifacts withheld</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 2/10. Forecast calibrated. No peer verdicts used.</p></div>"""
    lb.add_figure_cell(
        PROJECT, executive, html=poster, title="Reproduction poster"
    )
    poster_id = lb.last_cell_id(PROJECT, page=executive)
    if poster_id:
        lb.set_cell_pinned(PROJECT, poster_id, pinned=True, page=executive)

    c1 = f"""**VERIFIED AT EQUATION LEVEL WITH QUALIFICATIONS — 2/2.**

Across {mechanics["repeats"]} random problems, the maximum trust-region
boundary residual is
`{mechanics["max_trust_region_boundary_residual"]:.3e}`, the Stage-2 norm
relative error is `{mechanics["max_stage2_magnitude_relative_error"]:.3e}`,
and the fusion residual is zero.

The printed method has two edge qualifications. Equation 10 is undefined when
the optimized dual gradient is zero; symmetric opposite gradients expose this
`0/0` case. The method calls Stage 3 “adaptive,” but Algorithm 1 and equation
13 use a fixed tuned `nu` and supply no data-dependent adaptation rule.
"""
    lb.add_markdown_cell(PROJECT, claims["C1"], c1, title=PAGE_TITLES["C1"])
    lb.add_code_cell(
        PROJECT,
        claims["C1"],
        output=json.dumps(mechanics, indent=2, sort_keys=True),
        title="C1 machine-readable evidence",
        code_paths=[
            "repro_came_grad/came_grad_equations.py",
            "repro_came_grad/audit_claims.py",
        ],
        language="python",
    )

    c2 = f"""**NOT ESTABLISHED BY THE PRINTED EQUATIONS — 0/2.**

Two update distributions were constructed with exactly equal norm one. The
constant distribution has covariance trace
`{mechanics["equal_norm_covariance_counterexample"]["constant_covariance_trace"]}`,
while the direction-varying distribution has trace
`{mechanics["equal_norm_covariance_counterexample"]["variable_covariance_trace"]}`.
Thus enforcing update magnitude does not determine gradient-noise covariance
`Sigma` or guarantee restored SDE diffusion. A trust-region bound also does
not alone prove global convergence or movement toward flatter minima.
"""
    lb.add_markdown_cell(PROJECT, claims["C2"], c2, title=PAGE_TITLES["C2"])

    mimic = source["mimic_cxr"]["mean_absolute_difference"]
    iu = source["iu_xray"]["mean_absolute_difference"]
    c3 = f"""**INCONCLUSIVE, NOT RERUN — 0/2.**

Parsing the published table cells yields mean CE differences of
`{mimic:.6f}` on MIMIC-CXR and `{iu:.6f}` on IU X-Ray, consistent with the
paper's rounded 2.3 and 1.9 percentage-point summaries. This is source
arithmetic, not reproduction.

Eight paired training pipelines cannot run from the release: the optimizer,
dataset implementation, processed splits, weights, and complete dependency
specification are missing.
"""
    lb.add_markdown_cell(PROJECT, claims["C3"], c3, title=PAGE_TITLES["C3"])

    lb.add_markdown_cell(
        PROJECT,
        claims["C4"],
        """**INCONCLUSIVE, NOT RERUN — 0/2.**

The seven comparator configurations and matching REVTAF artifacts are absent.
No result is inferred from Table 3 alone.
""",
        title=PAGE_TITLES["C4"],
    )
    lb.add_markdown_cell(
        PROJECT,
        claims["C5"],
        """**INCONCLUSIVE, NOT RERUN — 0/2.**

No official ablation runner or optimizer implementation is released. Synthetic
equation checks cannot attribute radiology-report performance to S1, S2, or
S3.
""",
        title=PAGE_TITLES["C5"],
    )
    lb.add_code_cell(
        PROJECT,
        claims["C5"],
        output=json.dumps(release, indent=2, sort_keys=True),
        title="Official release inventory",
        code_paths=["repro_came_grad/audit_claims.py"],
        language="python",
    )

    lb.add_markdown_cell(
        PROJECT,
        conclusion,
        """The independent result supports the algebraic three-stage mechanism
only. It also identifies a zero-denominator edge case, a fixed rather than
adaptive fusion coefficient, and a counterexample to inferring diffusion
covariance from update magnitude.

The clinically important claims remain open. A future canonical rerun requires
the exact optimizer, eight model integrations, processed data/splits,
checkpoints or seeds, CheXbert weights, and comparator/ablation configurations.
Until then, paper-table concordance must remain separate from experimental
reproduction.
""",
        title="Conclusion",
    )

    pages_root = PROJECT / "logbook" / "pages"
    lines = [
        f"# {TITLE}",
        "",
        lb.TOC_HEADING,
        "",
        lb.TOC_HEADER,
        lb.TOC_SEP,
        f"| [Executive summary](#/{executive}) |",
    ]
    lines.extend(
        f"| [{PAGE_TITLES[cid]}](#/{slug}) |" for cid, slug in claims.items()
    )
    lines.extend([f"| [Conclusion](#/{conclusion}) |", ""])
    (pages_root / "index.md").write_text("\n".join(lines), encoding="utf-8")
    lb.write_site_files(PROJECT)
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
