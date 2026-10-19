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
SPACE_ID = "kondratevakate/repro-came-grad-mathematical-audit"
TITLE = (
    "Reproduction: The Double Dilemma in Multi-Task Radiology Report "
    "Generation: A Gradient Dynamics Analysis and Solution"
)
PAGES = {
    "C1": "Claim 1: gradient interaction identity",
    "C2": "Claim 2: Stage 1 Pareto guarantee",
    "C3": "Claim 3: Stage 2 magnitude and covariance",
    "C4": "Claim 4: validity after adaptive fusion",
    "C5": "Claim 5: clinical efficacy",
}


def write_space_readme() -> None:
    content = f"""---
title: "{TITLE}"
emoji: "\\U0001F4D0"
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
tags:
 - trackio
 - trackio-logbook
 - open-experiment
 - icml2026-repro
 - paper-T7y2wavrFM
 - arxiv:2605.22635
---

# {TITLE}

An open experiment logbook, published with [Trackio](https://github.com/gradio-app/trackio).
"""
    (PROJECT / "logbook" / "README.md").write_text(content, encoding="utf-8")


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


def claim_markdown(claim: dict) -> str:
    evidence = claim["evidence"]
    if claim["id"] == "C1":
        return f"""**Verdict - VERIFIED FOR THE EXACT TWO-TASK IDENTITY.**

Across {evidence["random_trials"]} seeded 17-dimensional trials, the maximum
absolute identity error is `{evidence["identity_max_abs_error"]:.3e}`. Reversing
the interaction sign fails by as much as
`{evidence["sign_mutation_max_abs_error"]:.3f}`. Exact opposing gradients carry
individual energy `{evidence["opposing_individual_energy"]:.1f}` but joint
energy `{evidence["opposing_joint_energy"]:.1f}`.

This verifies the algebraic two-task mechanism, not the broader SDE narrative.
"""
    if claim["id"] == "C2":
        return f"""**Verdict - FALSIFIED AS AN UNCONDITIONAL GUARANTEE.**

For gradients `{evidence["gradients"]}`, `mu={evidence["mu"]}` and
`rho={evidence["rho"]}`, the trust region is `{evidence["trust_region"]}`.
Both the exact primal and declared dual recover
`u*={evidence["primal_u_star"]}`. Its task inner products are
`{evidence["task_inner_products"]}`, so the worst value is
`{evidence["worst_inner_product"]}` rather than non-negative.

The optimization remains well-defined; the unconditional strict Pareto claim
fails when task gradients admit no common descent direction in the trust
region.
"""
    if claim["id"] == "C3":
        return f"""**Verdict - PARTIALLY VERIFIED WITH EPSILON QUALIFICATION.**

Conditionally scaling seeded samples by `kappa={evidence["kappa"]}` changes covariance trace
by `{evidence["observed_trace_ratio"]:.16f}`, matching `kappa^2 =
{evidence["declared_trace_ratio"]}`. For a small nonzero direction and the
declared epsilon denominator, however, the target norm is
`{evidence["target_norm"]}` and the observed norm is
`{evidence["observed_norm"]}`, a
`{100 * evidence["relative_target_shortfall"]:.2f}%` shortfall.

This verifies the scalar covariance identity only. It does not reproduce the
stochastic covariance of the complete normalized CAME-Grad mechanism.
"""
    if claim["id"] == "C4":
        return f"""**Verdict - QUALIFIED SCOPE LIMITATION AFTER ADAPTIVE FUSION.**

The fixed two-task construction solves the declared Stage 1 dual at boundary
`alpha={evidence["dual_alpha_star"]}`; its right derivative is positive
(`{evidence["dual_right_derivative_at_zero"]:.6f}`), confirming the boundary
minimum. Stage 1 inner products are
`{evidence["stage1_inner_products"]}`, both positive. Stage 2 reaches its target
norm to numerical tolerance. With legal weights `{evidence["task_weights"]}`
and `nu={evidence["nu"]}`, Stage 3 produces final inner products
`{evidence["final_inner_products"]}` and reintroduces conflict with task 2.

This does not falsify the paper's formal glossary definition, which applies
geometric validity to the rectified Stage 1 direction. It qualifies the broad
algorithm-level narrative; the appendix itself acknowledges conflict
resurgence for large fusion coefficients.
"""
    return """**Verdict - INCONCLUSIVE, NOT EXECUTED.**

The eight-method MIMIC-CXR and IU X-Ray experiments were not rerun. The pinned
repository imports `modules.CAME_Grad`, but that file is absent, and its README
states that the core optimizer and model weights are temporarily withheld.
Paper tables are not treated as reproduction evidence.
"""


def main() -> int:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    create_logbook()
    metadata = lb.read_metadata(PROJECT)
    metadata.update(
        {
            "space_id": SPACE_ID,
            "tags": ["icml2026-repro", "paper-T7y2wavrFM"],
            "paper": {"arxiv_id": "2605.22635"},
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

    executive = """**Outcome.** This CPU-only audit evaluates five claims
against the paper equations and pinned public artifact. The exact two-task
interaction identity is verified. The unconditional Stage 1 Pareto guarantee
is falsified by an exact counterexample. Post-fusion validity receives a scope
qualification, Stage 2 is partially verified, and the clinical claim is
inconclusive.

| Item | Value |
| --- | --- |
| Prepared score | 6/10 |
| GPU | None |
| Medical data | None |
| Unit tests | 6 passing |
| Author commit | `79059e39060d13ef6b6cb2ea9ad7a5d8e2519c83` |

The conclusions are limited to the stated equations and released artifact.
No leaderboard or peer reproduction is used as evidence.

Provenance: [arXiv](https://arxiv.org/abs/2605.22635) and
[author repository](https://github.com/vpsg-research/CAME-Grad).
"""
    lb.add_markdown_cell(
        PROJECT, executive_slug, executive, title="Executive summary"
    )
    cell_id = lb.last_cell_id(PROJECT, page=executive_slug)
    lb.set_cell_pinned(PROJECT, cell_id, pinned=True, page=executive_slug)

    poster = """<div style="font-family:system-ui,Arial,sans-serif;max-width:900px;margin:0 auto;border:1px solid #ccd5dd;padding:20px;background:#fff;color:#18212a">
<h2 style="margin:0 0 6px;font-size:18px;border-bottom:3px solid #176b87;padding-bottom:8px">CAME-Grad: mathematical guarantee audit</h2>
<p style="font-size:12px;color:#52606d">ICML 2026 | T7y2wavrFM | arXiv 2605.22635v2 | CPU-only</p>
<table style="font-size:12px;border-collapse:collapse;width:100%">
<tr style="background:#eef3f6"><th style="padding:6px;text-align:left">Claim</th><th>Executable finding</th><th>Verdict</th></tr>
<tr><td style="padding:6px">Interaction identity</td><td>1,000 trials; opposing gradients cancel</td><td style="color:#067647;font-weight:700">VERIFIED</td></tr>
<tr><td style="padding:6px">Stage 1 guarantee</td><td>Exact optimum harms one incompatible task</td><td style="color:#b42318;font-weight:700">FALSIFIED</td></tr>
<tr><td style="padding:6px">Stage 2 scaling</td><td>Covariance holds; epsilon misses exact norm</td><td style="color:#7a5c13;font-weight:700">PARTIAL</td></tr>
<tr><td style="padding:6px">Stage 3 validity</td><td>Legal fusion reintroduces task conflict</td><td style="color:#7a5c13;font-weight:700">QUALIFIED</td></tr>
<tr><td style="padding:6px">Clinical efficacy</td><td>Core optimizer and weights withheld</td><td style="color:#7a5c13;font-weight:700">INCONCLUSIVE</td></tr>
</table><p style="font-size:12px;margin:12px 0 0">Prepared 6/10. No peer reproductions used.</p></div>"""
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
            code_paths=["repro_camegrad/audit_claims.py"],
            language="python",
        )

    conclusion = """The algebraic conflict mechanism is real, but the stronger
optimizer guarantees require qualifications. Stage 1 cannot force every task
inner product to be non-negative when no common descent direction exists, and
Stage 3 can undo a Pareto-valid Stage 1 direction by mixing back a
task-weighted joint gradient.

These mathematical findings do not determine clinical performance. Reproducing
the reported MIMIC-CXR and IU X-Ray averages requires the withheld optimizer,
weights or fresh eight-method training, and the declared clinical evaluator.
"""
    lb.add_markdown_cell(
        PROJECT, conclusion_slug, conclusion, title="Conclusion"
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
    index.extend([f"| [Conclusion](#/{conclusion_slug}) |", ""])
    (PROJECT / "logbook" / "pages" / "index.md").write_text(
        "\n".join(index), encoding="utf-8"
    )
    lb.write_site_files(PROJECT)
    write_space_readme()
    readme = (PROJECT / "logbook" / "README.md").read_text(encoding="utf-8")
    if "sdk: static" not in readme or "paper-T7y2wavrFM" not in readme:
        raise RuntimeError("Space README front matter is incomplete")
    print(PROJECT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
